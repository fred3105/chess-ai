#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <chrono>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <iomanip>

// ONNX Runtime includes
#include <onnxruntime_cxx_api.h>

namespace py = pybind11;

// Piece values for deterministic evaluation
const int PIECE_VALUES[] = {0, 100, 320, 330, 500, 900, 20000};

// Position evaluation tables
const int PAWN_TABLE[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

const int KNIGHT_TABLE[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

enum class EvaluationMode {
    DETERMINISTIC,
    NNUE
};

struct Move {
    int from_square;
    int to_square;
    int piece;
    int captured;
    bool promotion;
    int promotion_piece;
    
    Move() : from_square(-1), to_square(-1), piece(0), captured(0), promotion(false), promotion_piece(0) {}
    Move(int from, int to, int p = 0, int cap = 0, bool prom = false, int prom_piece = 0) 
        : from_square(from), to_square(to), piece(p), captured(cap), promotion(prom), promotion_piece(prom_piece) {}
        
    std::string to_uci() const {
        if (from_square < 0 || to_square < 0) return "0000";
        
        char from_file = 'a' + (from_square % 8);
        char from_rank = '1' + (from_square / 8);
        char to_file = 'a' + (to_square % 8);
        char to_rank = '1' + (to_square / 8);
        
        std::string move = std::string(1, from_file) + from_rank + to_file + to_rank;
        if (promotion) {
            char prom_chars[] = "nbrq";
            if (promotion_piece >= 2 && promotion_piece <= 5) {
                move += prom_chars[abs(promotion_piece) - 2];
            }
        }
        return move;
    }
};

// Transposition table entry types
enum class EntryType {
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND
};

// Transposition table entry
struct TTEntry {
    uint64_t zobrist_hash;
    double score;
    int depth;
    EntryType type;
    Move best_move;
    
    TTEntry() : zobrist_hash(0), score(0.0), depth(-1), type(EntryType::EXACT) {}
};

// Zobrist hash class for position caching
class ZobristHash {
private:
    std::array<std::array<uint64_t, 12>, 64> piece_keys;  // [square][piece_type]
    uint64_t side_key;
    std::array<uint64_t, 16> castling_keys;  // castling rights
    std::array<uint64_t, 64> en_passant_keys;  // en passant squares
    
public:
    ZobristHash() {
        std::mt19937_64 rng(12345);  // Fixed seed for reproducibility
        std::uniform_int_distribution<uint64_t> dist;
        
        // Initialize piece keys
        for (int sq = 0; sq < 64; sq++) {
            for (int piece = 0; piece < 12; piece++) {
                piece_keys[sq][piece] = dist(rng);
            }
        }
        
        // Initialize other keys
        side_key = dist(rng);
        for (int i = 0; i < 16; i++) {
            castling_keys[i] = dist(rng);
        }
        for (int i = 0; i < 64; i++) {
            en_passant_keys[i] = dist(rng);
        }
    }
    
    int piece_to_index(int piece) const {
        if (piece == 0) return -1;
        int type = abs(piece) - 1;  // 0-5 for pawn-king
        return piece > 0 ? type : type + 6;
    }
    
    uint64_t hash_position(const std::array<int, 64>& board, bool white_to_move, 
                          int castling_rights, int en_passant_square) const {
        uint64_t hash = 0;
        
        // Hash pieces
        for (int sq = 0; sq < 64; sq++) {
            int piece = board[sq];
            if (piece != 0) {
                int piece_idx = piece_to_index(piece);
                if (piece_idx >= 0) {
                    hash ^= piece_keys[sq][piece_idx];
                }
            }
        }
        
        // Hash side to move
        if (white_to_move) {
            hash ^= side_key;
        }
        
        // Hash castling rights
        hash ^= castling_keys[castling_rights & 15];
        
        // Hash en passant
        if (en_passant_square >= 0 && en_passant_square < 64) {
            hash ^= en_passant_keys[en_passant_square];
        }
        
        return hash;
    }
};

// HalfKP Feature Extractor in C++
class HalfKPFeatureExtractor {
private:
    static constexpr int NUM_SQUARES = 64;
    static constexpr int NUM_PIECE_TYPES = 6;
    static constexpr int NUM_PIECES = NUM_PIECE_TYPES * 2;
    static constexpr int HALFKP_SIZE = NUM_SQUARES * NUM_PIECES * NUM_SQUARES;

public:
    std::pair<std::vector<float>, std::vector<float>> extract_halfkp_features(
        const std::array<int, 64>& board_array, int white_king_sq, int black_king_sq) {
        
        std::vector<float> white_features(40960, 0.0f);  // HalfKP features for white
        std::vector<float> black_features(40960, 0.0f);  // HalfKP features for black
        
        for (int square = 0; square < NUM_SQUARES; square++) {
            int piece = board_array[square];
            if (piece == 0) continue;
            
            int piece_type = abs(piece) - 1;  // 0-5 for pawn-king
            bool is_white = piece > 0;
            
            // White perspective
            int white_piece_idx = is_white ? piece_type : (piece_type + NUM_PIECE_TYPES);
            int white_feature_idx = white_king_sq * NUM_PIECES * NUM_SQUARES + 
                                   white_piece_idx * NUM_SQUARES + square;
            if (white_feature_idx < 40960) {
                white_features[white_feature_idx] = 1.0f;
            }
            
            // Black perspective (flip board)
            int black_square = 63 - square;
            int black_king_sq_flipped = 63 - black_king_sq;
            int black_piece_idx = is_white ? (piece_type + NUM_PIECE_TYPES) : piece_type;
            int black_feature_idx = black_king_sq_flipped * NUM_PIECES * NUM_SQUARES + 
                                   black_piece_idx * NUM_SQUARES + black_square;
            if (black_feature_idx < 40960) {
                black_features[black_feature_idx] = 1.0f;
            }
        }
        
        return {white_features, black_features};
    }
};

class PureCppChessEngine {
private:
    std::array<int, 64> board;
    bool white_to_move;
    int castling_rights;
    int en_passant_square;
    int halfmove_clock;
    int fullmove_number;
    
    EvaluationMode eval_mode;
    int nodes_searched;
    
    // Transposition table for position caching
    static constexpr size_t TT_SIZE = 1048576;  // 1M entries
    std::array<TTEntry, TT_SIZE> transposition_table;
    ZobristHash zobrist;
    int cache_hits;
    int cache_lookups;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> ort_env;
    std::unique_ptr<Ort::Session> ort_session;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    HalfKPFeatureExtractor feature_extractor;
    bool nnue_available;
    
    // NNUE evaluation cache
    static constexpr size_t NNUE_CACHE_SIZE = 8192;  // Smaller cache for NNUE evals
    std::array<std::pair<uint64_t, double>, NNUE_CACHE_SIZE> nnue_cache;
    int nnue_cache_hits;
    int nnue_cache_lookups;
    
    static int square_from_coords(int file, int rank) {
        return rank * 8 + file;
    }
    
    static std::pair<int, int> coords_from_square(int square) {
        return {square % 8, square / 8};
    }
    
    bool is_valid_square(int square) const {
        return square >= 0 && square < 64;
    }
    
    bool is_enemy_piece(int piece) const {
        if (white_to_move) return piece < 0;
        else return piece > 0;
    }
    
    bool is_friendly_piece(int piece) const {
        if (white_to_move) return piece > 0;
        else return piece < 0;
    }
    
    int piece_type(int piece) const {
        return abs(piece);
    }
    
    // Transposition table methods
    uint64_t get_position_hash() const {
        return zobrist.hash_position(board, white_to_move, castling_rights, en_passant_square);
    }
    
    void store_tt_entry(uint64_t hash, double score, int depth, EntryType type, const Move& best_move) {
        size_t index = hash % TT_SIZE;
        TTEntry& entry = transposition_table[index];
        
        // Replace if empty, deeper search, or same depth but different position
        if (entry.depth <= depth || entry.zobrist_hash != hash) {
            entry.zobrist_hash = hash;
            entry.score = score;
            entry.depth = depth;
            entry.type = type;
            entry.best_move = best_move;
        }
    }
    
    bool probe_tt_entry(uint64_t hash, int depth, double alpha, double beta, double& score, Move& best_move) {
        cache_lookups++;
        size_t index = hash % TT_SIZE;
        const TTEntry& entry = transposition_table[index];
        
        // Check for exact hash match to avoid false positives from collisions
        if (entry.zobrist_hash == hash && entry.depth >= depth) {
            cache_hits++;
            best_move = entry.best_move;
            
            // Validate that the cached move is still legal in current position
            auto legal_moves = generate_moves();
            bool move_is_legal = false;
            for (const auto& move : legal_moves) {
                if (move.from_square == best_move.from_square && 
                    move.to_square == best_move.to_square) {
                    move_is_legal = true;
                    break;
                }
            }
            
            // If cached move is illegal, this is likely a hash collision
            if (!move_is_legal && best_move.from_square >= 0) {
                return false;
            }
            
            switch (entry.type) {
                case EntryType::EXACT:
                    score = entry.score;
                    return true;
                case EntryType::LOWER_BOUND:
                    if (entry.score >= beta) {
                        score = entry.score;
                        return true;
                    }
                    break;
                case EntryType::UPPER_BOUND:
                    if (entry.score <= alpha) {
                        score = entry.score;
                        return true;
                    }
                    break;
            }
        }
        
        return false;
    }
    
    void clear_transposition_table() {
        for (auto& entry : transposition_table) {
            entry = TTEntry();
        }
        cache_hits = 0;
        cache_lookups = 0;
    }
    
    // NNUE cache methods
    bool probe_nnue_cache(uint64_t hash, double& eval) {
        nnue_cache_lookups++;
        size_t index = hash % NNUE_CACHE_SIZE;
        if (nnue_cache[index].first == hash) {
            nnue_cache_hits++;
            eval = nnue_cache[index].second;
            return true;
        }
        return false;
    }
    
    void store_nnue_cache(uint64_t hash, double eval) {
        size_t index = hash % NNUE_CACHE_SIZE;
        nnue_cache[index] = {hash, eval};
    }
    
    void clear_nnue_cache() {
        for (auto& entry : nnue_cache) {
            entry = {0, 0.0};
        }
        nnue_cache_hits = 0;
        nnue_cache_lookups = 0;
    }
    
    // Initialize ONNX Runtime
    bool init_nnue(const std::string& model_path) {
        try {
            ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ChessEngine");
            
            session_options = std::make_unique<Ort::SessionOptions>();
            session_options->SetIntraOpNumThreads(4);
            session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // Enable quantization optimizations for faster inference
            session_options->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            
            // Enable dynamic quantization during inference for faster execution
            // This will automatically quantize weights to int8 during runtime
            session_options->AddConfigEntry("session.dynamic_quantization", "1");
            
            // Enable additional CPU optimizations
            session_options->AddConfigEntry("session.disable_cpu_ep_fallback", "0");
            session_options->AddConfigEntry("session.use_per_session_threads", "1");
            
            // Set memory pattern optimization
            session_options->EnableMemPattern();
            session_options->EnableCpuMemArena();
            
            // Enable profiling to monitor quantization effects (optional)
            session_options->EnableProfiling("nnue_profile");
            
            // Try to enable additional optimization transformers
            session_options->AddConfigEntry("session.qdq_is_int8_allowed", "1");  // Allow QDQ ops to be int8
            session_options->AddConfigEntry("session.optimization.enable_gelu_approximation", "1");
            
            // Load ONNX model with quantization-optimized session
            ort_session = std::make_unique<Ort::Session>(*ort_env, model_path.c_str(), *session_options);
            
            std::cout << "✅ NNUE model loaded with dynamic quantization enabled" << std::endl;
            
            // Get input/output info
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Use hardcoded input/output names since we know them
            input_name_strings = {"white_features", "black_features"};
            output_name_strings = {"evaluation"};
            
            input_names.clear();
            output_names.clear();
            
            for (const auto& name : input_name_strings) {
                input_names.push_back(name.c_str());
            }
            
            for (const auto& name : output_name_strings) {
                output_names.push_back(name.c_str());
            }
            
            memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize ONNX Runtime: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Generate all legal moves
    std::vector<Move> generate_moves() const {
        std::vector<Move> moves;
        moves.reserve(256);
        
        for (int square = 0; square < 64; square++) {
            int piece = board[square];
            if (piece == 0 || !is_friendly_piece(piece)) continue;
            
            int type = piece_type(piece);
            auto [file, rank] = coords_from_square(square);
            
            switch (type) {
                case 1: generate_pawn_moves(square, file, rank, moves); break;
                case 2: generate_knight_moves(square, file, rank, moves); break;
                case 3: generate_bishop_moves(square, file, rank, moves); break;
                case 4: generate_rook_moves(square, file, rank, moves); break;
                case 5: generate_queen_moves(square, file, rank, moves); break;
                case 6: generate_king_moves(square, file, rank, moves); break;
            }
        }
        
        return moves;
    }
    
    void generate_pawn_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        int direction = white_to_move ? 1 : -1;
        int start_rank = white_to_move ? 1 : 6;
        int promotion_rank = white_to_move ? 7 : 0;
        
        // Forward moves
        int forward_square = square + direction * 8;
        if (is_valid_square(forward_square) && board[forward_square] == 0) {
            if (rank + direction == promotion_rank) {
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 5 : -5);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 4 : -4);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 3 : -3);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 2 : -2);
            } else {
                moves.emplace_back(square, forward_square, board[square]);
                
                if (rank == start_rank) {
                    int double_forward = square + direction * 16;
                    if (is_valid_square(double_forward) && board[double_forward] == 0) {
                        moves.emplace_back(square, double_forward, board[square]);
                    }
                }
            }
        }
        
        // Captures
        for (int df : {-1, 1}) {
            if (file + df < 0 || file + df > 7) continue;
            int capture_square = square + direction * 8 + df;
            if (is_valid_square(capture_square)) {
                int target = board[capture_square];
                if (target != 0 && is_enemy_piece(target)) {
                    if (rank + direction == promotion_rank) {
                        moves.emplace_back(square, capture_square, board[square], target, true, white_to_move ? 5 : -5);
                        moves.emplace_back(square, capture_square, board[square], target, true, white_to_move ? 4 : -4);
                        moves.emplace_back(square, capture_square, board[square], target, true, white_to_move ? 3 : -3);
                        moves.emplace_back(square, capture_square, board[square], target, true, white_to_move ? 2 : -2);
                    } else {
                        moves.emplace_back(square, capture_square, board[square], target);
                    }
                }
                else if (capture_square == en_passant_square) {
                    moves.emplace_back(square, capture_square, board[square], white_to_move ? -1 : 1);
                }
            }
        }
    }
    
    void generate_knight_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        const int knight_moves[] = {-17, -15, -10, -6, 6, 10, 15, 17};
        for (int delta : knight_moves) {
            int target_square = square + delta;
            if (!is_valid_square(target_square)) continue;
            
            auto [target_file, target_rank] = coords_from_square(target_square);
            if (abs(target_file - file) + abs(target_rank - rank) != 3) continue;
            
            int target = board[target_square];
            if (target == 0 || is_enemy_piece(target)) {
                moves.emplace_back(square, target_square, board[square], target);
            }
        }
    }
    
    void generate_sliding_moves(int square, int file, int rank, const std::vector<std::pair<int, int>>& directions, std::vector<Move>& moves) const {
        for (auto [df, dr] : directions) {
            for (int i = 1; i < 8; i++) {
                int new_file = file + df * i;
                int new_rank = rank + dr * i;
                if (new_file < 0 || new_file > 7 || new_rank < 0 || new_rank > 7) break;
                
                int target_square = square_from_coords(new_file, new_rank);
                int target = board[target_square];
                
                if (target == 0) {
                    moves.emplace_back(square, target_square, board[square]);
                } else if (is_enemy_piece(target)) {
                    moves.emplace_back(square, target_square, board[square], target);
                    break;
                } else {
                    break;
                }
            }
        }
    }
    
    void generate_bishop_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        generate_sliding_moves(square, file, rank, {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}, moves);
    }
    
    void generate_rook_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        generate_sliding_moves(square, file, rank, {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}, moves);
    }
    
    void generate_queen_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        generate_sliding_moves(square, file, rank, {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}}, moves);
    }
    
    void generate_king_moves(int square, int file, int rank, std::vector<Move>& moves) const {
        for (int df = -1; df <= 1; df++) {
            for (int dr = -1; dr <= 1; dr++) {
                if (df == 0 && dr == 0) continue;
                int new_file = file + df;
                int new_rank = rank + dr;
                if (new_file < 0 || new_file > 7 || new_rank < 0 || new_rank > 7) continue;
                
                int target_square = square_from_coords(new_file, new_rank);
                int target = board[target_square];
                
                if (target == 0 || is_enemy_piece(target)) {
                    moves.emplace_back(square, target_square, board[square], target);
                }
            }
        }
    }
    
    void make_move(const Move& move) {
        board[move.to_square] = move.promotion ? move.promotion_piece : move.piece;
        board[move.from_square] = 0;
        
        if (piece_type(move.piece) == 1 && move.to_square == en_passant_square) {
            int captured_square = move.to_square + (white_to_move ? -8 : 8);
            board[captured_square] = 0;
        }
        
        if (piece_type(move.piece) == 1 && abs(move.to_square - move.from_square) == 16) {
            en_passant_square = (move.from_square + move.to_square) / 2;
        } else {
            en_passant_square = -1;
        }
        
        white_to_move = !white_to_move;
        halfmove_clock++;
        if (!white_to_move) fullmove_number++;
    }
    
    void unmake_move(const Move& move) {
        board[move.from_square] = move.piece;
        board[move.to_square] = move.captured;
        
        if (piece_type(move.piece) == 1 && move.captured != 0 && move.to_square == en_passant_square) {
            int captured_square = move.to_square + (white_to_move ? 8 : -8);
            board[captured_square] = move.captured;
            board[move.to_square] = 0;
        }
        
        white_to_move = !white_to_move;
        halfmove_clock--;
        if (white_to_move) fullmove_number--;
    }
    
    // Deterministic evaluation
    double evaluate_deterministic() const {
        double score = 0.0;
        
        for (int square = 0; square < 64; square++) {
            int piece = board[square];
            if (piece == 0) continue;
            
            int type = piece_type(piece);
            int value = PIECE_VALUES[type];
            
            int position_bonus = 0;
            if (type == 1) {
                position_bonus = piece > 0 ? PAWN_TABLE[square] : PAWN_TABLE[63 - square];
            } else if (type == 2) {
                position_bonus = piece > 0 ? KNIGHT_TABLE[square] : KNIGHT_TABLE[63 - square];
            }
            
            if (piece > 0) {
                score += value + position_bonus;
            } else {
                score -= value + position_bonus;
            }
        }
        
        // Mobility bonus
        bool original_turn = white_to_move;
        
        const_cast<PureCppChessEngine*>(this)->white_to_move = true;
        auto white_moves = generate_moves();
        score += white_moves.size() * 10;
        
        const_cast<PureCppChessEngine*>(this)->white_to_move = false;
        auto black_moves = generate_moves();
        score -= black_moves.size() * 10;
        
        const_cast<PureCppChessEngine*>(this)->white_to_move = original_turn;
        
        return score;  // Always return from White's perspective
    }
    
    // Pure C++ NNUE evaluation
    double evaluate_nnue() {
        if (!nnue_available || !ort_session) {
            return evaluate_deterministic();
        }
        
        // Check NNUE cache first
        uint64_t position_hash = get_position_hash();
        double cached_eval;
        if (probe_nnue_cache(position_hash, cached_eval)) {
            return cached_eval;
        }
        
        try {
            // Find kings
            int white_king_sq = -1, black_king_sq = -1;
            for (int i = 0; i < 64; i++) {
                if (board[i] == 6) white_king_sq = i;
                if (board[i] == -6) black_king_sq = i;
            }
            
            if (white_king_sq == -1 || black_king_sq == -1) {
                return evaluate_deterministic();
            }
            
            // Extract HalfKP features
            auto [white_features, black_features] = feature_extractor.extract_halfkp_features(board, white_king_sq, black_king_sq);
            
            // Create input tensors
            std::array<int64_t, 2> input_shape = {1, 40960};
            
            auto white_tensor = Ort::Value::CreateTensor<float>(*memory_info, white_features.data(), white_features.size(), input_shape.data(), input_shape.size());
            auto black_tensor = Ort::Value::CreateTensor<float>(*memory_info, black_features.data(), black_features.size(), input_shape.data(), input_shape.size());
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(white_tensor));
            input_tensors.push_back(std::move(black_tensor));
            
            
            // Run inference
            auto output_tensors = ort_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_names.size(), output_names.data(), output_names.size());
            
            // Get result
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            double evaluation_cp = static_cast<double>(output_data[0]);
            
            // Clamp and normalize
            evaluation_cp = std::max(-3000.0, std::min(3000.0, evaluation_cp));
            double normalized_eval = evaluation_cp / 100.0;
            
            // Store in cache
            store_nnue_cache(position_hash, normalized_eval);
            
            return normalized_eval;  // Always return from White's perspective
            
        } catch (const std::exception& e) {
            std::cerr << "NNUE evaluation error: " << e.what() << std::endl;
            return evaluate_deterministic();
        }
    }
    
    double evaluate() {
        switch (eval_mode) {
            case EvaluationMode::NNUE:
                return evaluate_nnue();
            case EvaluationMode::DETERMINISTIC:
            default:
                return evaluate_deterministic();
        }
    }
    
    std::pair<double, Move> alpha_beta(int depth, double alpha, double beta, bool maximizing) {
        nodes_searched++;
        
        // Get position hash for transposition table
        uint64_t position_hash = get_position_hash();
        double tt_score;
        Move tt_move;
        
        // Probe transposition table
        if (probe_tt_entry(position_hash, depth, alpha, beta, tt_score, tt_move)) {
            return {tt_score, tt_move};
        }
        
        if (depth == 0) {
            double eval = evaluate();
            store_tt_entry(position_hash, eval, depth, EntryType::EXACT, Move());
            return {eval, Move()};
        }
        
        auto moves = generate_moves();
        if (moves.empty()) {
            double mate_score = maximizing ? 
                (-10000.0 + (6 - depth)) : 
                (10000.0 - (6 - depth));
            store_tt_entry(position_hash, mate_score, depth, EntryType::EXACT, Move());
            return {mate_score, Move()};
        }
        
        // Move ordering - prioritize TT move, then captures
        if (tt_move.from_square >= 0) {
            // Move TT move to front if it exists in legal moves
            auto it = std::find_if(moves.begin(), moves.end(), 
                [&tt_move](const Move& m) {
                    return m.from_square == tt_move.from_square && m.to_square == tt_move.to_square;
                });
            if (it != moves.end()) {
                std::swap(*it, moves[0]);
            }
        }
        
        // Sort remaining moves by capture value
        std::sort(moves.begin() + (tt_move.from_square >= 0 ? 1 : 0), moves.end(), 
                  [](const Move& a, const Move& b) {
                      return abs(a.captured) > abs(b.captured);
                  });
        
        Move best_move;
        double best_score;
        EntryType entry_type = EntryType::UPPER_BOUND;
        
        if (maximizing) {
            best_score = -std::numeric_limits<double>::infinity();
            for (const auto& move : moves) {
                make_move(move);
                auto [eval, _] = alpha_beta(depth - 1, alpha, beta, false);
                unmake_move(move);
                
                if (eval > best_score) {
                    best_score = eval;
                    best_move = move;
                }
                
                if (eval >= beta) {
                    // Beta cutoff - store as lower bound
                    store_tt_entry(position_hash, eval, depth, EntryType::LOWER_BOUND, move);
                    return {eval, move};
                }
                
                if (eval > alpha) {
                    alpha = eval;
                    entry_type = EntryType::EXACT;
                }
            }
        } else {
            best_score = std::numeric_limits<double>::infinity();
            for (const auto& move : moves) {
                make_move(move);
                auto [eval, _] = alpha_beta(depth - 1, alpha, beta, true);
                unmake_move(move);
                
                if (eval < best_score) {
                    best_score = eval;
                    best_move = move;
                }
                
                if (eval <= alpha) {
                    // Alpha cutoff - store as upper bound
                    store_tt_entry(position_hash, eval, depth, EntryType::UPPER_BOUND, move);
                    return {eval, move};
                }
                
                if (eval < beta) {
                    beta = eval;
                    entry_type = EntryType::EXACT;
                }
            }
        }
        
        // Store result in transposition table
        store_tt_entry(position_hash, best_score, depth, entry_type, best_move);
        return {best_score, best_move};
    }
    
public:
    PureCppChessEngine() : eval_mode(EvaluationMode::DETERMINISTIC), cache_hits(0), 
                           cache_lookups(0), nnue_cache_hits(0), nnue_cache_lookups(0), 
                           nnue_available(false) {
        set_start_position();
        clear_transposition_table();
        clear_nnue_cache();
    }
    
    bool init_nnue_model(const std::string& model_path) {
        nnue_available = init_nnue(model_path);
        return nnue_available;
    }
    
    void set_start_position() {
        board = {{
            -4, -2, -3, -5, -6, -3, -2, -4,
            -1, -1, -1, -1, -1, -1, -1, -1,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             1,  1,  1,  1,  1,  1,  1,  1,
             4,  2,  3,  5,  6,  3,  2,  4
        }};
        
        white_to_move = true;
        castling_rights = 15;
        en_passant_square = -1;
        halfmove_clock = 0;
        fullmove_number = 1;
    }
    
    void set_evaluation_mode(const std::string& mode) {
        if (mode == "nnue" || mode == "NNUE") {
            eval_mode = EvaluationMode::NNUE;
        } else {
            eval_mode = EvaluationMode::DETERMINISTIC;
        }
    }
    
    std::string get_evaluation_mode() const {
        switch (eval_mode) {
            case EvaluationMode::NNUE: return nnue_available ? "NNUE (Quantized)" : "NNUE (fallback to Deterministic)";
            case EvaluationMode::DETERMINISTIC: return "Deterministic";
            default: return "Unknown";
        }
    }
    
    void set_fen(const std::string& fen) {
        std::fill(board.begin(), board.end(), 0);
        
        std::istringstream ss(fen);
        std::string piece_placement;
        ss >> piece_placement;
        
        int square = 56;
        for (char c : piece_placement) {
            if (c == '/') {
                square -= 16;
            } else if (c >= '1' && c <= '8') {
                square += c - '0';
            } else {
                int piece = 0;
                switch (c) {
                    case 'P': piece = 1; break; case 'p': piece = -1; break;
                    case 'N': piece = 2; break; case 'n': piece = -2; break;
                    case 'B': piece = 3; break; case 'b': piece = -3; break;
                    case 'R': piece = 4; break; case 'r': piece = -4; break;
                    case 'Q': piece = 5; break; case 'q': piece = -5; break;
                    case 'K': piece = 6; break; case 'k': piece = -6; break;
                }
                if (square >= 0 && square < 64) {
                    board[square] = piece;
                }
                square++;
            }
        }
        
        std::string turn;
        if (ss >> turn) {
            white_to_move = (turn == "w");
        }
    }
    
    std::pair<std::string, double> get_best_move(int depth = 5) {
        nodes_searched = 0;
        cache_hits = 0;
        cache_lookups = 0;
        nnue_cache_hits = 0;
        nnue_cache_lookups = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [score, best_move] = alpha_beta(depth, -std::numeric_limits<double>::infinity(), 
                                           std::numeric_limits<double>::infinity(), white_to_move);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double cache_hit_rate = cache_lookups > 0 ? (100.0 * cache_hits / cache_lookups) : 0.0;
        double nnue_cache_hit_rate = nnue_cache_lookups > 0 ? (100.0 * nnue_cache_hits / nnue_cache_lookups) : 0.0;
        
        std::cout << "Search depth: " << depth << ", Nodes: " << nodes_searched 
                  << ", Time: " << duration.count() << "ms, Score: " << score 
                  << ", Cache: " << cache_hits << "/" << cache_lookups 
                  << " (" << std::fixed << std::setprecision(1) << cache_hit_rate << "%)";
        
        if (nnue_cache_lookups > 0) {
            std::cout << ", NNUE: " << nnue_cache_hits << "/" << nnue_cache_lookups 
                      << " (" << std::fixed << std::setprecision(1) << nnue_cache_hit_rate << "%)";
        }
        
        std::cout << " (eval: " << get_evaluation_mode() << ")" << std::endl;
        
        return {best_move.to_uci(), score};
    }
    
    std::vector<std::string> get_legal_moves() {
        auto moves = generate_moves();
        std::vector<std::string> result;
        result.reserve(moves.size());
        for (const auto& move : moves) {
            result.push_back(move.to_uci());
        }
        return result;
    }
    
    bool make_uci_move(const std::string& uci_move) {
        if (uci_move.length() < 4) return false;
        
        int from_square = (uci_move[0] - 'a') + (uci_move[1] - '1') * 8;
        int to_square = (uci_move[2] - 'a') + (uci_move[3] - '1') * 8;
        
        auto moves = generate_moves();
        for (const auto& move : moves) {
            if (move.from_square == from_square && move.to_square == to_square) {
                if (uci_move.length() == 5 && move.promotion) {
                    char prom_char = uci_move[4];
                    int prom_piece = 0;
                    switch (prom_char) {
                        case 'n': prom_piece = white_to_move ? 2 : -2; break;
                        case 'b': prom_piece = white_to_move ? 3 : -3; break;
                        case 'r': prom_piece = white_to_move ? 4 : -4; break;
                        case 'q': prom_piece = white_to_move ? 5 : -5; break;
                    }
                    if (abs(move.promotion_piece) == abs(prom_piece)) {
                        make_move(move);
                        return true;
                    }
                } else if (uci_move.length() == 4 && !move.promotion) {
                    make_move(move);
                    return true;
                }
            }
        }
        return false;
    }
};

// Python bindings
PYBIND11_MODULE(pure_cpp_chess_engine, m) {
    m.doc() = "Pure C++ chess engine with native ONNX Runtime NNUE evaluation";
    
    py::class_<PureCppChessEngine>(m, "PureCppChessEngine")
        .def(py::init<>())
        .def("init_nnue_model", &PureCppChessEngine::init_nnue_model)
        .def("set_start_position", &PureCppChessEngine::set_start_position)
        .def("set_fen", &PureCppChessEngine::set_fen)
        .def("set_evaluation_mode", &PureCppChessEngine::set_evaluation_mode)
        .def("get_evaluation_mode", &PureCppChessEngine::get_evaluation_mode)
        .def("get_best_move", &PureCppChessEngine::get_best_move, py::arg("depth") = 5)
        .def("get_legal_moves", &PureCppChessEngine::get_legal_moves)
        .def("make_uci_move", &PureCppChessEngine::make_uci_move);
}