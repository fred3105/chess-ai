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

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace py = pybind11;

// Piece values for deterministic evaluation
const int PIECE_VALUES[] = {0, 100, 320, 330, 500, 900, 20000}; // Empty, Pawn, Knight, Bishop, Rook, Queen, King

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

class HybridChessEngine {
private:
    std::array<int, 64> board;
    bool white_to_move;
    int castling_rights;
    int en_passant_square;
    int halfmove_clock;
    int fullmove_number;
    
    EvaluationMode eval_mode;
    int nodes_searched;
    
#ifdef USE_ONNX
    std::unique_ptr<Ort::Session> onnx_session;
    std::unique_ptr<Ort::Env> onnx_env;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
#endif
    
    // Python evaluation function (for NNUE)
    py::function python_evaluator;
    bool has_python_evaluator;
    
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
    
    // Generate all legal moves for current position
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
                // Promotion
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 5 : -5);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 4 : -4);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 3 : -3);
                moves.emplace_back(square, forward_square, board[square], 0, true, white_to_move ? 2 : -2);
            } else {
                moves.emplace_back(square, forward_square, board[square]);
                
                // Double move from start position
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
                // En passant
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
        
        const_cast<HybridChessEngine*>(this)->white_to_move = true;
        auto white_moves = generate_moves();
        score += white_moves.size() * 10;
        
        const_cast<HybridChessEngine*>(this)->white_to_move = false;
        auto black_moves = generate_moves();
        score -= black_moves.size() * 10;
        
        const_cast<HybridChessEngine*>(this)->white_to_move = original_turn;
        
        return white_to_move ? score : -score;
    }
    
    // NNUE evaluation via Python callback
    double evaluate_nnue() {
        if (!has_python_evaluator) {
            return evaluate_deterministic(); // Fallback
        }
        
        try {
            // Convert board to FEN for Python evaluation
            std::string fen = get_fen();
            py::object result = python_evaluator(fen);
            double score = result.cast<double>();
            return score * 100.0; // Convert to centipawns scale
        } catch (const std::exception& e) {
            std::cerr << "NNUE evaluation error: " << e.what() << std::endl;
            return evaluate_deterministic(); // Fallback to deterministic
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
        
        if (depth == 0) {
            return {evaluate(), Move()};
        }
        
        auto moves = generate_moves();
        if (moves.empty()) {
            return maximizing ? 
                std::make_pair(-10000.0 + (6 - depth), Move()) : 
                std::make_pair(10000.0 - (6 - depth), Move());
        }
        
        // Move ordering - prioritize captures
        std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
            return abs(a.captured) > abs(b.captured);
        });
        
        Move best_move;
        
        if (maximizing) {
            double max_eval = -std::numeric_limits<double>::infinity();
            for (const auto& move : moves) {
                make_move(move);
                auto [eval, _] = alpha_beta(depth - 1, alpha, beta, false);
                unmake_move(move);
                
                if (eval > max_eval) {
                    max_eval = eval;
                    best_move = move;
                }
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
            return {max_eval, best_move};
        } else {
            double min_eval = std::numeric_limits<double>::infinity();
            for (const auto& move : moves) {
                make_move(move);
                auto [eval, _] = alpha_beta(depth - 1, alpha, beta, true);
                unmake_move(move);
                
                if (eval < min_eval) {
                    min_eval = eval;
                    best_move = move;
                }
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
            return {min_eval, best_move};
        }
    }
    
public:
    HybridChessEngine() : eval_mode(EvaluationMode::DETERMINISTIC), has_python_evaluator(false) {
        set_start_position();
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
            case EvaluationMode::NNUE: return "NNUE";
            case EvaluationMode::DETERMINISTIC: return "Deterministic";
            default: return "Unknown";
        }
    }
    
    void set_python_evaluator(py::function evaluator) {
        python_evaluator = evaluator;
        has_python_evaluator = true;
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
    
    std::string get_fen() const {
        std::string fen;
        
        // Piece placement
        for (int rank = 7; rank >= 0; rank--) {
            int empty_count = 0;
            for (int file = 0; file < 8; file++) {
                int piece = board[rank * 8 + file];
                if (piece == 0) {
                    empty_count++;
                } else {
                    if (empty_count > 0) {
                        fen += std::to_string(empty_count);
                        empty_count = 0;
                    }
                    char c = '?';
                    switch (piece) {
                        case 1: c = 'P'; break; case -1: c = 'p'; break;
                        case 2: c = 'N'; break; case -2: c = 'n'; break;
                        case 3: c = 'B'; break; case -3: c = 'b'; break;
                        case 4: c = 'R'; break; case -4: c = 'r'; break;
                        case 5: c = 'Q'; break; case -5: c = 'q'; break;
                        case 6: c = 'K'; break; case -6: c = 'k'; break;
                    }
                    fen += c;
                }
            }
            if (empty_count > 0) {
                fen += std::to_string(empty_count);
            }
            if (rank > 0) fen += '/';
        }
        
        // Active color
        fen += white_to_move ? " w" : " b";
        
        // Castling rights (simplified)
        fen += " KQkq";
        
        // En passant
        fen += " -";
        
        // Halfmove clock and fullmove number
        fen += " " + std::to_string(halfmove_clock) + " " + std::to_string(fullmove_number);
        
        return fen;
    }
    
    std::pair<std::string, double> get_best_move(int depth = 5) {
        nodes_searched = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [score, best_move] = alpha_beta(depth, -std::numeric_limits<double>::infinity(), 
                                           std::numeric_limits<double>::infinity(), white_to_move);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Search depth: " << depth << ", Nodes: " << nodes_searched 
                  << ", Time: " << duration.count() << "ms, Score: " << score 
                  << " (eval: " << get_evaluation_mode() << ")" << std::endl;
        
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
    
    std::string get_board_string() const {
        std::string result;
        for (int rank = 7; rank >= 0; rank--) {
            for (int file = 0; file < 8; file++) {
                int piece = board[rank * 8 + file];
                char c = '.';
                switch (piece) {
                    case 1: c = 'P'; break; case -1: c = 'p'; break;
                    case 2: c = 'N'; break; case -2: c = 'n'; break;
                    case 3: c = 'B'; break; case -3: c = 'b'; break;
                    case 4: c = 'R'; break; case -4: c = 'r'; break;
                    case 5: c = 'Q'; break; case -5: c = 'q'; break;
                    case 6: c = 'K'; break; case -6: c = 'k'; break;
                }
                result += c;
            }
            result += '\n';
        }
        return result;
    }
};

// Python bindings
PYBIND11_MODULE(hybrid_chess_engine, m) {
    m.doc() = "Hybrid C++ chess engine with deterministic and NNUE evaluation";
    
    py::class_<HybridChessEngine>(m, "HybridChessEngine")
        .def(py::init<>())
        .def("set_start_position", &HybridChessEngine::set_start_position)
        .def("set_fen", &HybridChessEngine::set_fen)
        .def("get_fen", &HybridChessEngine::get_fen)
        .def("set_evaluation_mode", &HybridChessEngine::set_evaluation_mode)
        .def("get_evaluation_mode", &HybridChessEngine::get_evaluation_mode)
        .def("set_python_evaluator", &HybridChessEngine::set_python_evaluator)
        .def("get_best_move", &HybridChessEngine::get_best_move, py::arg("depth") = 5)
        .def("get_legal_moves", &HybridChessEngine::get_legal_moves)
        .def("make_uci_move", &HybridChessEngine::make_uci_move)
        .def("get_board_string", &HybridChessEngine::get_board_string);
}