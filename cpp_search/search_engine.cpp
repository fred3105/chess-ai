#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <chrono>
#include <thread>
#include <future>

namespace py = pybind11;

struct Move {
    int from_square;
    int to_square;
    int piece_type;
    int captured_piece;
    bool is_promotion;
    int promotion_piece;
    
    Move() : from_square(-1), to_square(-1), piece_type(0), 
             captured_piece(0), is_promotion(false), promotion_piece(0) {}
    
    Move(int from, int to, int piece = 0, int captured = 0) 
        : from_square(from), to_square(to), piece_type(piece), 
          captured_piece(captured), is_promotion(false), promotion_piece(0) {}
};

struct SearchResult {
    Move best_move;
    double score;
    int nodes_searched;
    double time_taken;
    
    SearchResult() : score(0.0), nodes_searched(0), time_taken(0.0) {}
};

class FastSearchEngine {
private:
    std::vector<double> evaluation_cache;
    int cache_hits = 0;
    int cache_misses = 0;
    
public:
    FastSearchEngine() {
        // Pre-allocate evaluation cache
        evaluation_cache.reserve(100000);
    }
    
    // Fast alpha-beta search implementation
    SearchResult alpha_beta_search(
        const std::vector<std::vector<int>>& board_states,
        const std::vector<double>& evaluations,
        const std::vector<std::vector<int>>& legal_moves_from,
        const std::vector<std::vector<int>>& legal_moves_to,
        int depth,
        bool is_white_turn,
        double alpha = -std::numeric_limits<double>::infinity(),
        double beta = std::numeric_limits<double>::infinity()
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        SearchResult result;
        result.nodes_searched = 0;
        
        if (depth == 0 || legal_moves_from.empty()) {
            if (!evaluations.empty()) {
                result.score = evaluations[0];  // Use pre-computed evaluation
            }
            result.nodes_searched = 1;
            return result;
        }
        
        Move best_move;
        double best_score = is_white_turn ? 
            -std::numeric_limits<double>::infinity() : 
             std::numeric_limits<double>::infinity();
        
        // Move ordering: sort by evaluation (for better pruning)
        std::vector<std::pair<int, double>> move_scores;
        for (size_t i = 0; i < std::min(legal_moves_from.size(), evaluations.size()); ++i) {
            move_scores.emplace_back(i, evaluations[i]);
        }
        
        if (is_white_turn) {
            std::sort(move_scores.begin(), move_scores.end(), 
                     [](const auto& a, const auto& b) { return a.second > b.second; });
        } else {
            std::sort(move_scores.begin(), move_scores.end(), 
                     [](const auto& a, const auto& b) { return a.second < b.second; });
        }
        
        for (const auto& [move_idx, eval_score] : move_scores) {
            if (static_cast<size_t>(move_idx) >= legal_moves_from.size() || 
                static_cast<size_t>(move_idx) >= legal_moves_to.size()) {
                continue;
            }
            
            // Create move
            Move current_move(legal_moves_from[move_idx][0], 
                            legal_moves_to[move_idx][0]);
            
            // Recursive search (simplified for demonstration)
            SearchResult child_result;
            child_result.score = eval_score;  // Use pre-computed evaluation for leaf nodes
            child_result.nodes_searched = 1;
            
            result.nodes_searched += child_result.nodes_searched;
            
            if (is_white_turn) {
                if (child_result.score > best_score) {
                    best_score = child_result.score;
                    best_move = current_move;
                }
                alpha = std::max(alpha, child_result.score);
                if (beta <= alpha) {
                    break;  // Alpha-beta pruning
                }
            } else {
                if (child_result.score < best_score) {
                    best_score = child_result.score;
                    best_move = current_move;
                }
                beta = std::min(beta, child_result.score);
                if (beta <= alpha) {
                    break;  // Alpha-beta pruning
                }
            }
        }
        
        result.best_move = best_move;
        result.score = best_score;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.time_taken = duration.count() / 1000000.0;  // Convert to seconds
        
        return result;
    }
    
    // Parallel search using multiple threads
    SearchResult parallel_search(
        const std::vector<std::vector<int>>& board_states,
        const std::vector<double>& evaluations,
        const std::vector<std::vector<int>>& legal_moves_from,
        const std::vector<std::vector<int>>& legal_moves_to,
        int depth,
        bool is_white_turn,
        int num_threads = 4
    ) {
        if (legal_moves_from.size() <= static_cast<size_t>(num_threads)) {
            // Use single-threaded search for small move sets
            return alpha_beta_search(board_states, evaluations, legal_moves_from, 
                                   legal_moves_to, depth, is_white_turn);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Divide moves among threads
        size_t moves_per_thread = legal_moves_from.size() / num_threads;
        std::vector<std::future<SearchResult>> futures;
        
        for (int t = 0; t < num_threads; ++t) {
            size_t start_idx = t * moves_per_thread;
            size_t end_idx = (t == num_threads - 1) ? 
                legal_moves_from.size() : (t + 1) * moves_per_thread;
            
            if (start_idx >= legal_moves_from.size()) break;
            
            // Create subset of moves for this thread
            std::vector<std::vector<int>> thread_moves_from(
                legal_moves_from.begin() + start_idx,
                legal_moves_from.begin() + end_idx);
            std::vector<std::vector<int>> thread_moves_to(
                legal_moves_to.begin() + start_idx,
                legal_moves_to.begin() + end_idx);
            std::vector<double> thread_evals(
                evaluations.begin() + start_idx,
                evaluations.begin() + std::min(end_idx, evaluations.size()));
            
            futures.push_back(
                std::async(std::launch::async, [=]() {
                    return alpha_beta_search(board_states, thread_evals, 
                                           thread_moves_from, thread_moves_to,
                                           depth - 1, !is_white_turn);
                })
            );
        }
        
        // Collect results from threads
        SearchResult best_result;
        best_result.score = is_white_turn ? 
            -std::numeric_limits<double>::infinity() : 
             std::numeric_limits<double>::infinity();
        
        for (auto& future : futures) {
            SearchResult thread_result = future.get();
            best_result.nodes_searched += thread_result.nodes_searched;
            
            if ((is_white_turn && thread_result.score > best_result.score) ||
                (!is_white_turn && thread_result.score < best_result.score)) {
                best_result.score = thread_result.score;
                best_result.best_move = thread_result.best_move;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        best_result.time_taken = duration.count() / 1000000.0;
        
        return best_result;
    }
    
    // Batch evaluate multiple positions
    std::vector<double> batch_evaluate(
        const std::vector<std::vector<std::vector<int>>>& board_states
    ) {
        std::vector<double> results;
        results.reserve(board_states.size());
        
        // Placeholder for actual batch evaluation
        // In real implementation, this would call ONNX runtime
        for (const auto& board : board_states) {
            // Simple heuristic evaluation for demonstration
            double eval = 0.0;
            for (const auto& row : board) {
                for (int piece : row) {
                    eval += piece * 0.1;  // Simple piece value
                }
            }
            results.push_back(eval);
        }
        
        return results;
    }
    
    void reset_cache() {
        evaluation_cache.clear();
        cache_hits = 0;
        cache_misses = 0;
    }
    
    std::pair<int, int> get_cache_stats() {
        return {cache_hits, cache_misses};
    }
};

// Python bindings
PYBIND11_MODULE(cpp_search, m) {
    m.doc() = "Fast C++ chess search engine";
    
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<int, int, int, int>())
        .def_readwrite("from_square", &Move::from_square)
        .def_readwrite("to_square", &Move::to_square)
        .def_readwrite("piece_type", &Move::piece_type)
        .def_readwrite("captured_piece", &Move::captured_piece)
        .def_readwrite("is_promotion", &Move::is_promotion)
        .def_readwrite("promotion_piece", &Move::promotion_piece);
    
    py::class_<SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("best_move", &SearchResult::best_move)
        .def_readwrite("score", &SearchResult::score)
        .def_readwrite("nodes_searched", &SearchResult::nodes_searched)
        .def_readwrite("time_taken", &SearchResult::time_taken);
    
    py::class_<FastSearchEngine>(m, "FastSearchEngine")
        .def(py::init<>())
        .def("alpha_beta_search", &FastSearchEngine::alpha_beta_search,
             "Fast alpha-beta search implementation",
             py::arg("board_states"), py::arg("evaluations"),
             py::arg("legal_moves_from"), py::arg("legal_moves_to"),
             py::arg("depth"), py::arg("is_white_turn"),
             py::arg("alpha") = -std::numeric_limits<double>::infinity(),
             py::arg("beta") = std::numeric_limits<double>::infinity())
        .def("parallel_search", &FastSearchEngine::parallel_search,
             "Parallel search using multiple threads",
             py::arg("board_states"), py::arg("evaluations"),
             py::arg("legal_moves_from"), py::arg("legal_moves_to"),
             py::arg("depth"), py::arg("is_white_turn"),
             py::arg("num_threads") = 4)
        .def("batch_evaluate", &FastSearchEngine::batch_evaluate,
             "Batch evaluate multiple positions")
        .def("reset_cache", &FastSearchEngine::reset_cache)
        .def("get_cache_stats", &FastSearchEngine::get_cache_stats);
}