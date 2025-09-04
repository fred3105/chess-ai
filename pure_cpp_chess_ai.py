#!/usr/bin/env uv run python3
"""
Pure C++ Chess AI - Zero Python overhead for NNUE evaluation
Uses ONNX Runtime directly in C++ for maximum performance
"""

import os
import sys
import time

import chess

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_search'))
    import pure_cpp_chess_engine
    PureCppChessEngine = pure_cpp_chess_engine.PureCppChessEngine
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    CPP_AVAILABLE = False


class PureCppChessAI:
    """
    Pure C++ Chess AI with zero Python overhead
    
    Features:
    - Native C++ ONNX Runtime integration
    - No Python callbacks during search
    - Maximum performance for both deterministic and NNUE evaluation
    """
    
    def __init__(self, evaluation_mode="deterministic", model_path="checkpoints/nnue_model.onnx"):
        if not CPP_AVAILABLE:
            raise ImportError("C++ engine not available. Run: cd cpp_search && python setup_pure.py build_ext --inplace")
        
        self.engine = PureCppChessEngine()
        self.engine.set_start_position()
        
        # Initialize NNUE if requested and available
        self.nnue_initialized = False
        if evaluation_mode.lower() == "nnue":
            if os.path.exists(model_path):
                self.nnue_initialized = self.engine.init_nnue_model(model_path)
                if self.nnue_initialized:
                    self.engine.set_evaluation_mode("nnue")
                else:
                    self.engine.set_evaluation_mode("deterministic")
            else:
                self.engine.set_evaluation_mode("deterministic")
        else:
            self.engine.set_evaluation_mode("deterministic")
            
    
    def set_evaluation_mode(self, mode):
        """Switch between evaluation modes"""
        mode = mode.lower()
        if mode == "nnue" and not self.nnue_initialized:
            return
            
        if mode in ["nnue", "deterministic"]:
            self.engine.set_evaluation_mode(mode)
    
    def get_evaluation_mode(self):
        """Get current evaluation mode"""
        return self.engine.get_evaluation_mode()
    
    def set_position(self, board: chess.Board):
        """Set the engine position from a python-chess board"""
        fen = board.fen()
        self.engine.set_fen(fen)
    
    def get_best_move(self, board: chess.Board, depth: int = 5, time_limit: float = None) -> tuple[chess.Move, float]:
        """Get the best move for the current position"""
        start_time = time.time()
        
        # Set position in C++ engine
        self.set_position(board)
        
        # Get best move from C++ engine
        uci_move, score = self.engine.get_best_move(depth)
        
        # Convert UCI move back to python-chess move
        if uci_move == "0000" or not uci_move:
            # No valid move found
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]  # Return any legal move as fallback
            else:
                return None, score
        else:
            try:
                best_move = chess.Move.from_uci(uci_move)
                # Verify the move is legal
                if best_move not in board.legal_moves:
                    legal_moves = list(board.legal_moves)
                    best_move = legal_moves[0] if legal_moves else None
            except (ValueError, chess.InvalidMoveError):
                # Invalid UCI move, fallback to any legal move
                legal_moves = list(board.legal_moves)
                best_move = legal_moves[0] if legal_moves else None
        
        elapsed_time = time.time() - start_time
        eval_mode = self.get_evaluation_mode()
        
        return best_move, score


if __name__ == "__main__":
    # Test both evaluation modes
    try:
        print("=== Testing Pure C++ Chess AI Performance ===")
        print()
        
        # Test deterministic evaluation
        print("1. Testing Pure C++ Deterministic Evaluation:")
        ai_det = PureCppChessAI(evaluation_mode="deterministic")
        board = chess.Board()
        
        move, score = ai_det.get_best_move(board, depth=6)
        print(f"Best move: {move}, Score: {score}")
        print()
        
        # Test NNUE evaluation  
        print("2. Testing Pure C++ NNUE Evaluation:")
        try:
            ai_nnue = PureCppChessAI(evaluation_mode="nnue")
            move, score = ai_nnue.get_best_move(board, depth=4)
            print(f"Best move: {move}, Score: {score}")
            print()
            
            # Test switching modes
            print("3. Testing Mode Switching:")
            ai_nnue.set_evaluation_mode("deterministic")
            move, score = ai_nnue.get_best_move(board, depth=5)
            print(f"Deterministic: {move}, Score: {score}")
            
            ai_nnue.set_evaluation_mode("nnue")  
            move, score = ai_nnue.get_best_move(board, depth=4)
            print(f"NNUE: {move}, Score: {score}")
            
        except Exception as e:
            print(f"NNUE test failed: {e}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("To build the C++ extension:")
        print("cd cpp_search && python setup_pure.py build_ext --inplace")