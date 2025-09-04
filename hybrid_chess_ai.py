#!/usr/bin/env uv run python3
"""
Hybrid C++ Chess AI - Choose between Deterministic and NNUE evaluation
"""

import os
import sys
import time

import chess
import numpy as np
import torch

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp_search'))
    import hybrid_chess_engine
    HybridChessEngine = hybrid_chess_engine.HybridChessEngine
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    CPP_AVAILABLE = False

try:
    from nnue_model import HalfKPFeatureExtractor, create_nnue_model
    NNUE_AVAILABLE = True
except ImportError:
    NNUE_AVAILABLE = False


class HybridChessAI:
    """
    Hybrid C++ Chess AI with dual evaluation modes:
    - Deterministic: Fast classical evaluation (material + position)  
    - NNUE: Your trained neural network evaluation
    """
    
    def __init__(self, evaluation_mode="deterministic", model_path="checkpoints/best_model.pth", device="cpu"):
        if not CPP_AVAILABLE:
            raise ImportError("C++ engine not available. Run: cd cpp_search && python setup_hybrid.py build_ext --inplace")
        
        self.engine = HybridChessEngine()
        self.engine.set_start_position()
        self.evaluation_mode = evaluation_mode.lower()
        
        # Initialize NNUE if requested
        if self.evaluation_mode == "nnue":
            if not NNUE_AVAILABLE:
                print("Warning: NNUE model not available, falling back to deterministic evaluation")
                self.evaluation_mode = "deterministic"
                self.engine.set_evaluation_mode("deterministic")
            else:
                self._setup_nnue(model_path, device)
        else:
            self.engine.set_evaluation_mode("deterministic")
            
        print(f"Hybrid C++ Chess AI initialized - Evaluation: {self.engine.get_evaluation_mode()}")
    
    def _setup_nnue(self, model_path, device):
        """Setup NNUE evaluation"""
        try:
            # Load NNUE model
            self.device = device
            self.model = create_nnue_model()
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                self.model.to(device)
                self.model.eval()
                
                # Initialize feature extractor
                self.feature_extractor = HalfKPFeatureExtractor()
                
                # Set up Python evaluation callback
                self.engine.set_evaluation_mode("nnue")
                self.engine.set_python_evaluator(self._evaluate_nnue_position)
                
                print(f"NNUE model loaded from {model_path}")
            else:
                print(f"Warning: NNUE model not found at {model_path}, using deterministic evaluation")
                self.evaluation_mode = "deterministic"
                self.engine.set_evaluation_mode("deterministic")
                
        except Exception as e:
            print(f"Error loading NNUE model: {e}")
            print("Falling back to deterministic evaluation")
            self.evaluation_mode = "deterministic"
            self.engine.set_evaluation_mode("deterministic")
    
    def _evaluate_nnue_position(self, fen_string):
        """NNUE evaluation callback for C++ engine"""
        try:
            # Parse FEN and create board
            board = chess.Board(fen_string)
            
            # Convert board to array
            board_array = self._board_to_array(board)
            
            # Find king squares
            white_king_sq = board.king(chess.WHITE)
            black_king_sq = board.king(chess.BLACK)
            
            if white_king_sq is None or black_king_sq is None:
                return 0.0
            
            # Extract HalfKP features
            white_features, black_features = self.feature_extractor.extract_halfkp_features(
                board_array, white_king_sq, black_king_sq
            )
            
            # Get evaluation from NNUE model
            with torch.no_grad():
                white_features = white_features.unsqueeze(0).to(self.device)
                black_features = black_features.unsqueeze(0).to(self.device)
                evaluation_cp = float(self.model(white_features, black_features).item())
            
            # Convert centipawns to normalized scale and return from white's perspective
            evaluation_cp = max(-3000, min(3000, evaluation_cp))
            normalized_eval = evaluation_cp / 100.0
            
            return normalized_eval
            
        except Exception as e:
            print(f"NNUE evaluation error: {e}")
            return 0.0
    
    def _board_to_array(self, board: chess.Board) -> np.ndarray:
        """Convert chess board to 8x8 numpy array for NNUE feature extraction"""
        array = np.zeros((8, 8), dtype=np.int8)

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6,
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                value = piece_values[piece.piece_type]
                if not piece.color:  # Black pieces negative
                    value = -value
                array[7 - row, col] = value  # Flip row for correct orientation

        return array
    
    def set_evaluation_mode(self, mode):
        """Switch between evaluation modes"""
        mode = mode.lower()
        if mode == "nnue" and not NNUE_AVAILABLE:
            print("NNUE not available, staying with current mode")
            return
            
        if mode in ["nnue", "deterministic"]:
            self.evaluation_mode = mode
            self.engine.set_evaluation_mode(mode)
            if mode == "nnue" and hasattr(self, 'model'):
                self.engine.set_python_evaluator(self._evaluate_nnue_position)
            print(f"Switched to {self.engine.get_evaluation_mode()} evaluation")
        else:
            print("Invalid mode. Use 'deterministic' or 'nnue'")
    
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
        print(f"{eval_mode} Engine - Depth {depth}: {best_move} (score: {score:.2f}) in {elapsed_time:.3f}s")
        
        return best_move, score


if __name__ == "__main__":
    # Test both evaluation modes
    try:
        print("=== Testing Hybrid Chess AI ===")
        print()
        
        # Test deterministic evaluation
        print("1. Testing Deterministic Evaluation:")
        ai_det = HybridChessAI(evaluation_mode="deterministic")
        board = chess.Board()
        
        move, score = ai_det.get_best_move(board, depth=4)
        print(f"Best move: {move}, Score: {score}")
        print()
        
        # Test NNUE evaluation
        if NNUE_AVAILABLE:
            print("2. Testing NNUE Evaluation:")
            try:
                ai_nnue = HybridChessAI(evaluation_mode="nnue")
                move, score = ai_nnue.get_best_move(board, depth=4)
                print(f"Best move: {move}, Score: {score}")
                print()
                
                # Test switching modes
                print("3. Testing Mode Switching:")
                ai_nnue.set_evaluation_mode("deterministic")
                move, score = ai_nnue.get_best_move(board, depth=3)
                print(f"Deterministic: {move}, Score: {score}")
                
                ai_nnue.set_evaluation_mode("nnue")  
                move, score = ai_nnue.get_best_move(board, depth=3)
                print(f"NNUE: {move}, Score: {score}")
                
            except Exception as e:
                print(f"NNUE test failed: {e}")
        else:
            print("2. NNUE evaluation not available (missing model files)")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("To build the C++ extension:")
        print("cd cpp_search && python setup_hybrid.py build_ext --inplace")