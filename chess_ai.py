import torch
from custom_class import ChessBoard
from train import EvalNet, encode_board
import chess
import time


class ChessAI:
    def __init__(self, model_path="chess_eval_net_jit.pt", device="mps"):
        """Initialize the Chess AI with trained evaluation function"""
        self.device = device
        try:
            # Try to load JIT compiled model first for speed
            self.model = torch.jit.load(model_path)
        except (FileNotFoundError, RuntimeError):
            # Fallback to regular model
            self.model = EvalNet()
            self.model.load_state_dict(
                torch.load("chess_eval_net.pth", map_location=device)
            )
            self.model.eval()

        self.model.to(device)
        print(f"Chess AI initialized with model on {device}")

    def evaluate_position(self, chess_board):
        """Evaluate a chess position using the trained neural network"""
        # Convert custom board to chess.Board format for encoding
        board = self.custom_board_to_chess_board(chess_board)

        # Encode the position
        encoded = encode_board(board)

        # Get evaluation from neural network
        with torch.no_grad():
            x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(self.device)
            evaluation = float(self.model(x).item())

        # Return evaluation from current player's perspective
        return evaluation if chess_board.is_white_to_move() else -evaluation

    def custom_board_to_chess_board(self, custom_board):
        """Convert custom ChessBoard to python-chess Board for evaluation"""
        board = chess.Board()
        board.clear_board()

        # Piece mapping
        piece_map = {
            1: chess.PAWN,
            2: chess.KNIGHT,
            3: chess.BISHOP,
            5: chess.ROOK,
            9: chess.QUEEN,
            100: chess.KING,
        }

        # Place pieces on the board
        for row in range(8):
            for col in range(8):
                piece_val = custom_board.board[row, col]
                if piece_val != 0:
                    piece_type = piece_map[abs(piece_val)]
                    color = chess.WHITE if piece_val > 0 else chess.BLACK
                    square = chess.square(col, row)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

        # Set turn
        board.turn = chess.WHITE if custom_board.is_white_to_move() else chess.BLACK

        return board

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-beta pruning minimax algorithm"""
        if depth == 0:
            return self.evaluate_position(board), None

        legal_moves = board.generate_legal_moves()

        # Terminal node (no legal moves)
        if not legal_moves:
            if board.is_king_in_check():
                # Checkmate - return very bad score for current player
                return -10000 + (6 - depth), None  # Prefer faster checkmate
            else:
                # Stalemate
                return 0, None

        best_move = None

        if maximizing_player:
            max_eval = float("-inf")
            for move in legal_moves:
                # Make move
                captured_piece = board.board[move[2], move[3]]
                board.make_move(move)

                # Recursive call
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, False)

                # Unmake move
                board.unmake_move(move, captured_piece)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in legal_moves:
                # Make move
                captured_piece = board.board[move[2], move[3]]
                board.make_move(move)

                # Recursive call
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta, True)

                # Unmake move
                board.unmake_move(move, captured_piece)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return min_eval, best_move

    def get_best_move(self, board, depth=4, time_limit=None):
        """Get the best move using alpha-beta search with optional time limit"""
        start_time = time.time()

        # Iterative deepening
        best_move = None
        best_score = None

        for current_depth in range(1, depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            try:
                score, move = self.alpha_beta(
                    board,
                    current_depth,
                    float("-inf"),
                    float("inf"),
                    board.is_white_to_move(),
                )

                if move is not None:
                    best_move = move
                    best_score = score

                elapsed_time = time.time() - start_time
                print(
                    f"Depth {current_depth}: Best move {self.move_to_algebraic(best_move)}, "
                    f"Score: {best_score:.3f}, Time: {elapsed_time:.2f}s"
                )

                if time_limit and elapsed_time > time_limit * 0.8:
                    break

            except KeyboardInterrupt:
                break

        return best_move, best_score

    def move_to_algebraic(self, move):
        """Convert move tuple to algebraic notation"""
        if move is None:
            return "None"

        from_row, from_col, to_row, to_col = move
        from_square = chr(ord("a") + from_col) + str(from_row + 1)
        to_square = chr(ord("a") + to_col) + str(to_row + 1)
        return f"{from_square}{to_square}"

    def play_game(self, opponent_color="black", depth=4):
        """Play a game against the AI"""
        board = ChessBoard("")  # Start with empty moves to get initial position

        print("Chess AI Game Started!")
        print("Enter moves in algebraic notation (e.g., e2e4)")
        print("Type 'quit' to exit")

        while True:
            board.print_board()
            legal_moves = board.generate_legal_moves()

            if not legal_moves:
                if board.is_king_in_check():
                    winner = "Black" if board.is_white_to_move() else "White"
                    print(f"Checkmate! {winner} wins!")
                else:
                    print("Stalemate! Draw!")
                break

            if (board.is_white_to_move() and opponent_color == "black") or (
                not board.is_white_to_move() and opponent_color == "white"
            ):
                # Human turn
                print(f"Your turn ({'White' if board.is_white_to_move() else 'Black'})")

                while True:
                    try:
                        move_input = input("Enter move: ").strip().lower()
                        if move_input == "quit":
                            return

                        # Parse algebraic notation (e.g., "e2e4")
                        if len(move_input) == 4:
                            from_col = ord(move_input[0]) - ord("a")
                            from_row = int(move_input[1]) - 1
                            to_col = ord(move_input[2]) - ord("a")
                            to_row = int(move_input[3]) - 1

                            move = (from_row, from_col, to_row, to_col)

                            if move in legal_moves:
                                board.make_move(move)
                                break
                            else:
                                print("Illegal move! Try again.")
                        else:
                            print("Invalid format! Use format like 'e2e4'")
                    except (ValueError, IndexError):
                        print("Invalid move format! Use format like 'e2e4'")
            else:
                # AI turn
                print(
                    f"AI thinking... ({'White' if board.is_white_to_move() else 'Black'})"
                )

                ai_move, score = self.get_best_move(board, depth=depth, time_limit=5.0)

                if ai_move:
                    print(
                        f"AI plays: {self.move_to_algebraic(ai_move)} (Score: {score:.3f})"
                    )
                    board.make_move(ai_move)
                else:
                    print("AI couldn't find a move!")
                    break


if __name__ == "__main__":
    # Initialize the Chess AI
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ai = ChessAI(device=device)

    # Start a game
    print("Welcome to Chess AI!")
    opponent_color = input("Choose your color (white/black): ").strip().lower()
    if opponent_color not in ["white", "black"]:
        opponent_color = "white"

    depth = int(input("Choose AI depth (1-6, recommended 4): ") or 4)

    ai.play_game(opponent_color=opponent_color, depth=depth)
