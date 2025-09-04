import io
import time

import cairosvg
import chess
import chess.svg
import numpy as np
import pygame
import torch
from PIL import Image

from nnue_model import HalfKPFeatureExtractor, create_nnue_model


class ChessAI:
    def __init__(self, model_path="checkpoints/best_model.pth", device="mps"):
        """Initialize the Chess AI with trained NNUE evaluation function"""
        self.device = device

        # Create NNUE model
        self.model = create_nnue_model()

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

        # Initialize feature extractor
        self.feature_extractor = HalfKPFeatureExtractor()

        print(f"Chess AI initialized with NNUE model on {device}")

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

    def evaluate_position(self, board):
        """Evaluate a chess position using the trained NNUE model"""
        # Convert board to numpy array
        board_array = self._board_to_array(board)

        # Find king squares
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        if white_king_sq is None or black_king_sq is None:
            # Invalid position - return neutral evaluation
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

        # Convert centipawns to normalized scale for alpha-beta search
        # Clamp extreme values and normalize to reasonable range
        evaluation_cp = max(-3000, min(3000, evaluation_cp))
        normalized_eval = evaluation_cp / 100.0  # Convert to "pawn units"
        # NNUE returns evaluation from White's perspective (positive = good for White)
        # No need to adjust based on turn - minimax handles perspective

        return normalized_eval

    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ):
        """Alpha-beta pruning minimax algorithm"""
        if depth == 0:
            eval_score = self.evaluate_position(board)
            return eval_score, None

        legal_moves = list(board.legal_moves)

        # Terminal node (no legal moves)
        if not legal_moves:
            if board.is_check():
                # Checkmate - return score based on who's checkmated
                if board.turn == chess.WHITE:  # White is checkmated
                    return -1000 + depth, None  # Very bad for White
                else:  # Black is checkmated
                    return 1000 - depth, None  # Very good for White
            else:
                # Stalemate
                return 0, None

        best_move = None

        if board.turn == chess.WHITE:  # White maximizes
            max_eval = float("-inf")
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:  # Black minimizes
            min_eval = float("inf")
            for move in legal_moves:
                board.push(move)
                eval_score, _ = self.alpha_beta(board, depth - 1, alpha, beta)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

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
                )

                if move is not None:
                    best_move = move
                    best_score = score

                elapsed_time = time.time() - start_time
                print(
                    f"Depth {current_depth}: Best move {best_move}, "
                    f"Score: {best_score:.3f}, Time: {elapsed_time:.2f}s"
                )

                if time_limit and elapsed_time > time_limit * 0.8:
                    break

            except KeyboardInterrupt:
                break

        return best_move, best_score


class ChessGUI:
    def __init__(self, ai, human_color="black", depth=4):
        pygame.init()

        self.ai = ai
        self.board = chess.Board()
        self.human_color = human_color
        self.depth = depth
        self.selected_square = None
        self.highlighted_moves = []

        # Display setup
        self.SQUARE_SIZE = 70
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.WINDOW_WIDTH = self.BOARD_SIZE + 40
        self.WINDOW_HEIGHT = self.BOARD_SIZE + 100

        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI")

        # Colors
        self.LIGHT_BROWN = (240, 217, 181)
        self.DARK_BROWN = (181, 136, 99)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (144, 238, 144)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Font
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)

        # Cache for piece surfaces
        self.piece_surfaces = {}

        self.clock = pygame.time.Clock()
        self.running = True
        self.ai_thinking = False

        # Schedule AI move if it goes first
        self.ai_move_timer = 0
        if self.human_color == "black":
            self.ai_move_timer = pygame.time.get_ticks() + 1000  # 1 second delay

    def draw_board(self):
        """Draw the chess board"""
        for row in range(8):
            for col in range(8):
                x = col * self.SQUARE_SIZE + 20
                y = (7 - row) * self.SQUARE_SIZE + 20  # Flip board for white on bottom

                # Determine square color
                if (row + col) % 2 == 0:
                    color = self.LIGHT_BROWN
                else:
                    color = self.DARK_BROWN

                # Highlight selected square
                if self.selected_square == (row, col):
                    color = self.YELLOW
                # Highlight possible moves
                elif (row, col) in [
                    (
                        chess.square_rank(move.to_square),
                        chess.square_file(move.to_square),
                    )
                    for move in self.highlighted_moves
                ]:
                    color = self.GREEN

                pygame.draw.rect(
                    self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
                )
                pygame.draw.rect(
                    self.screen,
                    self.BLACK,
                    (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE),
                    1,
                )

    def get_piece_surface(self, chess_piece):
        """Get pygame surface for a chess piece using SVG rendering"""
        piece_key = str(chess_piece)
        if piece_key in self.piece_surfaces:
            return self.piece_surfaces[piece_key]

        # Generate SVG for the piece
        svg_string = chess.svg.piece(chess_piece, size=self.SQUARE_SIZE - 10)

        try:
            # Convert SVG to PNG bytes using cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"))

            if png_bytes is None:
                raise Exception("Failed to convert SVG to PNG")

            # Load PNG into PIL Image
            pil_image = Image.open(io.BytesIO(png_bytes))

            # Convert PIL to RGBA if not already
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")

            # Convert PIL image to pygame surface
            pygame_image = pygame.image.fromstring(
                pil_image.tobytes(), pil_image.size, "RGBA"
            )

            # Cache the surface
            self.piece_surfaces[piece_key] = pygame_image
            return pygame_image

        except Exception:
            # Fallback to simple colored rectangle if SVG rendering fails
            surface = pygame.Surface((self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10))
            color = (255, 255, 255) if chess_piece.color == chess.WHITE else (0, 0, 0)
            surface.fill(color)
            self.piece_surfaces[piece_key] = surface
            return surface

    def draw_pieces(self):
        """Draw chess pieces using SVG rendering"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                col = chess.square_file(square)
                row = chess.square_rank(square)
                x = col * self.SQUARE_SIZE + 20 + 5  # 5px margin
                y = (7 - row) * self.SQUARE_SIZE + 20 + 5  # 5px margin

                piece_surface = self.get_piece_surface(piece)
                self.screen.blit(piece_surface, (x, y))

    def draw_status(self):
        """Draw game status"""
        current_player = "White" if self.board.turn == chess.WHITE else "Black"

        if self.ai_thinking:
            status_text = "AI thinking..."
        elif self.is_ai_turn():
            status_text = f"AI's turn ({current_player})"
        else:
            status_text = f"Your turn ({current_player})"

        text_surface = self.small_font.render(status_text, True, self.BLACK)
        self.screen.blit(text_surface, (20, self.BOARD_SIZE + 30))

    def handle_click(self, pos):
        """Handle mouse clicks"""
        if self.is_ai_turn() or self.ai_thinking:
            return

        x, y = pos
        if x < 20 or x > 20 + self.BOARD_SIZE or y < 20 or y > 20 + self.BOARD_SIZE:
            return

        col = (x - 20) // self.SQUARE_SIZE
        row = 7 - ((y - 20) // self.SQUARE_SIZE)  # Flip for white on bottom

        if row < 0 or row > 7 or col < 0 or col > 7:
            return

        legal_moves = list(self.board.legal_moves)

        # If no square selected, select this square if it has a piece
        if self.selected_square is None:
            square = chess.square(col, row)
            piece = self.board.piece_at(square)
            if piece:
                # Check if it's the right color's turn
                if piece.color == self.board.turn:
                    self.selected_square = (row, col)
                    # Highlight possible moves from this square
                    from_square = chess.square(col, row)
                    self.highlighted_moves = [
                        move for move in legal_moves if move.from_square == from_square
                    ]
        else:
            # Try to move to clicked square
            from_row, from_col = self.selected_square
            from_square = chess.square(from_col, from_row)
            to_square = chess.square(col, row)

            # Find the matching move from legal moves
            matching_move = None
            for move in legal_moves:
                if move.from_square == from_square and move.to_square == to_square:
                    matching_move = move
                    break

            if matching_move:
                # Valid move
                self.board.push(matching_move)
                self.selected_square = None
                self.highlighted_moves = []

                # Check for game end
                if self.check_game_end():
                    return

                # Schedule AI move
                self.ai_move_timer = pygame.time.get_ticks() + 500  # 0.5 second delay
            else:
                # Invalid move or selecting new piece
                square = chess.square(col, row)
                piece = self.board.piece_at(square)
                if piece:
                    if piece.color == self.board.turn:
                        self.selected_square = (row, col)
                        from_square = chess.square(col, row)
                        self.highlighted_moves = [
                            move
                            for move in legal_moves
                            if move.from_square == from_square
                        ]
                else:
                    self.selected_square = None
                    self.highlighted_moves = []

    def ai_move(self):
        """Make AI move"""
        if not self.is_ai_turn():
            return

        self.ai_thinking = True
        ai_move, score = self.ai.get_best_move(
            self.board, depth=self.depth, time_limit=5.0
        )

        if ai_move:
            self.board.push(ai_move)
            print(f"AI plays: {ai_move} (Score: {score:.3f})")

        self.ai_thinking = False
        self.check_game_end()

    def is_ai_turn(self):
        """Check if it's AI's turn"""
        return (self.board.turn == chess.WHITE and self.human_color == "black") or (
            self.board.turn == chess.BLACK and self.human_color == "white"
        )

    def check_game_end(self):
        """Check if game has ended"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            print(f"Checkmate! {winner} wins!")
            return True
        elif self.board.is_stalemate():
            print("Stalemate! Draw!")
            return True
        elif self.board.is_insufficient_material():
            print("Draw by insufficient material!")
            return True
        elif self.board.is_seventyfive_moves():
            print("Draw by 75-move rule!")
            return True
        elif self.board.is_fivefold_repetition():
            print("Draw by fivefold repetition!")
            return True
        return False

    def run(self):
        """Main game loop"""
        while self.running:
            current_time = pygame.time.get_ticks()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            # AI move timing
            if (
                self.ai_move_timer > 0
                and current_time >= self.ai_move_timer
                and self.is_ai_turn()
                and not self.ai_thinking
            ):
                self.ai_move_timer = 0
                self.ai_move()

            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            self.draw_pieces()
            self.draw_status()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    # Initialize the Chess AI
    ai = ChessAI(device="cpu")

    # Simple command line setup for now
    print("Welcome to Chess AI!")
    human_color = input("Choose your color (white/black) [white]: ").strip().lower()
    if human_color not in ["white", "black"]:
        human_color = "white"

    try:
        depth = int(input("Choose AI depth (0-3) [1]: ") or 1)
        if depth < 0 or depth > 3:
            depth = 1
        # Convert to odd depth to avoid evaluation perspective issues
        odd_depth = depth * 2 + 1
    except ValueError:
        odd_depth = 3

    # Start the chess GUI
    gui = ChessGUI(ai, human_color=human_color, depth=odd_depth)
    gui.run()
