import io

import cairosvg
import chess
import chess.svg
import pygame
from PIL import Image


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
        self.WINDOW_HEIGHT = self.BOARD_SIZE + 120

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

        # Show evaluation mode if available
        if hasattr(self.ai, "get_evaluation_mode"):
            eval_mode = self.ai.get_evaluation_mode()
            eval_text = f"Evaluation: {eval_mode} (Press 'E' to toggle)"
            eval_surface = self.small_font.render(eval_text, True, self.BLACK)
            self.screen.blit(eval_surface, (20, self.BOARD_SIZE + 50))

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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_e and hasattr(
                        self.ai, "set_evaluation_mode"
                    ):
                        # Toggle evaluation mode with 'E' key
                        current_mode = self.ai.get_evaluation_mode()
                        new_mode = (
                            "NNUE"
                            if current_mode == "Deterministic"
                            else "Deterministic"
                        )
                        self.ai.set_evaluation_mode(new_mode.lower())

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
    print("Welcome to Chess AI!")

    # Choose evaluation mode
    print("Choose evaluation mode:")
    print("1. Deterministic (ultra-fast classical evaluation)")
    print("2. NNUE (your trained neural network)")

    eval_choice = input("Select evaluation mode (1-2) [1]: ").strip()

    if eval_choice == "2":
        # Use NNUE evaluation with pure C++
        from pure_cpp_chess_ai import PureCppChessAI

        ai = PureCppChessAI(evaluation_mode="nnue")
        depth = 3
    else:
        from pure_cpp_chess_ai import PureCppChessAI

        ai = PureCppChessAI(evaluation_mode="deterministic")
        depth = 7

    # Game setup
    human_color = input("Choose your color (white/black) [white]: ").strip().lower()
    if human_color not in ["white", "black"]:
        human_color = "white"

    # Start the chess GUI
    gui = ChessGUI(ai, human_color=human_color, depth=depth)
    gui.run()
