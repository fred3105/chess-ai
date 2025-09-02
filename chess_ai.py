import torch
from custom_class import ChessBoard
from train import EvalNet, encode_board
import chess
import chess.svg
import time
import pygame
import io
from PIL import Image
import cairosvg


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

class ChessGUI:
    def __init__(self, ai, opponent_color="black", depth=4):
        pygame.init()
        
        self.ai = ai
        self.board = ChessBoard("")
        self.opponent_color = opponent_color
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
        
        # Piece mapping for chess.svg
        self.piece_mapping = {
            1: chess.PAWN, -1: chess.PAWN,
            2: chess.KNIGHT, -2: chess.KNIGHT,
            3: chess.BISHOP, -3: chess.BISHOP,
            5: chess.ROOK, -5: chess.ROOK,
            9: chess.QUEEN, -9: chess.QUEEN,
            100: chess.KING, -100: chess.KING
        }
        
        # Cache for piece surfaces
        self.piece_surfaces = {}
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.ai_thinking = False
        
        # Schedule AI move if it goes first
        self.ai_move_timer = 0
        if self.opponent_color == "black":
            self.ai_move_timer = pygame.time.get_ticks() + 1000  # 1 second delay
    
    def draw_board(self):
        """Draw the chess board"""
        for row in range(8):
            for col in range(8):
                x = col * self.SQUARE_SIZE + 20
                y = (7-row) * self.SQUARE_SIZE + 20  # Flip board for white on bottom
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = self.LIGHT_BROWN
                else:
                    color = self.DARK_BROWN
                
                # Highlight selected square
                if self.selected_square == (row, col):
                    color = self.YELLOW
                # Highlight possible moves
                elif (row, col) in [(move[2], move[3]) for move in self.highlighted_moves]:
                    color = self.GREEN
                
                pygame.draw.rect(self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE))
                pygame.draw.rect(self.screen, self.BLACK, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 1)
    
    def get_piece_surface(self, piece_value):
        """Get pygame surface for a chess piece using SVG rendering"""
        if piece_value in self.piece_surfaces:
            return self.piece_surfaces[piece_value]
        
        # Map piece value to chess.Piece
        piece_type = self.piece_mapping[abs(piece_value)]
        color = chess.WHITE if piece_value > 0 else chess.BLACK
        chess_piece = chess.Piece(piece_type, color)
        
        # Generate SVG for the piece
        svg_string = chess.svg.piece(chess_piece, size=self.SQUARE_SIZE - 10)
        
        try:
            # Convert SVG to PNG bytes using cairosvg
            png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
            
            # Load PNG into PIL Image
            pil_image = Image.open(io.BytesIO(png_bytes))
            
            # Convert PIL image to pygame surface
            image_string = pil_image.tobytes()
            pygame_image = pygame.image.fromstring(image_string, pil_image.size, pil_image.mode)
            
            # Cache the surface
            self.piece_surfaces[piece_value] = pygame_image
            return pygame_image
            
        except Exception:
            # Fallback to simple colored rectangle if SVG rendering fails
            surface = pygame.Surface((self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10))
            color = (255, 255, 255) if piece_value > 0 else (0, 0, 0)
            surface.fill(color)
            self.piece_surfaces[piece_value] = surface
            return surface
    
    def draw_pieces(self):
        """Draw chess pieces using SVG rendering"""
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row, col]
                if piece != 0:
                    x = col * self.SQUARE_SIZE + 20 + 5  # 5px margin
                    y = (7-row) * self.SQUARE_SIZE + 20 + 5  # 5px margin
                    
                    piece_surface = self.get_piece_surface(piece)
                    self.screen.blit(piece_surface, (x, y))
    
    def draw_status(self):
        """Draw game status"""
        current_player = "White" if self.board.is_white_to_move() else "Black"
        
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
        
        legal_moves = self.board.generate_legal_moves()
        
        # If no square selected, select this square if it has a piece
        if self.selected_square is None:
            piece = self.board.board[row, col]
            if piece != 0:
                # Check if it's the right color's turn
                is_white_piece = piece > 0
                if is_white_piece == self.board.is_white_to_move():
                    self.selected_square = (row, col)
                    # Highlight possible moves from this square
                    self.highlighted_moves = [move for move in legal_moves 
                                            if move[0] == row and move[1] == col]
        else:
            # Try to move to clicked square
            from_row, from_col = self.selected_square
            move = (from_row, from_col, row, col)
            
            if move in legal_moves:
                # Valid move
                self.board.make_move(move)
                self.selected_square = None
                self.highlighted_moves = []
                
                # Check for game end
                if self.check_game_end():
                    return
                
                # Schedule AI move
                self.ai_move_timer = pygame.time.get_ticks() + 500  # 0.5 second delay
            else:
                # Invalid move or selecting new piece
                piece = self.board.board[row, col]
                if piece != 0:
                    is_white_piece = piece > 0
                    if is_white_piece == self.board.is_white_to_move():
                        self.selected_square = (row, col)
                        self.highlighted_moves = [move for move in legal_moves 
                                                if move[0] == row and move[1] == col]
                else:
                    self.selected_square = None
                    self.highlighted_moves = []
    
    def ai_move(self):
        """Make AI move"""
        if not self.is_ai_turn():
            return
        
        self.ai_thinking = True
        ai_move, score = self.ai.get_best_move(self.board, depth=self.depth, time_limit=5.0)
        
        if ai_move:
            self.board.make_move(ai_move)
            print(f"AI plays: {self.ai.move_to_algebraic(ai_move)} (Score: {score:.3f})")
        
        self.ai_thinking = False
        self.check_game_end()
    
    def is_ai_turn(self):
        """Check if it's AI's turn"""
        return ((self.board.is_white_to_move() and self.opponent_color == "black") or
                (not self.board.is_white_to_move() and self.opponent_color == "white"))
    
    def check_game_end(self):
        """Check if game has ended"""
        legal_moves = self.board.generate_legal_moves()
        
        if not legal_moves:
            if self.board.is_king_in_check():
                winner = "Black" if self.board.is_white_to_move() else "White"
                print(f"Checkmate! {winner} wins!")
            else:
                print("Stalemate! Draw!")
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
            if (self.ai_move_timer > 0 and current_time >= self.ai_move_timer and 
                self.is_ai_turn() and not self.ai_thinking):
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ai = ChessAI(device=device)
    
    # Simple command line setup for now
    print("Welcome to Chess AI!")
    opponent_color = input("Choose your color (white/black) [white]: ").strip().lower()
    if opponent_color not in ["white", "black"]:
        opponent_color = "white"
    
    try:
        depth = int(input("Choose AI depth (1-6) [4]: ") or 4)
        if depth < 1 or depth > 6:
            depth = 4
    except ValueError:
        depth = 4
    
    # Start the chess GUI
    gui = ChessGUI(ai, opponent_color=opponent_color, depth=depth)
    gui.run()
