import re
import numpy as np


class ChessBoard:
    def __init__(self, list_moves: str):
        self.board = np.zeros((8, 8), dtype=int)
        self.board_positions = []  # Store intermediate positions
        self.create_starting_board()
        self.board_positions.append(self.board.copy())  # Store initial position
        self.move_count = 0  # Track move number to determine whose turn it is
        self.white_has_castled = False
        self.black_has_castled = False
        self.update_board_with_moves(list_moves)

    def create_starting_board(self):
        # Create the initial positions of the pieces
        # Positive values = white, negative values = black
        # Pawn=1, Knight=2, Bishop=3, Rook=5, Queen=9, King=100
        # Row 0 = rank 1, Row 7 = rank 8
        self.board[0, :] = [5, 2, 3, 9, 100, 3, 2, 5]  # White pieces (rank 1)
        self.board[1, :] = 1  # White pawns (rank 2)
        self.board[6, :] = -1  # Black pawns (rank 7)
        self.board[7, :] = [-5, -2, -3, -9, -100, -3, -2, -5]  # Black pieces (rank 8)

    def get_board_positions(self):
        """Return list of all board positions throughout the game"""
        return self.board_positions

    def is_white_to_move(self):
        """Determine if it's white's turn based on move count"""
        return self.move_count % 2 == 0

    def update_board_with_moves(self, list_moves: str):
        # Clean up the moves string and split properly
        # Remove move numbers and handle the format better
        moves_cleaned = re.sub(r"\d+\.", " ", list_moves)  # Remove move numbers
        moves_cleaned = re.sub(
            r"\s+", " ", moves_cleaned
        ).strip()  # Normalize whitespace
        moves = moves_cleaned.split()

        for move in moves:
            if move in ["1-0", "0-1", "1/2-1/2"]:  # Skip game results
                continue
            if not move.strip():  # Skip empty moves
                continue

            # Clean the move of check/checkmate symbols
            clean_move = move.rstrip("+#")

            try:
                # Check for castle moves
                if clean_move == "O-O":
                    self.castle_king_side()
                elif clean_move == "O-O-O":
                    self.castle_queen_side()
                else:
                    self.parse_and_execute_move(clean_move)

                # Store the board position after the move
                self.board_positions.append(self.board.copy())
                # Increment move count
                self.move_count += 1

            except Exception as e:
                print("Current board:")
                print(self.print_board())
                raise RuntimeError(f"Play {self.move_count + 1} (‘{move}’) failed: {e}")

    def parse_and_execute_move(self, move: str):
        # Handle pawn moves (no piece letter, starts with lowercase)
        if re.match(r"^[a-h][1-8]$", move):
            self.move_pawn(move)
        elif re.match(r"^[a-h]x[a-h][1-8]$", move):  # Pawn capture
            self.move_pawn_capture(move)
        elif re.match(r"^[a-h][1-8]=[QRBN]$", move):  # Pawn promotion
            self.move_pawn_promotion(move)
        elif re.match(r"^[a-h]x[a-h][1-8]=[QRBN]$", move):  # Pawn capture promotion
            self.move_pawn_capture_promotion(move)
        else:
            # Handle piece moves
            self.move_piece(move)

    def move_pawn(self, move: str):
        col = ord(move[0]) - ord("a")
        row = int(move[1]) - 1  # Convert to 0-based indexing

        # Find the pawn that can move to this square
        if self.is_white_to_move():
            # White pawns move from lower rows to higher rows (rank 2 to rank 8)
            # Check one square back
            if row > 0 and self.board[row - 1, col] == 1:
                self.move_from_to(row - 1, col, row, col)
            # Check two squares back (initial pawn move)
            elif row == 3 and self.board[1, col] == 1 and self.board[2, col] == 0:
                self.move_from_to(1, col, row, col)
            else:
                raise ValueError(f"Invalid white pawn move: {move}")
        else:
            # Black pawns move from higher rows to lower rows (rank 7 to rank 1)
            # Check one square forward (higher row number)
            if row < 7 and self.board[row + 1, col] == -1:
                self.move_from_to(row + 1, col, row, col)
            # Check two squares forward (initial pawn move)
            elif row == 4 and self.board[6, col] == -1 and self.board[5, col] == 0:
                self.move_from_to(6, col, row, col)
            else:
                raise ValueError(f"Invalid black pawn move: {move}")

    def move_pawn_capture(self, move: str):
        # Format: exd5 (pawn on e-file captures on d5)
        from_col = ord(move[0]) - ord("a")
        to_col = ord(move[2]) - ord("a")
        to_row = int(move[3]) - 1

        if self.is_white_to_move():
            from_row = to_row - 1  # White pawns move up
            pawn_value = 1
        else:
            from_row = to_row + 1  # Black pawns move down
            pawn_value = -1

        if (
            self.valid_position(from_row, from_col)
            and self.board[from_row, from_col] == pawn_value
        ):
            self.move_from_to(from_row, from_col, to_row, to_col)
        else:
            raise ValueError(f"Invalid pawn capture: {move}")

    def move_pawn_promotion(self, move: str):
        # Format: e8=Q
        col = ord(move[0]) - ord("a")
        row = int(move[1]) - 1
        promoted_piece = move[3]

        piece_values = {"Q": 9, "R": 5, "B": 3, "N": 2}
        promoted_value = piece_values[promoted_piece]

        if self.is_white_to_move():
            if row == 7 and self.board[6, col] == 1:  # White pawn promoting to 8th rank
                self.board[6, col] = 0
                self.board[row, col] = promoted_value
            else:
                raise ValueError(f"Invalid white pawn promotion: {move}")
        else:
            if (
                row == 0 and self.board[1, col] == -1
            ):  # Black pawn promoting to 1st rank
                self.board[1, col] = 0
                self.board[row, col] = -promoted_value
            else:
                raise ValueError(f"Invalid black pawn promotion: {move}")

    def move_pawn_capture_promotion(self, move: str):
        # Format: exd8=Q
        from_col = ord(move[0]) - ord("a")
        to_col = ord(move[2]) - ord("a")
        to_row = int(move[3]) - 1
        promoted_piece = move[5]

        piece_values = {"Q": 9, "R": 5, "B": 3, "N": 2}
        promoted_value = piece_values[promoted_piece]

        if self.is_white_to_move():
            from_row = to_row - 1  # White pawns move up
            pawn_value = 1
            final_value = promoted_value
        else:
            from_row = to_row + 1  # Black pawns move down
            pawn_value = -1
            final_value = -promoted_value

        if (
            self.valid_position(from_row, from_col)
            and self.board[from_row, from_col] == pawn_value
        ):
            self.board[from_row, from_col] = 0
            self.board[to_row, to_col] = final_value
        else:
            raise ValueError(f"Invalid pawn capture promotion: {move}")

    def castle_king_side(self):
        if self.is_white_to_move():
            if self.white_has_castled:
                raise ValueError("White has already castled")
            if self.board[0, 4] != 100 or self.board[0, 7] != 5:
                raise ValueError("Invalid king-side castle")
            if self.board[0, 5] != 0 or self.board[0, 6] != 0:
                raise ValueError("Invalid king-side castle, pieces in the way")

            self.board[0, 4] = 0  # Remove king
            self.board[0, 6] = 100  # Place king
            self.board[0, 7] = 0  # Remove rook
            self.board[0, 5] = 5  # Place rook
            self.white_has_castled = True
        else:
            if self.black_has_castled:
                raise ValueError("Black has already castled")
            if self.board[7, 4] != -100 or self.board[7, 7] != -5:
                raise ValueError("Invalid king-side castle")
            if self.board[7, 5] != 0 or self.board[7, 6] != 0:
                raise ValueError("Invalid king-side castle, pieces in the way")

            self.board[7, 4] = 0  # Remove king
            self.board[7, 6] = -100  # Place king
            self.board[7, 7] = 0  # Remove rook
            self.board[7, 5] = -5  # Place rook
            self.black_has_castled = True

    def castle_queen_side(self):
        if self.is_white_to_move():
            if self.white_has_castled:
                raise ValueError("White has already castled")
            if self.board[0, 4] != 100 or self.board[0, 0] != 5:
                raise ValueError("Invalid queen-side castle")
            if self.board[0, 1] != 0 or self.board[0, 2] != 0 or self.board[0, 3] != 0:
                raise ValueError("Invalid queen-side castle, pieces in the way")

            self.board[0, 4] = 0  # Remove king
            self.board[0, 2] = 100  # Place king
            self.board[0, 0] = 0  # Remove rook
            self.board[0, 3] = 5  # Place rook
            self.white_has_castled = True
        else:
            if self.black_has_castled:
                raise ValueError("Black has already castled")
            if self.board[7, 4] != -100 or self.board[7, 0] != -5:
                raise ValueError("Invalid queen-side castle")
            if self.board[7, 1] != 0 or self.board[7, 2] != 0 or self.board[7, 3] != 0:
                raise ValueError("Invalid queen-side castle, pieces in the way")

            self.board[7, 4] = 0  # Remove king
            self.board[7, 2] = -100  # Place king
            self.board[7, 0] = 0  # Remove rook
            self.board[7, 3] = -5  # Place rook
            self.black_has_castled = True

    def is_king_in_check(self, board_state=None):
        """Check if the current player's king is in check"""
        if board_state is None:
            board_state = self.board

        king_value = 100 if self.is_white_to_move() else -100

        # Find king position
        king_pos = None
        for r in range(8):
            for c in range(8):
                if board_state[r, c] == king_value:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            raise ValueError("King not found on the board!")

        # Simple attack detection (only for filtering illegal moves)
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),  # Rook/Queen
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),  # Bishop/Queen
        ]
        knight_moves = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]

        kr, kc = king_pos

        # Check knights
        for dr, dc in knight_moves:
            r, c = kr + dr, kc + dc
            if self.valid_position(r, c):
                if board_state[r, c] == (-2 if self.is_white_to_move() else 2):
                    return True

        # Check sliding pieces
        for dr, dc in directions:
            r, c = kr + dr, kc + dc
            while self.valid_position(r, c):
                piece = board_state[r, c]
                if piece != 0:
                    if self.is_white_to_move():
                        if piece < 0 and (
                            (abs(piece) == 5 and (dr == 0 or dc == 0))
                            or (abs(piece) == 3 and abs(dr) == abs(dc))
                            or (abs(piece) == 9)
                        ):
                            return True
                    else:
                        if piece > 0 and (
                            (piece == 5 and (dr == 0 or dc == 0))
                            or (piece == 3 and abs(dr) == abs(dc))
                            or (piece == 9)
                        ):
                            return True
                    break
                r += dr
                c += dc

        # Check pawns
        pawn_dir = 1 if self.is_white_to_move() else -1
        for dc in [-1, 1]:
            r, c = kr + pawn_dir, kc + dc
            if self.valid_position(r, c):
                if board_state[r, c] == (-1 if self.is_white_to_move() else 1):
                    return True

        return False

    def move_piece(self, move: str):
        # Parse the move to extract piece type, destination, and disambiguation
        piece_char = move[0]

        # Handle captures and regular moves differently
        if "x" in move:
            # Format: Nxe4, Bxf7, Qxd8, etc.
            parts = move.split("x")
            piece_and_disambig = parts[0]
            target_square = parts[1]

            if len(piece_and_disambig) > 1:
                disambiguation = piece_and_disambig[1:]
            else:
                disambiguation = ""
        else:
            # Format: Ne4, Bf7, Qd8, etc.
            piece_and_disambig = move[:-2]
            target_square = move[-2:]

            if len(piece_and_disambig) > 1:
                disambiguation = piece_and_disambig[1:]
            else:
                disambiguation = ""

        if len(target_square) != 2:
            raise ValueError(f"Invalid move format: {move}")

        target_col = ord(target_square[0]) - ord("a")
        target_row = int(target_square[1]) - 1

        if piece_char == "N":
            self.move_knight(target_row, target_col, disambiguation)
        elif piece_char == "Q":
            self.move_queen(target_row, target_col, disambiguation)
        elif piece_char == "B":
            self.move_bishop(target_row, target_col, disambiguation)
        elif piece_char == "R":
            self.move_rook(target_row, target_col, disambiguation)
        elif piece_char == "K":
            self.move_king(target_row, target_col, disambiguation)
        else:
            raise ValueError(f"Unknown piece type: {piece_char}")

    def get_piece_value(self, piece_char: str) -> int:
        piece_values = {"K": 100, "Q": 9, "R": 5, "B": 3, "N": 2}
        base_value = piece_values.get(piece_char, 0)
        return base_value if self.is_white_to_move() else -base_value

    def valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def move_from_to(
        self, orig_row: int, orig_col: int, target_row: int, target_col: int
    ):
        if not self.valid_position(orig_row, orig_col) or not self.valid_position(
            target_row, target_col
        ):
            raise ValueError("Invalid move - position out of bounds")

        piece = self.board[orig_row, orig_col]
        self.board[orig_row, orig_col] = 0
        self.board[target_row, target_col] = piece

    def find_piece_positions(self, piece_value: int):
        """Find all positions of a specific piece on the board"""
        positions = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] == piece_value:
                    positions.append((r, c))
        return positions

    def move_knight(self, target_row: int, target_col: int, disambiguation: str):
        piece_value = self.get_piece_value("N")
        knight_positions = self.find_piece_positions(piece_value)

        # Find knights that can reach the target
        possible_origins = []
        knight_moves = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]

        for knight_row, knight_col in knight_positions:
            for dr, dc in knight_moves:
                if knight_row + dr == target_row and knight_col + dc == target_col:
                    # simulate the move
                    temp_board = self.board.copy()
                    temp_board[knight_row, knight_col] = 0
                    temp_board[target_row, target_col] = piece_value
                    if not self.is_king_in_check(temp_board):
                        possible_origins.append((knight_row, knight_col))

        if not possible_origins:
            raise ValueError(
                f"No knight can move to {chr(target_col + ord('a'))}{target_row + 1}"
            )

        origin = self.disambiguate_move(possible_origins, disambiguation)
        self.move_from_to(origin[0], origin[1], target_row, target_col)

    def move_rook(self, target_row: int, target_col: int, disambiguation: str):
        piece_value = self.get_piece_value("R")
        rook_positions = self.find_piece_positions(piece_value)

        possible_origins = []
        for rook_row, rook_col in rook_positions:
            if self.can_move_straight(rook_row, rook_col, target_row, target_col):
                # simulate move
                temp_board = self.board.copy()
                temp_board[rook_row, rook_col] = 0
                temp_board[target_row, target_col] = piece_value
                if not self.is_king_in_check(temp_board):
                    possible_origins.append((rook_row, rook_col))

        if not possible_origins:
            raise ValueError(
                f"No rook can move to {chr(target_col + ord('a'))}{target_row + 1}"
            )

        origin = self.disambiguate_move(possible_origins, disambiguation)
        self.move_from_to(origin[0], origin[1], target_row, target_col)

    def move_bishop(self, target_row: int, target_col: int, disambiguation: str):
        piece_value = self.get_piece_value("B")
        bishop_positions = self.find_piece_positions(piece_value)

        possible_origins = []
        for bishop_row, bishop_col in bishop_positions:
            if self.can_move_diagonal(bishop_row, bishop_col, target_row, target_col):
                # simulate move
                temp_board = self.board.copy()
                temp_board[bishop_row, bishop_col] = 0
                temp_board[target_row, target_col] = piece_value
                if not self.is_king_in_check(temp_board):
                    possible_origins.append((bishop_row, bishop_col))

        if not possible_origins:
            raise ValueError(
                f"No bishop can move to {chr(target_col + ord('a'))}{target_row + 1}"
            )

        origin = self.disambiguate_move(possible_origins, disambiguation)
        self.move_from_to(origin[0], origin[1], target_row, target_col)

    def move_queen(self, target_row: int, target_col: int, disambiguation: str):
        piece_value = self.get_piece_value("Q")
        queen_positions = self.find_piece_positions(piece_value)

        possible_origins = []
        for queen_row, queen_col in queen_positions:
            if self.can_move_straight(
                queen_row, queen_col, target_row, target_col
            ) or self.can_move_diagonal(queen_row, queen_col, target_row, target_col):
                # simulate move
                temp_board = self.board.copy()
                temp_board[queen_row, queen_col] = 0
                temp_board[target_row, target_col] = piece_value
                if not self.is_king_in_check(temp_board):
                    possible_origins.append((queen_row, queen_col))

        if not possible_origins:
            raise ValueError(
                f"No queen can move to {chr(target_col + ord('a'))}{target_row + 1}"
            )

        origin = self.disambiguate_move(possible_origins, disambiguation)
        self.move_from_to(origin[0], origin[1], target_row, target_col)

    def move_king(self, target_row: int, target_col: int, disambiguation: str):
        piece_value = self.get_piece_value("K")
        king_positions = self.find_piece_positions(piece_value)

        if len(king_positions) != 1:
            raise ValueError("Invalid king position on board")

        king_row, king_col = king_positions[0]

        # King can move one square in any direction
        if abs(king_row - target_row) <= 1 and abs(king_col - target_col) <= 1:
            # simulate move
            temp_board = self.board.copy()
            temp_board[king_row, king_col] = 0
            temp_board[target_row, target_col] = piece_value
            if self.is_king_in_check(temp_board):
                raise ValueError("King cannot move into check")
            self.move_from_to(king_row, king_col, target_row, target_col)
        else:
            raise ValueError(
                f"King cannot move to {chr(target_col + ord('a'))}{target_row + 1}"
            )

    def can_move_straight(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        """Check if there's a clear straight path between two squares"""
        if from_row != to_row and from_col != to_col:
            return False  # Not a straight line

        if from_row == to_row and from_col == to_col:
            return False  # Same square

        # Determine direction
        dr = 0 if from_row == to_row else (1 if to_row > from_row else -1)
        dc = 0 if from_col == to_col else (1 if to_col > from_col else -1)

        # Check path is clear
        r, c = from_row + dr, from_col + dc
        while r != to_row or c != to_col:
            if self.board[r, c] != 0:
                return False
            r += dr
            c += dc

        return True

    def can_move_diagonal(
        self, from_row: int, from_col: int, to_row: int, to_col: int
    ) -> bool:
        """Check if there's a clear diagonal path between two squares"""
        if abs(from_row - to_row) != abs(from_col - to_col):
            return False  # Not a diagonal

        if from_row == to_row and from_col == to_col:
            return False  # Same square

        # Determine direction
        dr = 1 if to_row > from_row else -1
        dc = 1 if to_col > from_col else -1

        # Check path is clear
        r, c = from_row + dr, from_col + dc
        while r != to_row or c != to_col:
            if self.board[r, c] != 0:
                return False
            r += dr
            c += dc

        return True

    def generate_legal_moves(self):
        """Generate all legal moves for the current player"""
        moves = []
        
        # Get all pieces for the current player
        for row in range(8):
            for col in range(8):
                piece = self.board[row, col]
                if piece == 0:
                    continue
                    
                # Check if piece belongs to current player
                if (self.is_white_to_move() and piece > 0) or (not self.is_white_to_move() and piece < 0):
                    piece_moves = self.get_piece_moves(row, col, piece)
                    moves.extend(piece_moves)
        
        return moves

    def get_piece_moves(self, row, col, piece):
        """Get all possible moves for a piece at given position"""
        moves = []
        abs_piece = abs(piece)
        
        if abs_piece == 1:  # Pawn
            moves.extend(self.get_pawn_moves(row, col, piece))
        elif abs_piece == 2:  # Knight
            moves.extend(self.get_knight_moves(row, col, piece))
        elif abs_piece == 3:  # Bishop
            moves.extend(self.get_bishop_moves(row, col, piece))
        elif abs_piece == 5:  # Rook
            moves.extend(self.get_rook_moves(row, col, piece))
        elif abs_piece == 9:  # Queen
            moves.extend(self.get_queen_moves(row, col, piece))
        elif abs_piece == 100:  # King
            moves.extend(self.get_king_moves(row, col, piece))
            
        return moves

    def get_pawn_moves(self, row, col, piece):
        """Generate pawn moves"""
        moves = []
        direction = 1 if piece > 0 else -1  # White moves up, black moves down
        start_row = 1 if piece > 0 else 6
        
        # Forward moves
        new_row = row + direction
        if self.valid_position(new_row, col) and self.board[new_row, col] == 0:
            if self.is_legal_move(row, col, new_row, col):
                moves.append((row, col, new_row, col))
            
            # Two squares forward from starting position
            if row == start_row:
                new_row = row + 2 * direction
                if self.valid_position(new_row, col) and self.board[new_row, col] == 0:
                    if self.is_legal_move(row, col, new_row, col):
                        moves.append((row, col, new_row, col))
        
        # Captures
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if self.valid_position(new_row, new_col):
                target = self.board[new_row, new_col]
                if target != 0 and ((piece > 0 and target < 0) or (piece < 0 and target > 0)):
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
        
        return moves

    def get_knight_moves(self, row, col, piece):
        """Generate knight moves"""
        moves = []
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self.valid_position(new_row, new_col):
                target = self.board[new_row, new_col]
                if target == 0 or ((piece > 0 and target < 0) or (piece < 0 and target > 0)):
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
        
        return moves

    def get_bishop_moves(self, row, col, piece):
        """Generate bishop moves"""
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.valid_position(new_row, new_col):
                    break
                    
                target = self.board[new_row, new_col]
                if target == 0:
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
                elif (piece > 0 and target < 0) or (piece < 0 and target > 0):
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
                    break
                else:
                    break
        
        return moves

    def get_rook_moves(self, row, col, piece):
        """Generate rook moves"""
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.valid_position(new_row, new_col):
                    break
                    
                target = self.board[new_row, new_col]
                if target == 0:
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
                elif (piece > 0 and target < 0) or (piece < 0 and target > 0):
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
                    break
                else:
                    break
        
        return moves

    def get_queen_moves(self, row, col, piece):
        """Generate queen moves (combination of rook and bishop)"""
        moves = []
        moves.extend(self.get_rook_moves(row, col, piece))
        moves.extend(self.get_bishop_moves(row, col, piece))
        return moves

    def get_king_moves(self, row, col, piece):
        """Generate king moves"""
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.valid_position(new_row, new_col):
                target = self.board[new_row, new_col]
                if target == 0 or ((piece > 0 and target < 0) or (piece < 0 and target > 0)):
                    if self.is_legal_move(row, col, new_row, new_col):
                        moves.append((row, col, new_row, new_col))
        
        return moves

    def is_legal_move(self, from_row, from_col, to_row, to_col):
        """Check if a move is legal (doesn't leave king in check)"""
        # Save original state
        original_piece = self.board[from_row, from_col]
        captured_piece = self.board[to_row, to_col]
        
        # Make the move temporarily
        self.board[from_row, from_col] = 0
        self.board[to_row, to_col] = original_piece
        
        # Check if king is in check after this move
        legal = not self.is_king_in_check()
        
        # Restore original state
        self.board[from_row, from_col] = original_piece
        self.board[to_row, to_col] = captured_piece
        
        return legal

    def make_move(self, move):
        """Make a move on the board"""
        from_row, from_col, to_row, to_col = move
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = 0
        self.move_count += 1

    def unmake_move(self, move, captured_piece):
        """Unmake a move on the board"""
        from_row, from_col, to_row, to_col = move
        self.board[from_row, from_col] = self.board[to_row, to_col]
        self.board[to_row, to_col] = captured_piece
        self.move_count -= 1

    def copy(self):
        """Create a deep copy of the board"""
        new_board = ChessBoard("")
        new_board.board = self.board.copy()
        new_board.move_count = self.move_count
        new_board.white_has_castled = self.white_has_castled
        new_board.black_has_castled = self.black_has_castled
        return new_board

    def disambiguate_move(self, possible_origins: list, disambiguation: str):
        """Handle disambiguation when multiple pieces can make the same move"""
        if len(possible_origins) == 1:
            return possible_origins[0]

        if not disambiguation:
            raise ValueError("Ambiguous move - need disambiguation")

        if len(disambiguation) == 1:
            if disambiguation.isalpha():  # File disambiguation (e.g., 'Nbd2')
                target_col = ord(disambiguation) - ord("a")
                for origin in possible_origins:
                    if origin[1] == target_col:
                        return origin
            else:  # Rank disambiguation (e.g., 'N1d2')
                target_row = int(disambiguation) - 1
                for origin in possible_origins:
                    if origin[0] == target_row:
                        return origin
        elif len(disambiguation) == 2:  # Full square disambiguation (e.g., 'Ne1d2')
            target_col = ord(disambiguation[0]) - ord("a")
            target_row = int(disambiguation[1]) - 1
            for origin in possible_origins:
                if origin[0] == target_row and origin[1] == target_col:
                    return origin

        raise ValueError(f"Could not disambiguate move with '{disambiguation}'")

    def print_board(self):
        """Print the current board state for debugging"""
        piece_symbols = {
            100: "K",
            9: "Q",
            5: "R",
            3: "B",
            2: "N",
            1: "P",
            -100: "k",
            -9: "q",
            -5: "r",
            -3: "b",
            -2: "n",
            -1: "p",
            0: ".",
        }

        print("  a b c d e f g h")
        for row in range(8):
            print(f"{8 - row} ", end="")
            for col in range(8):
                print(f"{piece_symbols[self.board[7 - row, col]]} ", end="")
            print(f"{8 - row}")
        print("  a b c d e f g h")


# Example usage and testing
if __name__ == "__main__":
    # Test with the example PGN from the original code
    pgn_moves = (
        "1. e4 c6 2. d4 g6 3. f4 d5 4. e5 h5 5. Bd3 Nh6 6. Ne2 Bf5 7. O-O e6 8. c4 "
        "dxc4 9. Bxc4 Nd7 10. Nbc3 Nb6 11. Bb3 Be7 12. Ng3 h4 13. Nxf5 Nxf5 14. Ne2 "
        "Nd5 15. Qd3 a5 16. Bd2 a4 17. Bc2 Qb6 18. Rab1 Nb4 19. Bxb4 Qxb4 20. a3 Qa5 "
        "21. Rf3 Rd8 22. Qc3 Qa7 23. Bxf5 gxf5 24. Rd3 Rd5 25. Rbd1 Rg8 26. Qe1 Rg4 "
        "27. h3 Rg6 28. Nc3 Ra5 29. Kh2 Ra6 30. R1d2 Bd8 31. Qd1 Rh6 32. d5 cxd5 33. "
        "Nxd5 Kf8 34. Nf6 Bxf6 35. exf6 Kg8 36. Rd8+ Kh7 37. R2d7 Kg6 38. Rg8+ Kxf6 "
        "39. Rf8 Rh7 40. Rdxf7+ Rxf7 41. Qd8+ Kg7 42. Qg5+ Kxf8 43. Qd8+ Kg7 44. "
        "Qg5+ Kh8 45. Qh6+ Kg8 46. Qg5+ Rg7 47. Qd8+ Kf7 48. Qd7+ Kf6 49. Qd8+ Kg6 "
        "50. Qe8+ Rf7 51. Qg8+ Kf6 52. Qg5# 1-0"
    )

    chess_board = ChessBoard(pgn_moves)
    positions = chess_board.get_board_positions()

    print(f"Game parsed successfully! Total positions: {len(positions)}")
    print("Final board position:")
    chess_board.print_board()
