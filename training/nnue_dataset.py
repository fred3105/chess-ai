import logging
import os
import random
import struct

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.utils.data as data
from nnue_model import HalfKPFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessPosition:
    """Represents a chess position with evaluation"""

    # Binary format: 48 bytes total for efficient storage
    STRUCT_FORMAT = '<32s B B H f f B 11x'  # little-endian, 48 bytes total
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

    def __init__(
        self,
        fen: str,
        evaluation: float,
        outcome: float | None = None,
        position_type: str = "quiet",
    ):
        self.fen = fen
        self.evaluation = evaluation  # Engine evaluation in centipawns
        self.outcome = (
            outcome  # Game outcome: 1.0 (white win), 0.0 (draw), -1.0 (black win)
        )
        self.position_type = (
            position_type  # Position type: quiet, tactical, imbalanced, unusual
        )

    def to_bytes(self) -> bytes:
        """Convert position to binary format for efficient storage"""
        board = chess.Board(self.fen)

        # Pack board state (32 bytes)
        board_bytes = self._pack_board(board)

        # Pack metadata
        castling_turn = (self._pack_castling(board) << 4) | (1 if board.turn else 0)
        en_passant = board.ep_square if board.ep_square else 255
        clock_move = (board.halfmove_clock << 8) | min(board.fullmove_number, 255)

        # Position type as integer
        type_map = {"quiet": 0, "tactical": 1, "imbalanced": 2, "unusual": 3}
        pos_type_int = type_map.get(self.position_type, 0)

        return struct.pack(
            self.STRUCT_FORMAT,
            board_bytes,
            castling_turn,
            en_passant,
            clock_move,
            self.evaluation,
            self.outcome if self.outcome is not None else -999.0,  # Special value for None
            pos_type_int
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ChessPosition':
        """Create position from binary format"""
        if len(data) != cls.STRUCT_SIZE:
            raise ValueError(f"Invalid data size: {len(data)}, expected {cls.STRUCT_SIZE}")

        unpacked = struct.unpack(cls.STRUCT_FORMAT, data)
        board_bytes, castling_turn, en_passant, clock_move, evaluation, outcome, pos_type_int = unpacked

        # Reconstruct board
        board = cls._unpack_board(board_bytes)
        board.turn = bool(castling_turn & 0x0F)

        # Restore castling rights properly
        castling_rights_val = (castling_turn >> 4) & 0x0F
        board.castling_rights = 0
        if castling_rights_val & 0x01:
            board.castling_rights |= chess.BB_H1  # White kingside
        if castling_rights_val & 0x02:
            board.castling_rights |= chess.BB_A1  # White queenside
        if castling_rights_val & 0x04:
            board.castling_rights |= chess.BB_H8  # Black kingside
        if castling_rights_val & 0x08:
            board.castling_rights |= chess.BB_A8  # Black queenside

        board.ep_square = en_passant if en_passant != 255 else None
        board.halfmove_clock = (clock_move >> 8) & 0xFF
        board.fullmove_number = clock_move & 0xFF

        # Position type
        type_map = {0: "quiet", 1: "tactical", 2: "imbalanced", 3: "unusual"}
        position_type = type_map.get(pos_type_int, "quiet")

        return cls(board.fen(), evaluation, outcome if outcome != -999.0 else None, position_type)

    def _pack_board(self, board: chess.Board) -> bytes:
        """Pack board state into 32 bytes (4 bits per square)"""
        piece_map = {
            None: 0,
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }

        packed = bytearray(32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            value = 0
            if piece:
                value = piece_map[piece.piece_type]
                if not piece.color:  # Black pieces
                    value |= 0x08  # Set high bit

            byte_idx = square // 2
            if square % 2 == 0:
                packed[byte_idx] |= value
            else:
                packed[byte_idx] |= (value << 4)

        return bytes(packed)

    @classmethod
    def _unpack_board(cls, board_bytes: bytes) -> chess.Board:
        """Unpack board state from 32 bytes"""
        board = chess.Board(None)  # Empty board
        board.clear()

        piece_types = {
            1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP,
            4: chess.ROOK, 5: chess.QUEEN, 6: chess.KING
        }

        for square in chess.SQUARES:
            byte_idx = square // 2
            if square % 2 == 0:
                value = board_bytes[byte_idx] & 0x0F
            else:
                value = (board_bytes[byte_idx] >> 4) & 0x0F

            if value != 0:
                piece_type = piece_types[value & 0x07]
                color = chess.WHITE if (value & 0x08) == 0 else chess.BLACK
                board.set_piece_at(square, chess.Piece(piece_type, color))

        return board

    def _pack_castling(self, board: chess.Board) -> int:
        """Pack castling rights into 4 bits"""
        rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            rights |= 0x01
        if board.has_queenside_castling_rights(chess.WHITE):
            rights |= 0x02
        if board.has_kingside_castling_rights(chess.BLACK):
            rights |= 0x04
        if board.has_queenside_castling_rights(chess.BLACK):
            rights |= 0x08
        return rights


class NNUEDataset(data.Dataset):
    """Dataset for NNUE training with chunked loading and epoch compositing"""

    def __init__(
        self,
        positions: list[ChessPosition] | str,
        feature_extractor: HalfKPFeatureExtractor,
        chunk_sample_rate: float = 1.0,
        epoch_compositing: bool = False
    ):
        self.feature_extractor = feature_extractor
        self.chunk_sample_rate = chunk_sample_rate
        self.epoch_compositing = epoch_compositing
        self.epoch_count = 0

        if isinstance(positions, str):
            # Chunked dataset mode
            self.chunk_dir = positions
            from create_chunked_dataset import load_binary_metadata
            self.metadata = load_binary_metadata(positions)
            self.chunk_mode = True
            self.current_positions = []
            self._load_new_epoch()
        else:
            # Traditional mode with list of positions
            self.positions = positions
            self.chunk_mode = False

    def _load_new_epoch(self):
        """Load positions for new epoch with chunk sampling"""
        if not self.chunk_mode:
            return

        from create_chunked_dataset import _load_binary_chunk
        from pathlib import Path

        chunk_path = Path(self.chunk_dir)
        chunk_files = list(chunk_path.glob("positions_chunk_*.bin"))
        chunk_files.sort()

        # Sample subset of chunks for this epoch
        chunks_to_use = max(1, int(len(chunk_files) * self.chunk_sample_rate))
        selected_chunks = random.sample(chunk_files, chunks_to_use)

        logger.info(f"Epoch {self.epoch_count}: Loading {len(selected_chunks)} chunks out of {len(chunk_files)}")

        self.current_positions = []
        for chunk_file in selected_chunks:
            positions = _load_binary_chunk(chunk_file)
            self.current_positions.extend(positions)

        # Shuffle positions across chunks
        random.shuffle(self.current_positions)
        logger.info(f"Loaded {len(self.current_positions):,} positions for epoch {self.epoch_count}")

    def __len__(self) -> int:
        if self.chunk_mode:
            return len(self.current_positions)
        return len(self.positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle epoch compositing for chunked mode
        if self.chunk_mode and self.epoch_compositing and idx >= len(self.current_positions):
            self.epoch_count += 1
            self._load_new_epoch()
            idx = idx % len(self.current_positions)

        if self.chunk_mode:
            position = self.current_positions[idx]
        else:
            position = self.positions[idx]

        board = chess.Board(position.fen)

        # Convert board to numpy array
        board_array = self._board_to_array(board)

        # Find king squares
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        if white_king_sq is None or black_king_sq is None:
            # Invalid position, return zero features
            return (
                torch.zeros(self.feature_extractor.feature_dim),
                torch.zeros(self.feature_extractor.feature_dim),
                torch.tensor([0.0]),
            )

        # Extract HalfKP features
        white_features, black_features = self.feature_extractor.extract_halfkp_features(
            board_array, white_king_sq, black_king_sq
        )

        # Target evaluation
        target = torch.tensor([position.evaluation], dtype=torch.float32)

        return white_features, black_features, target

    def refresh_epoch(self):
        """Manually trigger loading of new epoch data"""
        if self.chunk_mode and self.epoch_compositing:
            self.epoch_count += 1
            self._load_new_epoch()

    def _board_to_array(self, board: chess.Board) -> np.ndarray:
        """Convert chess board to 8x8 numpy array"""
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


class PositionType:
    """Enum for position types"""

    QUIET = "quiet"
    TACTICAL = "tactical"
    IMBALANCED = "imbalanced"
    UNUSUAL = "unusual"


class DatasetGenerator:
    """Generate training datasets from various sources with careful position selection"""

    def __init__(self, engine_path: str | None = None):
        self.engine_path = engine_path
        self.feature_extractor = HalfKPFeatureExtractor()

    def generate_from_pgn(
        self,
        pgn_file: str,
        max_positions: int | float,
    ) -> list[ChessPosition]:
        """Generate positions from PGN file"""
        positions = []

        with open(pgn_file) as f:
            while len(positions) < max_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Extract positions from game
                game_positions = self._extract_positions_from_game(game)
                # Take all positions from game, up to remaining limit
                if max_positions == float('inf'):
                    positions.extend(game_positions)
                else:
                    remaining = max_positions - len(positions)
                    positions.extend(game_positions[:remaining])

                if len(positions) % 10000 == 0:
                    logger.info(f"Extracted {len(positions)} positions from {pgn_file}")

        return positions

    def _extract_positions_from_game(self, game: chess.pgn.Game) -> list[ChessPosition]:
        """Extract positions from a single game"""
        positions = []
        board = game.board()

        # Get game outcome (using standard -1, 0, 1 encoding)
        result = game.headers.get("Result", "*")
        if result == "1-0":
            outcome = 1.0  # White wins
        elif result == "0-1":
            outcome = -1.0  # Black wins
        elif result == "1/2-1/2":
            outcome = 0.0  # Draw
        else:
            return positions

        move_count = 0
        for move in game.mainline_moves():
            board.push(move)
            move_count += 1

            # Quality filter 1: Skip early opening and late endgame moves
            if move_count < 8 or len(board.piece_map()) < 8:
                continue

            # Classify position type
            pos_type = self._classify_position(board)

            # Quality filter 2: Filter tactical positions
            if pos_type == PositionType.TACTICAL:
                continue

            # Quality filter 3: Check material balance (avoid extreme imbalances)
            material_imbalance = abs(self._calculate_material_imbalance(board))
            if material_imbalance > 500:  # More than 5 pawns difference
                continue

            # Quality filter 4: Balanced sampling rates - 60% quiet, 20% imbalanced, 20% unusual
            if pos_type == PositionType.QUIET:
                sample_rate = 0.15  # Higher rate for quiet positions (60% of final dataset)
            elif pos_type == PositionType.IMBALANCED:
                sample_rate = 0.08  # Medium rate for imbalanced positions (20% of final dataset)
            elif pos_type == PositionType.UNUSUAL:
                sample_rate = 0.08  # Medium rate for unusual positions (20% of final dataset)
            else:
                sample_rate = 0.10  # Default rate

            if random.random() < sample_rate:
                # Evaluation based on material and outcome - more sophisticated for GM games
                eval_score = self._sophisticated_evaluation(board, outcome, move_count)

                # Quality filter 5: Filter by evaluation range for balanced training
                if abs(eval_score) > 800:  # Skip positions with extreme evaluations
                    continue

                position = ChessPosition(board.fen(), eval_score, outcome, pos_type)
                positions.append(position)

        return positions

    def _sophisticated_evaluation(
        self, board: chess.Board, outcome: float, move_count: int
    ) -> float:
        """More sophisticated evaluation for grandmaster games"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        white_material = 0
        black_material = 0
        white_development = 0
        black_development = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]

                if piece.color == chess.WHITE:
                    white_material += value
                    # Development bonus for pieces off starting squares
                    if piece.piece_type in [
                        chess.KNIGHT,
                        chess.BISHOP,
                    ] and square not in [1, 2, 5, 6]:
                        white_development += 20
                else:
                    black_material += value
                    if piece.piece_type in [
                        chess.KNIGHT,
                        chess.BISHOP,
                    ] and square not in [57, 58, 61, 62]:
                        black_development += 20

        material_eval = white_material - black_material
        development_eval = white_development - black_development

        # Positional evaluation based on phase of game
        phase_eval = 0
        if move_count < 15:  # Opening
            phase_eval = development_eval * 0.5
        elif move_count > 40:  # Endgame
            phase_eval = -abs(material_eval) * 0.1 if abs(material_eval) < 200 else 0

        # Outcome influence (stronger for GM games as they're more decisive)
        outcome_influence = outcome * 75  # +/- 75cp influence

        # Game phase adjustment
        game_phase_factor = 1.0
        if move_count < 10:  # Very early
            game_phase_factor = 0.8
        elif move_count > 60:  # Late endgame
            game_phase_factor = 1.2

        final_eval = (
            material_eval + phase_eval + outcome_influence
        ) * game_phase_factor

        # Perspective from current player
        if not board.turn:  # Black to move
            final_eval = -final_eval

        return float(np.clip(final_eval, -3000, 3000))

    def _classify_position(self, board: chess.Board) -> str:
        """Classify position type for balanced dataset creation"""
        # Check for tactical positions (checks, significant captures)
        if board.is_check():
            return PositionType.TACTICAL

        # Only consider significant captures (not just any capture)
        significant_captures = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece and captured_piece.piece_type in [
                    chess.QUEEN,
                    chess.ROOK,
                    chess.BISHOP,
                    chess.KNIGHT,
                ]:
                    significant_captures += 1
        # Only tactical if there are multiple significant captures available
        if significant_captures >= 2:
            return PositionType.TACTICAL

        # Check material imbalance
        material_imbalance = self._calculate_material_imbalance(board)
        if abs(material_imbalance) >= 300:  # More than 3 pawns difference
            return PositionType.IMBALANCED

        # Check for unusual positions (few pieces, pawn endgames, etc.)
        piece_count = len(board.piece_map())
        if piece_count < 10:  # Endgame positions
            return PositionType.UNUSUAL

        # Check for unusual material distribution
        if self._has_unusual_material(board):
            return PositionType.UNUSUAL

        return PositionType.QUIET

    def _calculate_material_imbalance(self, board: chess.Board) -> float:
        """Calculate material imbalance between sides"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        return white_material - black_material

    def _has_unusual_material(self, board: chess.Board) -> bool:
        """Check for unusual material distributions"""
        white_pieces = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}
        black_pieces = {"Q": 0, "R": 0, "B": 0, "N": 0, "P": 0}

        piece_map = {
            chess.QUEEN: "Q",
            chess.ROOK: "R",
            chess.BISHOP: "B",
            chess.KNIGHT: "N",
            chess.PAWN: "P",
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_char = piece_map[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_pieces[piece_char] += 1
                else:
                    black_pieces[piece_char] += 1

        # Check for unusual patterns
        total_pieces = sum(white_pieces.values()) + sum(black_pieces.values())

        # Very few pieces (endgame)
        if total_pieces < 8:
            return True

        # Multiple queens
        if white_pieces["Q"] > 1 or black_pieces["Q"] > 1:
            return True

        # No queens but many pieces (unusual trade pattern)
        if white_pieces["Q"] == 0 and black_pieces["Q"] == 0 and total_pieces > 16:
            return True

        # Extreme pawn structure
        if white_pieces["P"] < 2 or black_pieces["P"] < 2:
            return True

        return False

    def _is_quiet_position(self, board: chess.Board) -> bool:
        """Check if position is quiet (no immediate tactics)"""
        return self._classify_position(board) == PositionType.QUIET

    def _simple_evaluation(self, board: chess.Board, outcome: float) -> float:
        """Simple material-based evaluation with outcome bias"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        material_eval = white_material - black_material

        # Bias evaluation towards game outcome
        outcome_bias = outcome * 100  # +/- 100cp bias

        final_eval = material_eval + outcome_bias

        # Perspective from white's point of view
        if not board.turn:  # Black to move
            final_eval = -final_eval

        return float(np.clip(final_eval, -3000, 3000))

    def generate_random_positions(
        self, num_positions: int = 10000
    ) -> list[ChessPosition]:
        """Generate random positions for testing"""
        positions = []

        for _ in range(num_positions):
            # Start from random opening
            board = chess.Board()

            # Play random moves
            for _ in range(random.randint(10, 40)):
                if board.is_game_over():
                    break
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))

            if not board.is_game_over() and len(board.piece_map()) >= 8:
                # Random evaluation
                eval_score = random.gauss(0, 100)  # Centered around 0
                positions.append(ChessPosition(board.fen(), eval_score))

        return positions


def create_balanced_datasets(
    data_dir: str = "data",
    train_size: int = 500000,
    val_size: int = 50000,
    pgn_files: list[str] | None = None,
    engine_path: str | None = None,
) -> tuple[NNUEDataset, NNUEDataset]:
    """Create carefully balanced training and validation datasets from grandmaster PGN files"""

    os.makedirs(data_dir, exist_ok=True)
    generator = DatasetGenerator(engine_path)

    all_positions = []

    # Generate from PGN files if provided
    target_positions = (train_size + val_size) * 3  # Generate more to allow filtering
    if pgn_files:
        for pgn_file in pgn_files:
            if os.path.exists(pgn_file):
                logger.info(f"Processing {pgn_file}")
                positions = generator.generate_from_pgn(
                    pgn_file,
                    max_positions=target_positions
                    - len(all_positions),  # Only get what we need
                )
                all_positions.extend(positions)
                logger.info(f"Found {len(positions)} positions from {pgn_file}")

                # Stop early if we have enough positions
                if len(all_positions) >= target_positions:
                    logger.info(
                        f"Collected sufficient positions ({len(all_positions)}), stopping early"
                    )
                    break
            else:
                logger.warning(f"PGN file not found: {pgn_file}")

    # Only use real positions from GM games - no random generation
    logger.info(f"Using only real positions from GM games: {len(all_positions)} total")

    # Separate positions by type
    quiet_positions = []
    tactical_positions = []
    imbalanced_positions = []
    unusual_positions = []

    for pos in all_positions:
        pos_type = pos.position_type
        if pos_type == PositionType.QUIET:
            quiet_positions.append(pos)
        elif pos_type == PositionType.TACTICAL:
            tactical_positions.append(pos)
        elif pos_type == PositionType.IMBALANCED:
            imbalanced_positions.append(pos)
        else:  # UNUSUAL
            unusual_positions.append(pos)

    logger.info(f"Total positions found: {len(all_positions)}")
    logger.info("Position distribution:")
    logger.info(f"  Quiet: {len(quiet_positions)}")
    logger.info(f"  Tactical: {len(tactical_positions)} (will be excluded)")
    logger.info(f"  Imbalanced: {len(imbalanced_positions)}")
    logger.info(f"  Unusual: {len(unusual_positions)}")

    # Log how many we can actually use
    usable_positions = (
        len(quiet_positions) + len(imbalanced_positions) + len(unusual_positions)
    )
    logger.info(f"Usable positions (quiet + imbalanced + unusual): {usable_positions}")

    target_total = train_size + val_size
    if usable_positions < target_total:
        logger.warning(
            f"Not enough usable positions! Requested {target_total}, using all available {usable_positions}"
        )
        # Use all available positions - adjust split to maintain roughly the same ratio
        if usable_positions > 0:
            train_ratio = train_size / target_total
            train_size = int(usable_positions * train_ratio)
            val_size = usable_positions - train_size
            logger.info(
                f"Using all positions - Train: {train_size}, Validation: {val_size}"
            )
        else:
            logger.error("No usable positions available!")
            train_size = val_size = 0
    else:
        logger.info(
            f"Sufficient positions available for target dataset size of {target_total}"
        )

    # Create balanced datasets
    def create_balanced_split(total_size: int) -> list[ChessPosition]:
        # Use all non-tactical positions if we don't have enough
        all_usable = quiet_positions + imbalanced_positions + unusual_positions

        if len(all_usable) <= total_size:
            # Use all available positions
            random.shuffle(all_usable)
            logger.info(f"Using all {len(all_usable)} available non-tactical positions")
            return all_usable
        else:
            # We have enough - try to maintain balance
            quiet_needed = total_size // 2

            selected_positions = []

            # Select quiet positions
            if len(quiet_positions) >= quiet_needed:
                selected_positions.extend(random.sample(quiet_positions, quiet_needed))
            else:
                selected_positions.extend(quiet_positions)

            # Select imbalanced positions only (exclude tactical)
            remaining_needed = total_size - len(selected_positions)
            remaining_pools = [imbalanced_positions, unusual_positions]
            remaining_pools = [pool for pool in remaining_pools if len(pool) > 0]

            if remaining_pools and remaining_needed > 0:
                per_type = remaining_needed // len(remaining_pools)
                extra = remaining_needed % len(remaining_pools)

                for i, pool in enumerate(remaining_pools):
                    take = per_type + (1 if i < extra else 0)
                    if len(pool) >= take:
                        selected_positions.extend(random.sample(pool, take))
                    else:
                        selected_positions.extend(pool)

            random.shuffle(selected_positions)
            return selected_positions[:total_size]

    # Create train and validation sets
    train_positions = create_balanced_split(train_size)
    val_positions = create_balanced_split(val_size)

    # Log final distribution
    def log_distribution(positions: list[ChessPosition], dataset_name: str):
        type_counts = {}
        for pos in positions:
            pos_type = pos.position_type
            type_counts[pos_type] = type_counts.get(pos_type, 0) + 1

        logger.info(f"{dataset_name} dataset ({len(positions)} positions):")
        for pos_type, count in type_counts.items():
            percentage = (count / len(positions)) * 100
            logger.info(f"  {pos_type}: {count} ({percentage:.1f}%)")

    log_distribution(train_positions, "Training")
    log_distribution(val_positions, "Validation")

    # Create datasets
    feature_extractor = HalfKPFeatureExtractor()
    train_dataset = NNUEDataset(train_positions, feature_extractor)
    val_dataset = NNUEDataset(val_positions, feature_extractor)

    return train_dataset, val_dataset


# Keep the old function for compatibility
def create_datasets(*args, **kwargs):
    """Legacy function - redirects to create_balanced_datasets"""
    return create_balanced_datasets(*args, **kwargs)


def save_positions(positions: list[ChessPosition], filename: str):
    """Save positions to file"""
    logger.info(f"Saving {len(positions)} positions to {filename}")

    with open(filename, "w") as f:
        for pos in positions:
            outcome_str = str(pos.outcome) if pos.outcome is not None else "None"
            f.write(f"{pos.fen}|{pos.evaluation}|{outcome_str}\n")


def load_positions(filename: str) -> list[ChessPosition]:
    """Load positions from file"""
    positions = []

    if not os.path.exists(filename):
        return positions

    logger.info(f"Loading positions from {filename}")

    with open(filename) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                fen = parts[0]
                evaluation = float(parts[1])
                outcome = (
                    float(parts[2]) if len(parts) > 2 and parts[2] != "None" else None
                )
                positions.append(ChessPosition(fen, evaluation, outcome))

    logger.info(f"Loaded {len(positions)} positions")
    return positions


if __name__ == "__main__":
    # Test dataset creation
    logger.info("Testing dataset creation...")

    # Create small test datasets
    train_dataset, val_dataset = create_datasets(
        train_size=1000,
        val_size=100,
        pgn_files=None,  # Will generate random positions
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Test data loading
    train_loader = data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2
    )

    batch = next(iter(train_loader))
    white_features, black_features, targets = batch

    logger.info("Batch shapes:")
    logger.info(f"  White features: {white_features.shape}")
    logger.info(f"  Black features: {black_features.shape}")
    logger.info(f"  Targets: {targets.shape}")
    logger.info(f"  Feature sparsity: {(white_features > 0).float().mean().item():.4f}")
    logger.info(
        f"  Target range: [{targets.min().item():.1f}, {targets.max().item():.1f}]"
    )
