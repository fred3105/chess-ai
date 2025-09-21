#!/usr/bin/env python3
"""
Simple HalfKP NNUE model for chess evaluation
Clean implementation focused on working architecture
"""

import chess
import numpy as np
import torch
import torch.nn as nn


class HalfKPNNUE(nn.Module):
    """
    HalfKP NNUE model with proper HalfKP feature encoding
    Clean implementation of standard architecture
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()

        # True HalfKP: 64 king squares * (64 piece squares * 10 piece types)
        # = 64 * 640 = 40,960 features per king perspective
        self.input_size = 40960  # Features per king perspective
        self.hidden_size = hidden_size

        # Feature transformer (shared weights for both king perspectives)
        self.feature_transformer = nn.Linear(self.input_size, hidden_size, bias=True)

        # Network layers
        self.layer1 = nn.Linear(hidden_size * 2, 32)  # Concatenated white + black
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        # Feature transformer
        nn.init.uniform_(
            self.feature_transformer.weight,
            -1 / np.sqrt(self.input_size),
            1 / np.sqrt(self.input_size),
        )
        nn.init.zeros_(self.feature_transformer.bias)

        # Hidden layers
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer1.bias)

        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer2.bias)

        # Output layer
        nn.init.uniform_(self.output.weight, -0.1, 0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, white_features, black_features, stm):
        """
        Forward pass

        Args:
            white_features: [batch, max_features] - active feature indices for white
            black_features: [batch, max_features] - active feature indices for black
            stm: [batch] - side to move (1=white, 0=black)

        Returns:
            [batch] - evaluation scores
        """
        batch_size = white_features.shape[0]

        # Accumulate features (similar to NNUE accumulator)
        white_hidden = self._accumulate_features(white_features, batch_size)
        black_hidden = self._accumulate_features(black_features, batch_size)

        # Clip activations (simulating quantization)
        white_hidden = torch.clamp(torch.relu(white_hidden), 0, 127)
        black_hidden = torch.clamp(torch.relu(black_hidden), 0, 127)

        # Arrange based on side to move
        stm = stm.unsqueeze(1)  # [batch, 1]

        # STM perspective first, opponent second
        us = stm * white_hidden + (1 - stm) * black_hidden
        them = stm * black_hidden + (1 - stm) * white_hidden

        # Concatenate
        x = torch.cat([us, them], dim=1)

        # Forward through network
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)

        # Apply sigmoid to map to [0,1] for game outcome probabilities
        return torch.sigmoid(x.squeeze())

    def _accumulate_features(self, feature_indices, batch_size):
        """Accumulate features from sparse indices"""
        # Initialize with bias
        accumulated = self.feature_transformer.bias.unsqueeze(0).repeat(batch_size, 1)

        # Add weights for active features
        for batch_idx in range(batch_size):
            active_indices = feature_indices[batch_idx]
            valid_indices = active_indices[active_indices >= 0]

            if len(valid_indices) > 0:
                # Clamp indices to valid range
                valid_indices = torch.clamp(valid_indices, 0, self.input_size - 1)
                # Linear layer weight shape is [out_features, in_features], so we need to transpose
                accumulated[batch_idx] += self.feature_transformer.weight[:, valid_indices].sum(dim=1)

        return accumulated


class HalfKPFeatureExtractor:
    """Extract HalfKP features from chess positions"""

    def __init__(self):
        # Piece mapping for features
        self.piece_to_idx = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,  # White
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,  # Black
        }

    def fen_to_features(self, fen: str):
        """
        Convert FEN to HalfKP features

        Returns:
            white_features: list - active feature indices from white king perspective
            black_features: list - active feature indices from black king perspective
            stm: int - side to move (1=white, 0=black)
        """
        board = chess.Board(fen)

        # Find kings
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)

        if white_king is None or black_king is None:
            raise ValueError(f"Invalid position: missing king in {fen}")

        # Extract features for each king perspective
        white_features = self._extract_king_features(
            board, white_king, is_white_perspective=True
        )
        black_features = self._extract_king_features(
            board, black_king, is_white_perspective=False
        )

        # Side to move
        stm = 1 if board.turn == chess.WHITE else 0

        return white_features, black_features, stm

    def _extract_king_features(
        self, board: chess.Board, king_square: int, is_white_perspective: bool
    ):
        """Extract HalfKP features for one king perspective"""
        # Create sparse feature vector
        active_features = []

        # Apply horizontal flip for black king perspective (color symmetry)
        if not is_white_perspective:
            king_square = king_square ^ 7  # Flip horizontally

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:

                # Apply same flip to piece square if black perspective
                piece_square = square ^ 7 if not is_white_perspective else square

                # Get piece type (0-9): P,N,B,R,Q for white/black
                piece_type = self._get_halfkp_piece_type(piece)

                # HalfKP index: king_square * 640 + piece_square * 10 + piece_type
                feature_idx = king_square * 640 + piece_square * 10 + piece_type
                active_features.append(feature_idx)

        return active_features

    def _get_halfkp_piece_type(self, piece):
        """Get HalfKP piece type (0-9)"""
        piece_type = piece.piece_type - 1  # Convert 1-5 to 0-4
        color_offset = 0 if piece.color == chess.WHITE else 5
        return piece_type + color_offset

    def batch_fen_to_features(self, fens: list, max_features: int = 32):
        """Convert batch of FENs to padded feature tensors"""
        batch_size = len(fens)
        white_features = torch.full((batch_size, max_features), -1, dtype=torch.long)
        black_features = torch.full((batch_size, max_features), -1, dtype=torch.long)
        stm = np.zeros(batch_size, dtype=np.int64)

        for i, fen in enumerate(fens):
            try:
                wf, bf, s = self.fen_to_features(fen)

                # Pad/truncate to max_features
                wf_len = min(len(wf), max_features)
                bf_len = min(len(bf), max_features)

                if wf_len > 0:
                    white_features[i, :wf_len] = torch.tensor(wf[:wf_len], dtype=torch.long)
                if bf_len > 0:
                    black_features[i, :bf_len] = torch.tensor(bf[:bf_len], dtype=torch.long)

                stm[i] = s
            except Exception as e:
                print(f"Warning: Failed to process FEN {i}: {e}")
                # Use empty features for invalid positions
                pass

        return (
            white_features,
            black_features,
            torch.tensor(stm),
        )


def create_model(hidden_size: int = 256):
    """Create HalfKP NNUE model"""
    return HalfKPNNUE(hidden_size)


if __name__ == "__main__":
    # Test the model
    print("Testing HalfKP NNUE model...")

    model = create_model()
    extractor = HalfKPFeatureExtractor()

    # Test FENs
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # Castling rights
    ]

    white_features, black_features, stm = extractor.batch_fen_to_features(test_fens)

    print(f"White features shape: {white_features.shape}")
    print(f"Black features shape: {black_features.shape}")
    print(f"STM shape: {stm.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(white_features, black_features, stm.float())

    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
