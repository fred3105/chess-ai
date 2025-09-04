import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedReLU(nn.Module):
    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), 0, self.max_value)


class NNUE(nn.Module):
    def __init__(
        self,
        feature_dim: int = 40960,  # HalfKP feature dimension (64*64*10)
        hidden_dim: int = 256,
        l1_hidden_dim: int = 32,
        l2_hidden_dim: int = 32,
        output_scale: float = 600.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_scale = output_scale

        # Feature transformer - maps sparse HalfKP features to dense representation
        self.feature_transformer = nn.Linear(feature_dim, hidden_dim, bias=True)

        # L1 layer - processes concatenated king perspectives
        self.l1 = nn.Linear(hidden_dim * 2, l1_hidden_dim, bias=True)

        # L2 layer
        self.l2 = nn.Linear(l1_hidden_dim, l2_hidden_dim, bias=True)

        # Output layer
        self.output = nn.Linear(l2_hidden_dim, 1, bias=True)

        # Activations
        self.clipped_relu = ClippedReLU(1.0)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize feature transformer with small weights
        nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.feature_transformer.bias)

        # Initialize other layers with Xavier normal
        for layer in [self.l1, self.l2, self.output]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, white_features: torch.Tensor, black_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for NNUE evaluation

        Args:
            white_features: HalfKP features from white king perspective [batch_size, feature_dim]
            black_features: HalfKP features from black king perspective [batch_size, feature_dim]

        Returns:
            Evaluation score in centipawns [batch_size, 1]
        """
        # Transform features to hidden representation
        white_hidden = self.clipped_relu(self.feature_transformer(white_features))
        black_hidden = self.clipped_relu(self.feature_transformer(black_features))

        # Concatenate king perspectives
        combined = torch.cat([white_hidden, black_hidden], dim=1)

        # L1 layer
        l1_out = self.clipped_relu(self.l1(combined))
        l1_out = self.dropout(l1_out)

        # L2 layer
        l2_out = self.clipped_relu(self.l2(l1_out))
        l2_out = self.dropout(l2_out)

        # Output layer
        output = self.output(l2_out)

        # Scale to centipawn range
        return output * self.output_scale

    def get_feature_weights(self) -> torch.Tensor:
        """Get the feature transformer weights for analysis"""
        return self.feature_transformer.weight.data

    def set_feature_weights(self, weights: torch.Tensor):
        """Set the feature transformer weights"""
        self.feature_transformer.weight.data = weights


class HalfKPFeatureExtractor:
    """Extract HalfKP features from chess positions"""

    def __init__(self):
        # Piece type mapping: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
        self.piece_types = 6
        self.squares = 64
        # HalfKP: For each king square (64), we have piece_square (64) * piece_type (5, no king) * color (2)
        # So: 64 * 64 * 5 * 2 = 40960
        self.feature_dim = 64 * 64 * 5 * 2

    def square_to_index(self, square: int) -> int:
        """Convert square (0-63) to feature index"""
        return square

    def piece_to_type(self, piece: int) -> int:
        """Convert piece value to piece type (0-5)"""
        return abs(piece) - 1 if piece != 0 else -1

    def extract_halfkp_features(
        self, board_state: np.ndarray, white_king_sq: int, black_king_sq: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract HalfKP features for both king perspectives

        Args:
            board_state: 8x8 numpy array with piece values
            white_king_sq: White king square (0-63)
            black_king_sq: Black king square (0-63)

        Returns:
            Tuple of (white_features, black_features) as sparse tensors
        """
        white_features = torch.zeros(self.feature_dim, dtype=torch.float32)
        black_features = torch.zeros(self.feature_dim, dtype=torch.float32)

        for square in range(64):
            row, col = square // 8, square % 8
            piece = int(board_state[row, col])  # Convert to regular Python int

            if piece == 0:  # Empty square
                continue

            piece_type = self.piece_to_type(piece)
            piece_color = 1 if piece > 0 else 0  # 1 for white, 0 for black

            # Skip kings for HalfKP features (they're implicit in the king square)
            if piece_type == 5:  # King
                continue

            # Calculate feature indices for both king perspectives
            # HalfKP encoding: king_sq * (64 * 5 * 2) + piece_sq * (5 * 2) + piece_type * 2 + piece_color
            base_idx = int(piece_type * 2 + piece_color)
            white_idx = int(white_king_sq) * 640 + square * 10 + base_idx
            black_idx = int(black_king_sq) * 640 + square * 10 + base_idx

            # Ensure indices are within bounds
            if white_idx < self.feature_dim:
                white_features[white_idx] = 1.0
            if black_idx < self.feature_dim:
                black_features[black_idx] = 1.0

        return white_features, black_features


def create_nnue_model(
    hidden_dim: int = 256,
    l1_hidden_dim: int = 32,
    l2_hidden_dim: int = 32,
    dropout_rate: float = 0.1,
) -> NNUE:
    """Create a standard NNUE model with recommended parameters"""
    return NNUE(
        hidden_dim=hidden_dim,
        l1_hidden_dim=l1_hidden_dim,
        l2_hidden_dim=l2_hidden_dim,
        dropout_rate=dropout_rate,
    )


if __name__ == "__main__":
    # Test the model
    model = create_nnue_model()

    # Create dummy input
    batch_size = 32
    feature_dim = 40960

    white_features = torch.randn(batch_size, feature_dim) > 0.95  # Sparse features
    black_features = torch.randn(batch_size, feature_dim) > 0.95

    white_features = white_features.float()
    black_features = black_features.float()

    # Forward pass
    output = model(white_features, black_features)
    print(f"Model output shape: {output.shape}")
    print(f"Sample outputs: {output[:5].flatten()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
