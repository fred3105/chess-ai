import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess.pgn
import numpy as np
import os
import gc
from glob import glob

# Wandb for experiment tracking - install with: pip install wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    print(
        "Warning: wandb not installed. Install with 'pip install wandb' for experiment tracking."
    )
    WANDB_AVAILABLE = False
    wandb = None

# -------------------------
# 1. Encode chess board with extra channels
# -------------------------
PIECE_TO_IDX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def encode_board(board):
    # 12 piece channels only for (8, 8, 12) input format
    planes = np.zeros((12, 64), dtype=np.float32)  # 12 piece channels only
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        planes[PIECE_TO_IDX[piece.symbol()], square] = 1.0

    return planes.reshape(8, 8, 12)


# -------------------------
# 2. Dataset
# -------------------------
class ChessDataset(Dataset):
    def __init__(self, positions, labels):
        self.positions = positions
        self.labels = labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Convert position to proper shape and normalize labels to 0-1 range
        position = torch.tensor(self.positions[idx], dtype=torch.float32)
        # Normalize label from [-1, 1] to [0, 1]
        label = torch.tensor((self.labels[idx] + 1) / 2, dtype=torch.float32)
        return position, label


# -------------------------
# 3. Modern Chess Neural Network Architectures
# -------------------------


class ChessEvalCNN(nn.Module):
    """CNN architecture matching the specified CONV model with ~3.2M parameters"""

    def __init__(self, dropout_rate=0.0):
        super().__init__()

        # Input: 12 channels (pieces only) x 8x8 board
        # Architecture: 12 -> 768 -> 384 -> 192 channels

        # First conv: (8,8,12) -> (4,4,768) with stride 2
        self.conv1 = nn.Conv2d(12, 768, kernel_size=3, stride=2, padding=1)

        # Second conv: (4,4,768) -> (2,2,384) with stride 2
        self.conv2 = nn.Conv2d(768, 384, kernel_size=3, stride=2, padding=1)

        # Third conv: (2,2,384) -> (1,1,192) with stride 2
        self.conv3 = nn.Conv2d(384, 192, kernel_size=3, stride=2, padding=1)

        # Flatten: (1,1,192) -> (192,)
        self.flatten = nn.Flatten()

        # Dense layers: 192 -> 96 -> 1
        self.fc1 = nn.Linear(192, 96)
        self.fc_out = nn.Linear(96, 1)

        # ReLU activation
        self.activation = nn.ReLU(inplace=True)

        self._init_weights()

    def forward(self, x):
        # Input shape: (batch_size, 8, 8, 12)
        # Convert to PyTorch format: (batch_size, 12, 8, 8)
        if len(x.shape) == 4 and x.shape[-1] == 12:
            x = x.permute(
                0, 3, 1, 2
            )  # (batch, height, width, channels) -> (batch, channels, height, width)

        # First conv: (batch, 12, 8, 8) -> (batch, 768, 4, 4)
        x = self.activation(self.conv1(x))

        # Second conv: (batch, 768, 4, 4) -> (batch, 384, 2, 2)
        x = self.activation(self.conv2(x))

        # Third conv: (batch, 384, 2, 2) -> (batch, 192, 1, 1)
        x = self.activation(self.conv3(x))

        # Flatten: (batch, 192, 1, 1) -> (batch, 192)
        x = self.flatten(x)

        # Dense layer: (batch, 192) -> (batch, 96)
        x = self.activation(self.fc1(x))

        # Output layer: (batch, 96) -> (batch, 1)
        # Use sigmoid for 0-1 normalized output
        output = torch.sigmoid(self.fc_out(x))
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Xavier initialization for ELU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ChessNet(nn.Module):
    def __init__(self, device="cpu"):
        super(ChessNet, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(12, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        # Use GroupNorm instead of BatchNorm for MPS compatibility
        self.gn1 = nn.GroupNorm(16, 128)  # 16 groups for 128 channels
        self.gn2 = nn.GroupNorm(16, 128)
        self.gn3 = nn.GroupNorm(16, 128)

        self.policy_conv = nn.Conv2d(128, 73, 1)
        self.policy_gn = nn.GroupNorm(1, 73)  # 1 group for 73 channels
        self.policy_fc = nn.Linear(73 * 64, 4672)

        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_gn = nn.GroupNorm(4, 32)  # 4 groups for 32 channels
        self.value_fc1 = nn.Linear(32 * 64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Input shape: (batch_size, 8, 8, 12)
        # Convert to PyTorch format: (batch_size, 12, 8, 8)
        if len(x.shape) == 4 and x.shape[-1] == 12:
            x = x.permute(0, 3, 1, 2)

        # Shared layers
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))

        policy = F.relu(self.policy_gn(self.policy_conv(x)))
        policy = torch.flatten(policy, start_dim=1)  # More MPS-friendly than view
        policy = self.policy_fc(policy)
        policy = F.log_softmax(
            policy, dim=1
        )  # More numerically stable than softmax + log

        value = F.relu(self.value_gn(self.value_conv(x)))
        value = torch.flatten(value, start_dim=1)  # More MPS-friendly than view
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


# Legacy MLP for compatibility/speed comparison
class EvalNet(nn.Module):
    def __init__(self, input_size=768, hidden_sizes=[384, 192], dropout_rate=0.15):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Maximum speed architecture - minimal layers, optimal for M3 Pro
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc_out = nn.Linear(hidden_sizes[1], 1)

        # Direct skip connection from input to final layer for representation power
        self.skip = nn.Linear(input_size, 1)

        # Use ReLU for maximum speed (fastest activation on M3 Pro)
        self.activation = nn.ReLU(inplace=True)

        # Minimal dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self._init_weights()

    def forward(self, x):
        # Minimal forward pass for maximum speed
        h1 = self.activation(self.fc1(x))
        if self.training:
            h1 = self.dropout(h1)

        h2 = self.activation(self.fc2(h1))
        if self.training:
            h2 = self.dropout(h2)

        # Main path
        main_out = self.fc_out(h2)

        # Skip connection for better learning
        skip_out = self.skip(x)

        # Combine and activate
        output = torch.tanh(main_out + skip_out)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# -------------------------
# 5. PGN parser pipeline -> .npz files
# -------------------------
def parse_pgn_file(pgn_file, max_positions=None):
    """Parse a single PGN file and return positions, labels, and dummy policies"""
    positions = []
    labels = []
    policies = []
    games_processed = 0

    with open(pgn_file, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_processed += 1

            # Print progress every 1000 games
            if games_processed % 1000 == 0:
                print(
                    f"  Processed {games_processed} games, {len(positions)} positions..."
                )

            board = game.board()
            result = game.headers.get("Result", "1/2-1/2")
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0

            for move in game.mainline_moves():
                board.push(move)
                positions.append(encode_board(board))
                labels.append(outcome)
                # Create dummy policy (uniform distribution over 4672 possible moves)
                dummy_policy = np.ones(4672) / 4672
                policies.append(dummy_policy)

                if max_positions and len(positions) >= max_positions:
                    break

            if max_positions and len(positions) >= max_positions:
                print(f"  Reached max positions limit ({max_positions}), stopping...")
                break

    print(f"  Finished processing {games_processed} games")
    return np.array(positions), np.array(labels), np.array(policies)


def get_pgn_files_from_directory(data_dir):
    """Get list of all .pgn files from the /data directory"""
    pgn_files = glob(os.path.join(data_dir, "*.pgn"))
    return pgn_files


# -------------------------
# AlphaZero Loss and Dataset Classes
# -------------------------


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        # Use torch.clamp to prevent log(0) and improve numerical stability
        y_policy_clamped = torch.clamp(y_policy.float(), min=1e-8)
        policy_error = torch.sum((-policy * torch.log(y_policy_clamped)), 1)
        total_error = (torch.flatten(value_error).float() + policy_error).mean()
        return total_error


class board_data(Dataset):
    def __init__(self, dataset):
        self.X, self.Y, self.policy = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        state = torch.tensor(self.X[idx], dtype=torch.float32)
        policy = torch.tensor(self.policy[idx], dtype=torch.float32)
        value = torch.tensor(self.Y[idx], dtype=torch.float32)
        return state, policy, value


# -------------------------
# 5. Data augmentation for better generalization
# -------------------------
def augment_position(batch_x, batch_y):
    """Advanced data augmentation for stronger training"""
    # Multiple augmentation strategies
    aug_choice = torch.rand(1).item()

    if aug_choice < 0.4:  # 40% - Horizontal flip
        x_aug = batch_x.clone()
        x_aug = torch.flip(x_aug, dims=[2])  # Flip along width dimension
        y_aug = 1 - batch_y  # Flip evaluation for 0-1 range

    elif aug_choice < 0.7:  # 30% - Perspective flip (white<->black)
        x_aug = batch_x.clone()
        # Swap white and black pieces (channels 0-5 <-> 6-11)
        temp = x_aug[:, :, :, 0:6].clone()
        x_aug[:, :, :, 0:6] = x_aug[:, :, :, 6:12]
        x_aug[:, :, :, 6:12] = temp

        # Flip board vertically (rank 1->8, 2->7, etc)
        x_aug = torch.flip(x_aug, dims=[1])  # Flip along height dimension
        y_aug = 1 - batch_y  # Flip evaluation from opponent perspective

    else:  # 30% - Keep original
        x_aug = batch_x.clone()
        y_aug = batch_y  # Keep same evaluation

    return x_aug, y_aug


# -------------------------
# 6. Evaluation test positions for strength assessment
# -------------------------
EVALUATION_POSITIONS = [
    # Tactical positions with best moves
    {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "best_move": "d2d3",  # Italian Game - solid development
        "description": "Italian Game opening position",
    },
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "best_move": "e7e5",  # King's Pawn response
        "description": "Opening: response to e4",
    },
    {
        "fen": "r2qkb1r/ppp2ppp/2n1bn2/2bpp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 5",
        "best_move": "d4d5",  # Take bishop
        "description": "Four Knights Game development",
    },
    {
        "fen": "2rqkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 6",
        "best_move": "e4d5",  # Take center
        "description": "Central control position",
    },
    {
        "fen": "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "best_move": "f3e5",  # Solid center support
        "description": "Italian vs Italian",
    },
    {
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "best_move": "g8f6",  # Counter-attack
        "description": "Italian Game defense",
    },
    {
        "fen": "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6",
        "best_move": "d5c4",  # Take on c4
        "description": "Middlegame tactical motif",
    },
    {
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5",
        "best_move": "d7d5",  # Solid defense
        "description": "Italian Game solid defense",
    },
]


def evaluate_model_strength(model, device="mps", verbose=False):
    """
    Test the model on specific chess positions to evaluate its tactical understanding.
    Returns a score based on how well it evaluates the positions.
    """
    model.eval()
    correct_evaluations = 0
    total_positions = len(EVALUATION_POSITIONS)
    evaluation_scores = []

    with torch.no_grad():
        for i, pos_data in enumerate(EVALUATION_POSITIONS):
            try:
                # Create board from FEN
                board = chess.Board(pos_data["fen"])

                # Test different candidate moves and see if model prefers the best one
                move_evaluations = []
                legal_moves = list(board.legal_moves)

                # Limit to top 5 most reasonable moves to speed up evaluation
                moves_to_test = legal_moves[: min(5, len(legal_moves))]

                for move in moves_to_test:
                    board_copy = board.copy()
                    board_copy.push(move)
                    move_eval = fast_eval(model, board_copy, device)

                    # For white to move, we want higher evaluations for better moves
                    # For black to move, we want lower evaluations for better moves
                    if board.turn:  # White to move
                        move_evaluations.append((move, move_eval))
                    else:  # Black to move
                        move_evaluations.append((move, -move_eval))

                # Sort by evaluation (best moves first)
                move_evaluations.sort(key=lambda x: x[1], reverse=True)

                # Check if the best move according to our test set is in top choices
                best_move_uci = pos_data["best_move"]
                try:
                    best_move = chess.Move.from_uci(best_move_uci)
                    if best_move in [m[0] for m in move_evaluations[:2]]:  # Top 2 moves
                        correct_evaluations += 1
                        score = 1.0
                    elif best_move in [
                        m[0] for m in move_evaluations[:3]
                    ]:  # Top 3 moves
                        score = 0.5
                    else:
                        score = 0.0

                    evaluation_scores.append(score)

                    if verbose:
                        print(f"Position {i + 1}: {pos_data['description']}")
                        print(f"Best move: {best_move_uci}, Score: {score}")
                        print(
                            f"Top moves by model: {[str(m[0]) for m in move_evaluations[:3]]}"
                        )
                        print()

                except ValueError:
                    # Invalid move in test data, skip
                    evaluation_scores.append(0.0)

            except Exception as e:
                if verbose:
                    print(f"Error evaluating position {i + 1}: {e}")
                evaluation_scores.append(0.0)

    # Calculate metrics
    avg_score = (
        sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0.0
    )
    accuracy = correct_evaluations / total_positions

    return {
        "accuracy": accuracy,
        "average_score": avg_score,
        "correct_positions": correct_evaluations,
        "total_positions": total_positions,
        "individual_scores": evaluation_scores,
    }


# -------------------------
# 7. Fast inference using torch.jit with optimization
# -------------------------
def fast_eval(model, board, device="cpu"):
    model.to(device)
    model.eval()
    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad(), torch.inference_mode():
        model_output = model(x)
        # Handle different model outputs
        if isinstance(model_output, tuple):  # ChessNet returns (policy, value)
            policy, value = model_output
            output = value.item()
        else:  # Other models return single output
            output = model_output.item()
        # Convert from 0-1 output back to -1 to 1 range
        return float(2 * output - 1)


def train(
    net,
    dataset,
    epoch_start=0,
    epoch_stop=20,
    cpu=0,
    dataset_name="",
    global_epoch_offset=0,
):
    torch.manual_seed(cpu)
    # Try MPS with improved tensor operations
    mps = torch.backends.mps.is_available()
    device = "mps" if mps else "cpu"
    print(f"Using device: {device}")
    net.to(device)
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=0.001
    )  # Reduced learning rate for stability
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300, 400], gamma=0.2
    )

    train_set = board_data(dataset)
    train_loader = DataLoader(
        train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False
    )
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if mps:
                state, policy, value = (
                    state.to(device).float(),
                    policy.float().to(device),
                    value.to(device).float(),
                )
            optimizer.zero_grad()
            policy_pred, value_pred = net(
                state
            )  # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            # Check for NaN loss and skip if found
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {i}, skipping...")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches of size = batch_size
                avg_loss = total_loss / 10
                print(
                    "Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f"
                    % (
                        os.getpid(),
                        epoch + 1,
                        (i + 1) * 30,
                        len(train_set),
                        avg_loss,
                    )
                )

                # Log to Wandb if available
                if WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "batch_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "epoch": global_epoch_offset + epoch + 1,
                            "dataset": dataset_name,
                            "local_epoch": epoch + 1,
                            "batch": i + 1,
                        }
                    )

                losses_per_batch.append(avg_loss)
                total_loss = 0.0
        epoch_loss = (
            sum(losses_per_batch) / len(losses_per_batch) if losses_per_batch else 0.0
        )
        losses_per_epoch.append(epoch_loss)

        # Log epoch metrics to Wandb
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                    "epoch": global_epoch_offset + epoch + 1,
                    "dataset": dataset_name,
                    "local_epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        scheduler.step()  # Move scheduler step to after epoch completes
        if len(losses_per_epoch) > 100:
            if (
                abs(
                    sum(losses_per_epoch[-4:-1]) / 3
                    - sum(losses_per_epoch[-16:-13]) / 3
                )
                <= 0.01
            ):
                break


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Get list of all .pgn files from /data directory
    pgn_files = get_pgn_files_from_directory("data")

    if not pgn_files:
        print("No .pgn files found in /data directory!")
        exit(1)

    print(f"Found {len(pgn_files)} .pgn files to process")

    # Initialize ChessNet model for AlphaZero-style training
    net = ChessNet(device="mps" if torch.backends.mps.is_available() else "cpu")
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Model has {total_params:,} parameters")

    # Initialize Wandb once for the entire training session
    if WANDB_AVAILABLE:
        wandb.init(
            project="chess-eval-optimization",
            config={
                "architecture": "ChessNet",
                "learning_rate": 0.001,
                "batch_size": 30,
                "epochs_per_dataset": 20,
                "total_datasets": len(pgn_files),
                "device": "mps" if torch.backends.mps.is_available() else "cpu",
                "optimizer": "Adam",
                "scheduler": "MultiStepLR",
                "loss_function": "AlphaLoss",
            },
        )
        wandb.watch(net, log="all", log_freq=100)

    global_epoch_count = 0

    # Train on each PGN file successively
    for i, pgn_file in enumerate(pgn_files):
        print(
            f"\n=== Training on file {i + 1}/{len(pgn_files)}: {os.path.basename(pgn_file)} ==="
        )

        # Load this specific PGN file with a reasonable limit
        print(f"Loading {pgn_file}...")
        positions, labels, policies = parse_pgn_file(pgn_file, max_positions=200000)
        dataset = (positions, labels, policies)
        print(f"Loaded {len(positions)} positions")

        # Train for 20 epochs on this dataset
        train(
            net,
            dataset,
            epoch_start=0,
            epoch_stop=20,
            cpu=0,
            dataset_name=os.path.basename(pgn_file),
            global_epoch_offset=global_epoch_count,
        )
        global_epoch_count += 20

        # Log model evaluation metrics to Wandb if available
        if WANDB_AVAILABLE:
            try:
                eval_results = evaluate_model_strength(
                    net, device="mps" if torch.backends.mps.is_available() else "cpu"
                )
                wandb.log(
                    {
                        "model_accuracy": eval_results["accuracy"],
                        "model_avg_score": eval_results["average_score"],
                        "correct_positions": eval_results["correct_positions"],
                        "total_positions": eval_results["total_positions"],
                        "dataset": os.path.basename(pgn_file),
                        "global_epoch": global_epoch_count,
                    }
                )
                print(
                    f"Model evaluation - Accuracy: {eval_results['accuracy']:.2f}, Avg Score: {eval_results['average_score']:.2f}"
                )
            except Exception as e:
                print(f"Warning: Model evaluation failed: {e}")

        # Save model after each dataset
        torch.save(net.state_dict(), f"chess_net_dataset_{i + 1}.pth")
        print(f"Model saved as chess_net_dataset_{i + 1}.pth")

        # Free memory by deleting the dataset and running garbage collection
        del positions, labels, policies, dataset
        gc.collect()

        print(f"Completed training on {os.path.basename(pgn_file)}")

    # Save final model
    torch.save(net.state_dict(), "chess_net_final.pth")
    print("\n=== Training completed on all datasets ===")
    print("Final model saved as chess_net_final.pth")

    # Final model evaluation and Wandb cleanup
    if WANDB_AVAILABLE:
        try:
            final_eval_results = evaluate_model_strength(
                net,
                device="mps" if torch.backends.mps.is_available() else "cpu",
                verbose=True,
            )
            wandb.log(
                {
                    "final_model_accuracy": final_eval_results["accuracy"],
                    "final_model_avg_score": final_eval_results["average_score"],
                    "final_global_epoch": global_epoch_count,
                }
            )
            print(f"\nFinal Model Performance:")
            print(f"Accuracy: {final_eval_results['accuracy']:.2f}")
            print(f"Average Score: {final_eval_results['average_score']:.2f}")
            wandb.finish()
        except Exception as e:
            print(f"Warning: Final evaluation failed: {e}")
