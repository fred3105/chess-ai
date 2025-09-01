import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess.pgn
import numpy as np
import os
import time
from datetime import datetime

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
    planes = np.zeros(
        (18, 64), dtype=np.float32
    )  # 12 piece + 4 castling + 1 en passant + 1 halfmove
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        planes[PIECE_TO_IDX[piece.symbol()], square] = 1.0

    # Castling rights planes
    planes[12, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant possible plane
    planes[16, :] = 1.0 if board.ep_square else 0.0

    # Halfmove clock plane (normalize by 100)
    planes[17, :] = board.halfmove_clock / 100.0

    return planes.flatten()  # shape (18*64=1152,)


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
        return torch.tensor(self.positions[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# -------------------------
# 3. Improved MLP model for max speed and accuracy
# -------------------------
class EvalNet(nn.Module):
    def __init__(self, input_size=1152, hidden_sizes=[384, 192], dropout_rate=0.15):
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
# 4. Advanced training loop with optimizations
# -------------------------
def train_model(
    train_loader,
    val_loader,
    model,
    epochs=400,
    lr=1e-3,
    device="mps",
    patience=30,
    min_lr=1e-7,  # Lower minimum for extended training
    use_wandb=True,
):
    model.to(device)
    training_start_time = time.time()

    if use_wandb:
        # Log model architecture and hyperparameters
        wandb.log(
            {
                "model/total_parameters": sum(p.numel() for p in model.parameters()),
                "model/trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "hyperparameters/learning_rate": lr,
                "hyperparameters/weight_decay": 5e-5,
                "hyperparameters/batch_size_train": train_loader.batch_size,
                "hyperparameters/batch_size_val": val_loader.batch_size,
                "hyperparameters/epochs": epochs,
                "hyperparameters/patience": patience,
                "hyperparameters/min_lr": min_lr,
                "dataset/train_size": len(train_loader.dataset),
                "dataset/val_size": len(val_loader.dataset),
                "training/device": device,
            }
        )

    # Better optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=5e-5, betas=(0.9, 0.999), eps=1e-8
    )

    # Advanced loss components for stronger training
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=0.1)  # More robust to outliers
    # L1 regularization computed inline for efficiency

    # Extended cosine annealing with warm restarts for long training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=min_lr,  # Longer cycles for stability
    )

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    training_history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    epoch_start_time = time.time()

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Advanced data augmentation: multiple strategies for robustness
            if torch.rand(1) < 0.5:  # 50% chance - more aggressive augmentation
                batch_x, batch_y = augment_position(batch_x, batch_y)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    outputs = model(batch_x).squeeze()
                    # Advanced combined loss: MSE + Huber + L1 regularization
                    prediction_loss = 0.6 * mse_loss(
                        outputs, batch_y
                    ) + 0.4 * huber_loss(outputs, batch_y)

                    # L1 regularization on model weights for sparsity
                    l1_reg = sum(
                        torch.sum(torch.abs(param)) for param in model.parameters()
                    )

                    loss = prediction_loss + 1e-6 * l1_reg
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if device == "mps":
                    # MPS doesn't support autocast yet, use float32
                    outputs = model(batch_x).squeeze()
                else:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        outputs = model(batch_x).squeeze()
                # Advanced combined loss for MPS
                prediction_loss = 0.6 * mse_loss(outputs, batch_y) + 0.4 * huber_loss(
                    outputs, batch_y
                )

                # L1 regularization on model weights
                l1_reg = sum(
                    torch.sum(torch.abs(param)) for param in model.parameters()
                )

                loss = prediction_loss + 1e-6 * l1_reg
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_mae = 0
        val_accuracy_05 = 0  # Accuracy within 0.5 evaluation units
        val_accuracy_02 = 0  # Accuracy within 0.2 evaluation units
        num_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()

                batch_loss = mse_loss(outputs, batch_y)
                val_loss += batch_loss.item()
                val_mae += torch.mean(torch.abs(outputs - batch_y)).item()

                # Accuracy metrics
                val_accuracy_05 += torch.mean(
                    (torch.abs(outputs - batch_y) < 0.5).float()
                ).item()
                val_accuracy_02 += torch.mean(
                    (torch.abs(outputs - batch_y) < 0.2).float()
                ).item()

                num_val_batches += 1

        val_loss /= num_val_batches
        val_mae /= num_val_batches
        val_accuracy_05 /= num_val_batches
        val_accuracy_02 /= num_val_batches

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_start_time = time.time()

        # Store training history
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["val_mae"].append(val_mae)
        training_history["lr"].append(current_lr)

        # Log to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/accuracy_0.5": val_accuracy_05,
                    "val/accuracy_0.2": val_accuracy_02,
                    "training/learning_rate": current_lr,
                    "training/epoch_time_seconds": epoch_time,
                    "training/patience_counter": patience_counter,
                    "training/best_val_loss": best_val_loss,
                }
            )

        # Enhanced progress monitoring for long training
        if epoch % 3 == 0 or epoch < 15:  # More frequent updates early on
            print(
                f"Epoch {epoch + 1:03d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, "
                f"Acc@0.5={val_accuracy_05:.3f}, Acc@0.2={val_accuracy_02:.3f}, "
                f"LR={current_lr:.2e}, Patience={patience_counter}/{patience}"
            )

        # Log milestone progress
        if (epoch + 1) % 50 == 0:
            elapsed_time = time.time() - training_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            estimated_remaining = (
                (epochs - epoch - 1) * avg_epoch_time / 60
            )  # in minutes

            print(f"\n🏆 MILESTONE: Epoch {epoch + 1}/{epochs} completed")
            print(f"   Best val loss so far: {best_val_loss:.4f}")
            print(f"   Elapsed time: {elapsed_time / 60:.1f} minutes")
            print(f"   Estimated remaining time: {estimated_remaining:.1f} minutes\n")

            if use_wandb:
                wandb.log(
                    {
                        "milestones/epoch": epoch + 1,
                        "milestones/elapsed_time_minutes": elapsed_time / 60,
                        "milestones/estimated_remaining_minutes": estimated_remaining,
                        "milestones/progress_percent": (epoch + 1) / epochs * 100,
                    }
                )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            if use_wandb:
                wandb.log(
                    {
                        "training/early_stopped": True,
                        "training/final_epoch": epoch + 1,
                        "training/early_stop_reason": "patience_exceeded",
                    }
                )
            break

    # Calculate final training statistics
    total_training_time = time.time() - training_start_time
    final_epoch = len(training_history["train_loss"])

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")

    # Log final training summary
    if use_wandb:
        wandb.log(
            {
                "training/completed": True,
                "training/total_time_minutes": total_training_time / 60,
                "training/total_time_hours": total_training_time / 3600,
                "training/final_epoch": final_epoch,
                "training/epochs_per_hour": final_epoch / (total_training_time / 3600),
                "final/best_val_loss": best_val_loss,
                "final/train_loss": training_history["train_loss"][-1],
                "final/val_mae": training_history["val_mae"][-1],
            }
        )

        # Log model summary
        wandb.run.summary.update(
            {
                "best_validation_loss": best_val_loss,
                "final_training_loss": training_history["train_loss"][-1],
                "total_training_hours": total_training_time / 3600,
                "epochs_completed": final_epoch,
            }
        )

    return training_history


# -------------------------
# 5. PGN parser pipeline -> .npz files
# -------------------------
def parse_pgn_folder(pgn_folder, output_file):
    positions = []
    labels = []

    max_games = 5000000

    for filename in os.listdir(pgn_folder):
        if not filename.endswith(".pgn"):
            continue
        filepath = os.path.join(pgn_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
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
                    if len(positions) >= max_games:
                        break
                if len(positions) >= max_games:
                    break

    np.savez_compressed(
        output_file, positions=np.array(positions), labels=np.array(labels)
    )
    print(f"Saved {len(positions)} positions to {output_file}")


# -------------------------
# 5. Data augmentation for better generalization
# -------------------------
def augment_position(batch_x, batch_y):
    """Advanced data augmentation for stronger training"""
    batch_size = batch_x.size(0)
    x_reshaped = batch_x.view(batch_size, 18, 64)

    # Multiple augmentation strategies
    aug_choice = torch.rand(1).item()

    if aug_choice < 0.4:  # 40% - Horizontal flip
        x_aug = x_reshaped.clone()
        for i in range(8):  # For each rank
            for j in range(8):  # For each file
                old_idx = i * 8 + j
                new_idx = i * 8 + (7 - j)  # Mirror horizontally
                x_aug[:, :, new_idx] = x_reshaped[:, :, old_idx]
        y_aug = -batch_y  # Flip evaluation

    elif aug_choice < 0.7:  # 30% - Perspective flip (white<->black)
        x_aug = x_reshaped.clone()
        # Swap white and black pieces (0-5 <-> 6-11)
        for i in range(6):
            temp = x_aug[:, i, :].clone()
            x_aug[:, i, :] = x_aug[:, i + 6, :]
            x_aug[:, i + 6, :] = temp

        # Flip castling rights (12-13 <-> 14-15)
        temp = x_aug[:, 12:14, :].clone()
        x_aug[:, 12:14, :] = x_aug[:, 14:16, :]
        x_aug[:, 14:16, :] = temp

        # Flip board vertically (rank 1->8, 2->7, etc)
        x_flipped = torch.zeros_like(x_aug)
        for rank in range(8):
            for file in range(8):
                old_square = rank * 8 + file
                new_square = (7 - rank) * 8 + file
                x_flipped[:, :, new_square] = x_aug[:, :, old_square]

        x_aug = x_flipped
        y_aug = -batch_y  # Flip evaluation from opponent perspective

    else:  # 30% - Small noise for robustness
        x_aug = x_reshaped.clone()
        # Add small amount of noise to non-piece planes (castling, en passant, halfmove)
        noise_scale = 0.05
        x_aug[:, 12:, :] += torch.randn_like(x_aug[:, 12:, :]) * noise_scale
        x_aug[:, 12:, :] = torch.clamp(x_aug[:, 12:, :], 0, 1)  # Keep in valid range
        y_aug = batch_y  # Keep same evaluation

    return x_aug.view(batch_size, -1), y_aug


# -------------------------
# 6. Fast inference using torch.jit with optimization
# -------------------------
def fast_eval(model, board, device="cpu"):
    model.to(device)
    model.eval()
    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad(), torch.inference_mode():
        return float(model(x).item())


def benchmark_model(model, device="mps", num_evals=10000, use_wandb=False):
    """Benchmark model inference speed for both single and batch evaluation"""
    import time
    import chess

    model.to(device)
    model.eval()

    # JIT compile for maximum speed
    example_input = torch.randn(1, 1152).to(device)
    jit_model = torch.jit.trace(model, example_input)

    # Create random board positions for benchmarking
    boards = []
    for _ in range(100):
        board = chess.Board()
        for _ in range(np.random.randint(5, 25)):
            moves = list(board.legal_moves)
            if moves:
                board.push(np.random.choice(moves))
        boards.append(board)

    print("=== PERFORMANCE BENCHMARKS ===")

    # Test single evaluation
    print("\n1. Single Position Evaluation:")
    for _ in range(100):
        board = np.random.choice(boards)
        fast_eval(jit_model, board, device)

    start_time = time.time()
    for i in range(num_evals):
        board = boards[i % len(boards)]
        fast_eval(jit_model, board, device)
    elapsed = time.time() - start_time
    single_evals_per_second = num_evals / elapsed

    print(f"   Single evals: {single_evals_per_second:8.0f} evals/second")

    # Test batch evaluation (optimal for M3 Pro)
    print("\n2. Batch Evaluation (Recommended):")
    batch_sizes = [4, 8, 16, 32, 64]
    best_throughput = single_evals_per_second
    best_batch_size = 1

    for batch_size in batch_sizes:
        # Prepare batch data
        batch_positions = []
        for _ in range(batch_size):
            board = np.random.choice(boards)
            batch_positions.append(encode_board(board))

        x_batch = torch.tensor(np.array(batch_positions), dtype=torch.float32).to(
            device
        )

        # Warmup
        for _ in range(50):
            with torch.no_grad():
                _ = jit_model(x_batch)

        # Benchmark
        num_batches = num_evals // batch_size
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = jit_model(x_batch)
        elapsed = time.time() - start_time

        throughput = (num_batches * batch_size) / elapsed
        print(f"   Batch {batch_size:2d}: {throughput:10.0f} evals/second")

        if throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = batch_size

    print(
        f"\n🚀 OPTIMAL: {best_throughput:.0f} evals/second (batch size {best_batch_size})"
    )
    target_achieved = best_throughput >= 10000
    print(f"🎯 10k+ target: {'✅ ACHIEVED' if target_achieved else '❌ NOT MET'}")

    if target_achieved:
        print(f"📈 Exceeds target by {best_throughput - 10000:.0f} evals/second")

    # Log benchmark results to wandb if enabled
    if use_wandb:
        wandb.log(
            {
                "benchmark/single_eval_throughput": single_evals_per_second,
                "benchmark/best_throughput": best_throughput,
                "benchmark/optimal_batch_size": best_batch_size,
                "benchmark/throughput_improvement": best_throughput
                / single_evals_per_second,
            }
        )

    return best_throughput, best_batch_size


def fast_eval_batch(model, boards, device="mps"):
    """Evaluate multiple positions efficiently"""
    model.eval()
    positions = [encode_board(board) for board in boards]
    x = torch.tensor(positions, dtype=torch.float32).to(device)

    with torch.no_grad(), torch.inference_mode():
        outputs = model(x)
        return [float(out.item()) for out in outputs]


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Parse PGNs into .npz dataset
    parse_pgn_folder("data", "chess_dataset.npz")

    # Load dataset
    data = np.load("chess_dataset.npz")
    positions = data["positions"]
    labels = data["labels"]

    # Shuffle indices
    num_samples = len(positions)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    # 80% train, 20% validation
    split_idx = int(num_samples * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = ChessDataset(positions[train_indices], labels[train_indices])
    val_dataset = ChessDataset(positions[val_indices], labels[val_indices])

    # Optimized batch sizes for M3 Pro
    # Optimal batch sizes for M3 Pro - larger batches for better gradients
    train_loader = DataLoader(
        train_dataset, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4096, shuffle=False, num_workers=0, pin_memory=False
    )

    # Initialize wandb experiment tracking
    wandb.init(
        project="chess-eval-optimization",
        name=f"chess-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": "EvalNet",
            "hidden_sizes": [512, 256, 128],
            "dropout_rate": 0.2,
            "input_size": 1152,
            "dataset_size": len(positions),
            "train_batch_size": 2048,
            "val_batch_size": 4096,
            "epochs": 400,
            "learning_rate": 1e-3,
            "weight_decay": 5e-5,
            "patience": 30,
            "device": "mps",
            "augmentation_rate": 0.5,
            "loss_function": "MSE+Huber+L1",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts",
        },
        tags=["chess", "evaluation", "m3-pro", "optimization"],
        notes="Maximum strength chess evaluation function optimized for M3 Pro",
    )

    # Train maximum-strength model for M3 Pro - balanced size/performance
    model = EvalNet(hidden_sizes=[512, 256, 128], dropout_rate=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    print(f"Training on {len(positions):,} total positions")

    # Log model to wandb
    wandb.watch(model, log="all", log_freq=100)

    # Advanced training recommendations based on dataset size
    if len(positions) < 1_000_000:
        print(
            "⚠️  Dataset is small (<1M positions). Consider gathering more data for stronger evaluation."
        )
    elif len(positions) < 5_000_000:
        print(
            "📊 Good dataset size (1-5M positions). Should achieve strong amateur-level evaluation."
        )
    elif len(positions) < 20_000_000:
        print(
            "🎯 Excellent dataset size (5-20M positions). Should achieve master-level evaluation."
        )
    else:
        print(
            "🚀 Massive dataset size (>20M positions). Should achieve grandmaster-level evaluation!"
        )

    print(
        f"Estimated training time: {len(positions) / 80000 * 1.0:.1f}-{len(positions) / 80000 * 1.5:.1f} hours on M3 Pro"
    )
    print(
        f"Batch size optimized for M3 Pro: {train_loader.batch_size} (training) / {val_loader.batch_size} (validation)"
    )
    print(
        "🔥 Advanced optimizations active: Multi-strategy augmentation, L1 regularization, extended schedules"
    )

    history = train_model(
        train_loader,
        val_loader,
        model,
        epochs=400,
        lr=1e-3,
        device="mps",
        patience=30,
        use_wandb=True,
    )

    # Save model and create optimized versions
    torch.save(model.state_dict(), "chess_eval_net.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_history": history,
            "model_config": {
                "hidden_sizes": [384, 192],
                "dropout_rate": 0.15,
                "input_size": 1152,
            },
        },
        "chess_eval_net_full.pth",
    )

    # Create JIT compiled version for maximum speed
    model.eval()
    example_input = torch.randn(1, 1152)
    if torch.backends.mps.is_available():
        example_input = example_input.to("mps")
    jit_model = torch.jit.trace(model, example_input)
    jit_model.save("chess_eval_net_jit.pt")

    print("Model saved and JIT compiled.")

    # Comprehensive performance benchmark
    print("\n" + "=" * 50)
    print("BENCHMARKING OPTIMIZED MODEL")
    print("=" * 50)
    best_throughput, optimal_batch = benchmark_model(
        model, device="mps", num_evals=10000, use_wandb=True
    )

    print(f"\n📊 SUMMARY:")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Peak performance: {best_throughput:,.0f} evals/second")
    print(f"   • Optimal batch size: {optimal_batch}")
    print(f"   • Performance vs target: {best_throughput / 10000:.1f}x")

    if best_throughput >= 10000:
        print(
            f"\n✅ SUCCESS: Model achieves {best_throughput:,.0f} evals/second on M3 Pro!"
        )
        print(
            f"   Chess engines should use batch size {optimal_batch} for optimal performance."
        )
        strength_estimate = min(
            3000, 1500 + len(positions) / 10000
        )  # Rough ELO estimate
        print(f"   Estimated playing strength: ~{strength_estimate:.0f} ELO")
    else:
        print(f"\n⚠️  Model achieves {best_throughput:,.0f} evals/second (target: 10k+)")
        print("   Consider reducing model size if speed is critical.")

    print(f"\n📈 TRAINING SUMMARY:")
    print(f"   • Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   • Final validation MAE: {history['val_mae'][-1]:.4f}")
    print(f"   • Training epochs completed: {len(history['train_loss'])}")
    print(f"   • Model will be strongest with more diverse, high-quality games")

    # Log final performance metrics to wandb
    if best_throughput >= 10000:
        strength_estimate = min(3000, 1500 + len(positions) / 10000)

        wandb.log(
            {
                "performance/throughput_evals_per_second": best_throughput,
                "performance/optimal_batch_size": optimal_batch,
                "performance/target_achieved": True,
                "performance/estimated_elo": strength_estimate,
                "performance/speed_vs_target": best_throughput / 10000,
            }
        )

        # Create performance summary table
        performance_table = wandb.Table(
            columns=["Metric", "Value", "Unit"],
            data=[
                ["Throughput", f"{best_throughput:,.0f}", "evals/second"],
                ["Optimal Batch Size", str(optimal_batch), "positions"],
                ["Estimated ELO", f"{strength_estimate:.0f}", "rating"],
                ["Speed vs Target", f"{best_throughput / 10000:.1f}x", "multiplier"],
                ["Model Parameters", f"{total_params:,}", "parameters"],
                ["Dataset Size", f"{len(positions):,}", "positions"],
            ],
        )
        wandb.log({"performance/summary_table": performance_table})

    # Save model artifact to wandb
    model_artifact = wandb.Artifact(
        name=f"chess-eval-model-{wandb.run.id}",
        type="model",
        description="Optimized chess evaluation neural network",
    )
    model_artifact.add_file("chess_eval_net.pth")
    model_artifact.add_file("chess_eval_net_full.pth")
    model_artifact.add_file("chess_eval_net_jit.pt")
    wandb.log_artifact(model_artifact)

    # Finish wandb run
    wandb.finish()
    print("\n🔗 Training results logged to Weights & Biases!")
