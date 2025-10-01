#!/usr/bin/env python3
"""
Simple training script for HalfKP NNUE model
Clean and straightforward implementation
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from create_dataset import get_chunk_files, load_chunk
from halfkp_model import HalfKPFeatureExtractor, create_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import wandb


class ChessDataset(Dataset):
    """Dataset for chess positions from a single chunk file"""

    def __init__(self, chunk_file: str):
        self.extractor = HalfKPFeatureExtractor()

        print(f"Loading chunk: {chunk_file}")
        self.positions = load_chunk(chunk_file)
        print(f"Loaded {len(self.positions):,} positions")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, result = self.positions[idx]

        try:
            white_features, black_features, stm = self.extractor.fen_to_features(fen)

            # Convert to padded tensors (max 32 features)
            max_features = 32
            white_tensor = torch.full((max_features,), -1, dtype=torch.long)
            black_tensor = torch.full((max_features,), -1, dtype=torch.long)

            # Fill with actual features (truncate if necessary)
            wf_len = min(len(white_features), max_features)
            bf_len = min(len(black_features), max_features)

            if wf_len > 0:
                white_tensor[:wf_len] = torch.tensor(
                    white_features[:wf_len], dtype=torch.long
                )
            if bf_len > 0:
                black_tensor[:bf_len] = torch.tensor(
                    black_features[:bf_len], dtype=torch.long
                )

            return {
                "white_features": white_tensor,
                "black_features": black_tensor,
                "stm": torch.tensor(stm, dtype=torch.float32),
                "target": torch.tensor(result, dtype=torch.float32),
            }
        except Exception as e:
            print(f"Error processing position {idx}: {e}")
            # Return dummy data for invalid positions
            return {
                "white_features": torch.full((32,), -1, dtype=torch.long),
                "black_features": torch.full((32,), -1, dtype=torch.long),
                "stm": torch.tensor(0.0, dtype=torch.float32),
                "target": torch.tensor(0.5, dtype=torch.float32),
            }


def train_chunk(
    model, dataloader, optimizer, criterion, device, chunk_name, log_freq=100
):
    """Train on one chunk with detailed logging"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = len(dataloader)

    print(f"    Training on {chunk_name} ({num_batches:,} batches)")

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        white_features = batch["white_features"].to(device)
        black_features = batch["black_features"].to(device)
        stm = batch["stm"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        outputs = model(white_features, black_features, stm)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Calculate chess-specific accuracy
        # Model now outputs [0,1] range via sigmoid, so no need to clamp

        # Define outcome categories: loss (<0.33), draw (0.33-0.67), win (>0.67)
        pred_outcomes = torch.zeros_like(outputs)
        pred_outcomes[outputs < 0.33] = 0.0  # Loss
        pred_outcomes[(outputs >= 0.33) & (outputs <= 0.67)] = 0.5  # Draw
        pred_outcomes[outputs > 0.67] = 1.0  # Win

        # Targets should already be 0.0, 0.5, or 1.0
        correct = (torch.abs(pred_outcomes - targets) < 0.1).sum().item()

        total_correct += correct
        total_samples += targets.size(0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log progress during chunk training
        if (batch_idx + 1) % log_freq == 0 or batch_idx == num_batches - 1:
            current_loss = total_loss / (batch_idx + 1)
            current_acc = total_correct / total_samples if total_samples > 0 else 0.0
            progress = (batch_idx + 1) / num_batches * 100

            print(
                f"      Batch {batch_idx + 1:,}/{num_batches:,} ({progress:.1f}%) | "
                f"Loss: {current_loss:.6f} | Acc: {current_acc:.4f}"
            )

            # Log to wandb with chunk-specific metrics
            wandb.log(
                {
                    "chunk_batch_loss": loss.item(),
                    "chunk_running_loss": current_loss,
                    "chunk_running_accuracy": current_acc,
                    "batch_progress": progress,
                    "output_mean": outputs.mean().item(),
                    "output_std": outputs.std().item(),
                    "target_mean": targets.mean().item(),
                }
            )

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"      ✓ {chunk_name} complete: Loss {avg_loss:.6f} | Acc {accuracy:.4f}")
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            white_features = batch["white_features"].to(device)
            black_features = batch["black_features"].to(device)
            stm = batch["stm"].to(device)
            targets = batch["target"].to(device)

            outputs = model(white_features, black_features, stm)
            loss = criterion(outputs, targets)

            # Calculate chess-specific accuracy (same as training)
            pred_outcomes = torch.zeros_like(outputs)
            pred_outcomes[outputs < 0.33] = 0.0
            pred_outcomes[(outputs >= 0.33) & (outputs <= 0.67)] = 0.5
            pred_outcomes[outputs > 0.67] = 1.0

            correct = (torch.abs(pred_outcomes - targets) < 0.1).sum().item()

            total_correct += correct
            total_samples += targets.size(0)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load checkpoint and return start epoch, best accuracy, best loss"""
    try:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0.0)
        best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"✅ Resumed from epoch {start_epoch}")
        print(f"   Best accuracy so far: {best_accuracy:.4f}")
        print(f"   Best loss so far: {best_loss:.6f}")

        return start_epoch, best_accuracy, best_loss
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        return 0, 0.0, float("inf")


def main():
    # Training parameters - optimized for L40s GPU
    batch_size = 8192
    initial_lr = 0.002
    final_lr = 0.00001
    num_epochs = 25
    hidden_size = 256

    # Checkpoint settings
    resume_from_checkpoint = "latest_checkpoint.pt"  # Change to None to start fresh

    # Initialize wandb
    wandb.init(
        project="chess-nnue",
        config={
            "batch_size": batch_size,
            "initial_lr": initial_lr,
            "final_lr": final_lr,
            "num_epochs": num_epochs,
            "hidden_size": hidden_size,
            "architecture": "HalfKP-NNUE",
            "input_features": 40960,
            "training_approach": "chunk_by_chunk",
            "scheduler": "cosine_annealing",
            "resume_from": resume_from_checkpoint,
        },
    )

    # Device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})

    # Get chunk files
    chunk_files = get_chunk_files()
    if not chunk_files:
        print("No chunk files found! Run create_dataset.py first.")
        return

    print(f"Found {len(chunk_files)} chunk files")

    # Split into train/test (80/20)
    split_idx = int(0.8 * len(chunk_files))
    train_chunks = chunk_files[:split_idx]
    test_chunks = chunk_files[split_idx:]

    print(f"Training chunks: {len(train_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Create model
    model = create_model(hidden_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {total_params:,} parameters")

    # Log model architecture
    wandb.config.update(
        {"total_parameters": total_params, "num_train_chunks": len(train_chunks)}
    )
    wandb.watch(model, log="all", log_freq=500)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=final_lr)

    # Try to resume from checkpoint
    start_epoch = 0
    best_accuracy = 0.0
    best_loss = float("inf")

    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        start_epoch, best_accuracy, best_loss = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler
        )
    else:
        print("🆕 Starting fresh training")

    print(
        f"📈 Learning rate schedule: {initial_lr:.6f} → {final_lr:.6f} over {num_epochs} epochs"
    )

    # Training loop - chunk by chunk approach

    for epoch in range(start_epoch, num_epochs):
        print(f"\n🔥 EPOCH {epoch + 1}/{num_epochs}")
        epoch_start_time = time.time()

        epoch_train_losses = []
        epoch_train_accs = []

        # Train on each chunk sequentially
        for chunk_idx, train_chunk_file in enumerate(train_chunks):
            chunk_name = train_chunk_file.name
            print(f"  📁 Chunk {chunk_idx + 1}/{len(train_chunks)}: {chunk_name}")

            # Initialize to None to avoid unbound errors
            train_dataset = None
            train_loader = None

            # Load chunk dataset
            try:
                train_dataset = ChessDataset(str(train_chunk_file))
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                    if device.type == "cuda"
                    else False,  # MPS does not support pin_memory
                )

                # Train on this chunk
                chunk_loss, chunk_acc = train_chunk(
                    model, train_loader, optimizer, criterion, device, chunk_name
                )

                epoch_train_losses.append(chunk_loss)
                epoch_train_accs.append(chunk_acc)

                # Log chunk metrics
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "chunk_idx": chunk_idx + 1,
                        "chunk_train_loss": chunk_loss,
                        "chunk_train_accuracy": chunk_acc,
                    }
                )

            except Exception as e:
                print(f"    ❌ Error processing {chunk_name}: {e}")
                continue
            finally:
                # Clean up memory after each chunk
                if train_dataset is not None:
                    del train_dataset
                if train_loader is not None:
                    del train_loader
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"    🧹 Memory cleared after {chunk_name}")

        # Calculate epoch averages
        avg_train_loss = (
            sum(epoch_train_losses) / len(epoch_train_losses)
            if epoch_train_losses
            else float("inf")
        )
        avg_train_acc = (
            sum(epoch_train_accs) / len(epoch_train_accs) if epoch_train_accs else 0.0
        )

        # Evaluate on test set (using first test chunk for speed)
        if test_chunks:
            print(f"  🧪 Evaluating on test chunk: {test_chunks[0].name}")
            try:
                test_dataset = ChessDataset(str(test_chunks[0]))
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True if device.type in ["cuda", "mps"] else False,
                )
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                # Clean up test memory too
                del test_dataset
                del test_loader
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                print("    🧹 Test memory cleared")

            except Exception as e:
                print(f"    ❌ Error in evaluation: {e}")
                test_loss, test_acc = float("inf"), 0.0
        else:
            test_loss, test_acc = float("inf"), 0.0

        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch_train_loss": avg_train_loss,
                "epoch_train_accuracy": avg_train_acc,
                "epoch_test_loss": test_loss,
                "epoch_test_accuracy": test_acc,
                "epoch_time_minutes": epoch_time / 60,
                "learning_rate": current_lr,
            }
        )

        print(f"\n  📊 EPOCH {epoch + 1} SUMMARY:")
        print(f"    Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.4f}")
        print(f"    Test Loss:  {test_loss:.6f} | Test Acc:  {test_acc:.4f}")
        print(f"    Learning Rate: {current_lr:.8f}")
        print(f"    Time: {epoch_time / 60:.1f} minutes")

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_loss = test_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "best_accuracy": best_accuracy,
                    "best_loss": best_loss,
                },
                "best_model.pt",
            )
            wandb.save("best_model.pt")
            print(
                f"    ✓ 🏆 New best model! Acc: {test_acc:.4f} | Loss: {test_loss:.6f}"
            )

        # Step the scheduler
        scheduler.step()

        # Always save latest checkpoint every epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "best_accuracy": best_accuracy,
                "best_loss": best_loss,
            },
            "latest_checkpoint.pt",
        )
        print(f"    💾 Latest checkpoint saved: epoch {epoch + 1}")

    print("\n🎉 Training Complete!")
    print(f"   Best test accuracy: {best_accuracy:.4f}")
    print(f"   Best test loss: {best_loss:.6f}")
    print(f"   Final learning rate: {scheduler.get_last_lr()[0]:.8f}")

    # Log final metrics
    wandb.log(
        {
            "final_best_accuracy": best_accuracy,
            "final_best_loss": best_loss,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
