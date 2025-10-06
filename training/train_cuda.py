"""
Script made to run on L40S GPU with 48GB memory
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from create_dataset import get_chunk_files, load_chunk
from halfkp_model import HalfKPFeatureExtractor, create_model
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb


def preload_all_data_to_gpu(chunk_files, device, max_chunks=None):
    """
    Load all chunk files and preprocess them directly into GPU tensors
    Returns tensors already on GPU for zero-copy training
    """
    extractor = HalfKPFeatureExtractor()

    all_white_features = []
    all_black_features = []
    all_stm = []
    all_targets = []

    chunks_to_load = chunk_files[:max_chunks] if max_chunks else chunk_files

    print(f"\n📥 Preloading {len(chunks_to_load)} chunks into GPU memory...")
    start_time = time.time()

    for chunk_idx, chunk_file in enumerate(chunks_to_load):
        print(
            f"  Loading chunk {chunk_idx + 1}/{len(chunks_to_load)}: {chunk_file.name}"
        )

        positions = load_chunk(str(chunk_file))
        chunk_size = len(positions)

        # Preallocate tensors for this chunk
        max_features = 32
        white_tensor = torch.full((chunk_size, max_features), -1, dtype=torch.long)
        black_tensor = torch.full((chunk_size, max_features), -1, dtype=torch.long)
        stm_tensor = torch.zeros(chunk_size, dtype=torch.float32)
        target_tensor = torch.zeros(chunk_size, dtype=torch.float32)

        # Process all positions in this chunk
        for idx, (fen, result) in enumerate(positions):
            try:
                white_features, black_features, stm = extractor.fen_to_features(fen)

                # Fill tensors
                wf_len = min(len(white_features), max_features)
                bf_len = min(len(black_features), max_features)

                if wf_len > 0:
                    white_tensor[idx, :wf_len] = torch.tensor(
                        white_features[:wf_len], dtype=torch.long
                    )
                if bf_len > 0:
                    black_tensor[idx, :bf_len] = torch.tensor(
                        black_features[:bf_len], dtype=torch.long
                    )

                stm_tensor[idx] = stm
                target_tensor[idx] = result

            except Exception:
                # Leave as dummy data (already initialized)
                target_tensor[idx] = 0.5

        # Move this chunk to GPU immediately
        all_white_features.append(white_tensor.to(device))
        all_black_features.append(black_tensor.to(device))
        all_stm.append(stm_tensor.to(device))
        all_targets.append(target_tensor.to(device))

        print(f"    ✓ Loaded {chunk_size:,} positions to GPU")

    # Concatenate all chunks into single tensors on GPU
    print("\n  🔗 Concatenating all data...")
    white_features = torch.cat(all_white_features, dim=0)
    black_features = torch.cat(all_black_features, dim=0)
    stm = torch.cat(all_stm, dim=0)
    targets = torch.cat(all_targets, dim=0)

    load_time = time.time() - start_time
    total_positions = len(targets)
    gpu_memory_gb = (
        white_features.element_size() * white_features.nelement()
        + black_features.element_size() * black_features.nelement()
        + stm.element_size() * stm.nelement()
        + targets.element_size() * targets.nelement()
    ) / (1024**3)

    print("\n✅ Preloading complete!")
    print(f"   Total positions: {total_positions:,}")
    print(f"   GPU memory used: {gpu_memory_gb:.2f} GB")
    print(f"   Load time: {load_time:.1f}s")

    return white_features, black_features, stm, targets


def train_epoch_gpu(
    model,
    white_features,
    black_features,
    stm,
    targets,
    optimizer,
    criterion,
    batch_size,
    shuffle_indices=None,
):
    """
    Train for one epoch on GPU-resident data
    All data is already on GPU, so training is extremely fast
    """
    model.train()

    num_samples = len(targets)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Shuffle if requested
    if shuffle_indices is not None:
        white_features = white_features[shuffle_indices]
        black_features = black_features[shuffle_indices]
        stm = stm[shuffle_indices]
        targets = targets[shuffle_indices]

    total_loss = 0.0
    total_correct = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        # Slice batch directly from GPU tensors (zero copy!)
        batch_white = white_features[start_idx:end_idx]
        batch_black = black_features[start_idx:end_idx]
        batch_stm = stm[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]

        # Forward pass
        outputs = model(batch_white, batch_black, batch_stm)

        # Calculate loss
        loss = criterion(outputs, batch_targets)

        # Calculate accuracy
        pred_outcomes = torch.zeros_like(outputs)
        pred_outcomes[outputs < 0.33] = 0.0
        pred_outcomes[(outputs >= 0.33) & (outputs <= 0.67)] = 0.5
        pred_outcomes[outputs > 0.67] = 1.0

        correct = (torch.abs(pred_outcomes - batch_targets) < 0.1).sum().item()
        total_correct += correct

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            current_loss = total_loss / (batch_idx + 1)
            current_acc = (
                total_correct / ((batch_idx + 1) * batch_size) if batch_idx > 0 else 0
            )
            progress = (batch_idx + 1) / num_batches * 100

            print(
                f"    Batch {batch_idx + 1:,}/{num_batches:,} ({progress:.1f}%) | "
                f"Loss: {current_loss:.6f} | Acc: {current_acc:.4f}"
            )

            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "running_loss": current_loss,
                    "running_accuracy": current_acc,
                    "output_mean": outputs.mean().item(),
                    "output_std": outputs.std().item(),
                }
            )

    avg_loss = total_loss / num_batches
    accuracy = total_correct / num_samples

    return avg_loss, accuracy


def evaluate_gpu(
    model, white_features, black_features, stm, targets, criterion, batch_size
):
    """Evaluate model on GPU-resident data"""
    model.eval()

    num_samples = len(targets)
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            # Slice batch directly from GPU tensors
            batch_white = white_features[start_idx:end_idx]
            batch_black = black_features[start_idx:end_idx]
            batch_stm = stm[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            # Forward pass
            outputs = model(batch_white, batch_black, batch_stm)
            loss = criterion(outputs, batch_targets)

            # Calculate accuracy
            pred_outcomes = torch.zeros_like(outputs)
            pred_outcomes[outputs < 0.33] = 0.0
            pred_outcomes[(outputs >= 0.33) & (outputs <= 0.67)] = 0.5
            pred_outcomes[outputs > 0.67] = 1.0

            correct = (torch.abs(pred_outcomes - batch_targets) < 0.1).sum().item()
            total_correct += correct
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    accuracy = total_correct / num_samples

    return avg_loss, accuracy


def main():
    # Training parameters - optimized for L40S GPU
    batch_size = 262144  # Larger batch size since data is on GPU
    initial_lr = 0.002
    final_lr = 0.00001
    num_epochs = 25
    hidden_size = 256

    # Limit number of chunks to fit in GPU memory (48GB L40S)
    # Set to None to load all chunks
    max_train_chunks = None  # Adjust if you run out of memory
    max_test_chunks = 1  # Use 1 chunk for testing

    # Checkpoint settings
    resume_from_checkpoint = "latest_checkpoint_gpu.pt"

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
            "training_approach": "gpu_preload",
            "scheduler": "cosine_annealing",
            "max_train_chunks": max_train_chunks,
        },
    )

    # Device - must be CUDA for this script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("❌ This script requires CUDA GPU!")
        print("   Use train.py for CPU/MPS training")
        return

    print(f"✅ Using device: {device}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
    )

    wandb.config.update(
        {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
        }
    )

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

    # Preload all data to GPU
    train_white, train_black, train_stm, train_targets = preload_all_data_to_gpu(
        train_chunks, device, max_chunks=max_train_chunks
    )

    test_white, test_black, test_stm, test_targets = preload_all_data_to_gpu(
        test_chunks, device, max_chunks=max_test_chunks
    )

    # Create model
    model = create_model(hidden_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Created model with {total_params:,} parameters")

    wandb.config.update(
        {
            "total_parameters": total_params,
            "train_samples": len(train_targets),
            "test_samples": len(test_targets),
        }
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
        try:
            print(f"\n📂 Loading checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"]
            best_accuracy = checkpoint.get("best_accuracy", 0.0)
            best_loss = checkpoint.get("best_loss", float("inf"))

            print(f"✅ Resumed from epoch {start_epoch}")
            print(f"   Best accuracy: {best_accuracy:.4f}")
            print(f"   Best loss: {best_loss:.6f}")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting fresh training")
    else:
        print("\n🆕 Starting fresh training")

    print(
        f"\n📈 Learning rate: {initial_lr:.6f} → {final_lr:.6f} over {num_epochs} epochs"
    )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'=' * 80}")
        print(f"🔥 EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'=' * 80}")

        epoch_start_time = time.time()

        # Train
        print(f"\n  📊 Training on {len(train_targets):,} positions...")
        train_loss, train_acc = train_epoch_gpu(
            model,
            train_white,
            train_black,
            train_stm,
            train_targets,
            optimizer,
            criterion,
            batch_size,
        )

        print(f"  ✓ Training complete: Loss {train_loss:.6f} | Acc {train_acc:.4f}")

        # Evaluate
        print(f"\n  🧪 Evaluating on {len(test_targets):,} positions...")
        test_loss, test_acc = evaluate_gpu(
            model, test_white, test_black, test_stm, test_targets, criterion, batch_size
        )

        print(f"  ✓ Evaluation complete: Loss {test_loss:.6f} | Acc {test_acc:.4f}")

        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        # Log epoch metrics
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "epoch_time_seconds": epoch_time,
                "learning_rate": current_lr,
            }
        )

        print(f"\n  📊 EPOCH {epoch + 1} SUMMARY:")
        print(f"    Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.4f}")
        print(f"    Test Loss:  {test_loss:.6f} | Test Acc:  {test_acc:.4f}")
        print(f"    Learning Rate: {current_lr:.8f}")
        print(f"    Time: {epoch_time:.1f}s ({epoch_time / 60:.2f} min)")

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
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "best_accuracy": best_accuracy,
                    "best_loss": best_loss,
                },
                "best_model_gpu.pt",
            )
            wandb.save("best_model_gpu.pt")
            print(f"    ✓ 🏆 New best model saved! Acc: {test_acc:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "best_accuracy": best_accuracy,
                "best_loss": best_loss,
            },
            "latest_checkpoint_gpu.pt",
        )
        print("    💾 Checkpoint saved")

    print(f"\n{'=' * 80}")
    print("🎉 Training Complete!")
    print(f"{'=' * 80}")
    print(f"   Best test accuracy: {best_accuracy:.4f}")
    print(f"   Best test loss: {best_loss:.6f}")

    wandb.log(
        {
            "final_best_accuracy": best_accuracy,
            "final_best_loss": best_loss,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
