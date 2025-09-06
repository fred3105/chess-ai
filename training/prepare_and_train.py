#!/usr/bin/env python3
"""
Load the saved positional positions, create train/val split, and start training immediately.
Optimized for M3 Pro MacBook training.
"""

import argparse
import logging
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from nnue_dataset import ChessPosition, NNUEDataset, PositionType
from nnue_model import HalfKPFeatureExtractor, create_nnue_model
from tqdm import tqdm

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_positions_from_chunks(
    chunk_dir: str, chunk_index: int = 0
) -> tuple[list[ChessPosition], int]:
    """Load positions from chunked dataset directory"""
    chunk_path = Path(chunk_dir)
    metadata_file = chunk_path / "dataset_metadata.pkl"

    if not metadata_file.exists():
        # Fall back to single file loading
        return load_positions_chunked_single(chunk_dir, 500_000, chunk_index)

    logger.info(f"🔄 Loading from chunked dataset: {chunk_dir}")

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    total_positions = metadata["total_positions"]
    total_chunks = metadata["total_chunks"]

    if chunk_index >= total_chunks:
        logger.info("✅ All chunks processed!")
        return [], total_positions

    chunk_file = chunk_path / metadata["chunks"][chunk_index]
    logger.info(f"📦 Loading chunk {chunk_index + 1}/{total_chunks}: {chunk_file}")

    with open(chunk_file, "rb") as f:
        positions = pickle.load(f)

    logger.info(f"✅ Loaded {len(positions):,} positions from chunk {chunk_index + 1}")
    return positions, total_positions


def load_positions_chunked_single(
    positions_file: str, chunk_size: int = 500_000, chunk_index: int = 0
) -> tuple[list[ChessPosition], int]:
    """Load a chunk of positions from a single large dataset file"""
    logger.info(f"🔄 Loading chunk {chunk_index + 1} from {positions_file}...")

    # Get file size for progress estimation
    file_size_gb = Path(positions_file).stat().st_size / 1e9
    logger.info(f"📁 File size: {file_size_gb:.1f} GB")

    try:
        logger.info("🔄 Attempting to load pickle file...")
        with open(positions_file, "rb") as f:
            # Try to load with different protocols
            try:
                all_positions = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"❌ Standard pickle load failed: {e}")
                logger.error("🚨 Dataset file appears corrupted!")
                logger.error(
                    "💡 Suggestion: Recreate dataset with: uv run training/create_chunked_dataset.py"
                )
                raise EOFError("Pickle file appears corrupted or incomplete") from e

        total_positions = len(all_positions)
        logger.info(f"📊 Total positions in file: {total_positions:,}")

        # Calculate chunk boundaries
        start_idx = chunk_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_positions)

        if start_idx >= total_positions:
            logger.info("✅ All chunks processed!")
            return [], total_positions

        # Shuffle all positions first, then take chunk (ensures randomness across chunks)
        if chunk_index == 0:  # Only shuffle on first chunk to maintain consistency
            logger.info("🔀 Shuffling all positions for chunk consistency...")
            random.shuffle(all_positions)

        chunk_positions = all_positions[start_idx:end_idx]

        logger.info(
            f"✅ Loaded chunk {chunk_index + 1}: positions {start_idx:,} to {end_idx:,} ({len(chunk_positions):,} positions)"
        )
        logger.info(f"📈 Progress: {end_idx / total_positions * 100:.1f}% of dataset")

        return chunk_positions, total_positions

    except MemoryError:
        logger.error("❌ Out of memory! Even loading full file for chunking failed.")
        logger.info("💡 You may need to pre-process the data into smaller files")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading positions: {e}")
        raise


def load_positions_safe(
    positions_file: str, max_positions: int | None = None
) -> list[ChessPosition]:
    """Safely load positions with memory management (legacy method)"""
    logger.info(f"🔄 Loading positions from {positions_file}...")

    # Get file size for progress estimation
    file_size_gb = Path(positions_file).stat().st_size / 1e9
    logger.info(f"📁 File size: {file_size_gb:.1f} GB")

    if file_size_gb > 3.0:
        logger.warning(
            "⚠️  Large file detected - this may take several minutes and use significant RAM"
        )
        logger.info("💡 Consider using --chunked-training for better memory management")

    try:
        with open(positions_file, "rb") as f:
            positions = pickle.load(f)

        total_positions = len(positions)
        logger.info(f"✅ Loaded {total_positions:,} positions ({file_size_gb:.1f} GB)")

        # Optionally limit positions for memory management
        if max_positions and total_positions > max_positions:
            logger.info(f"🔪 Trimming to {max_positions:,} positions to manage memory")
            random.shuffle(positions)  # Shuffle before trimming for random subset
            positions = positions[:max_positions]
            logger.info(f"✅ Using {len(positions):,} positions")

        return positions

    except MemoryError:
        logger.error("❌ Out of memory! File too large to load all at once.")
        logger.info("💡 Try running with --chunked-training instead")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading positions: {e}")
        raise


def analyze_positions(positions: list[ChessPosition]) -> dict:
    """Analyze the position distribution"""
    # Position type distribution
    type_counts = {}
    for pos in positions:
        pos_type = pos.position_type
        type_counts[pos_type] = type_counts.get(pos_type, 0) + 1

    # Evaluation statistics
    evaluations = [pos.evaluation for pos in positions]
    outcomes = [pos.outcome for pos in positions if pos.outcome is not None]

    import numpy as np

    stats = {
        "total_positions": len(positions),
        "position_type_distribution": {
            pos_type: {"count": count, "percentage": count / len(positions) * 100}
            for pos_type, count in type_counts.items()
        },
        "evaluation_stats": {
            "mean": float(np.mean(evaluations)),
            "std": float(np.std(evaluations)),
            "min": float(np.min(evaluations)),
            "max": float(np.max(evaluations)),
        },
        "outcome_distribution": {
            "white_wins": outcomes.count(1.0) / len(outcomes) if outcomes else 0,
            "draws": outcomes.count(0.0) / len(outcomes) if outcomes else 0,
            "black_wins": outcomes.count(-1.0) / len(outcomes) if outcomes else 0,
        },
    }

    return stats


def create_balanced_split(
    positions: list[ChessPosition], val_size: int = 20000
) -> tuple[list[ChessPosition], list[ChessPosition]]:
    """Create balanced train/validation split with fixed validation size"""
    logger.info(f"🔄 Creating train/validation split (val_size={val_size:,})...")

    # Separate by position type for balanced splitting
    quiet_positions = [p for p in positions if p.position_type == PositionType.QUIET]
    imbalanced_positions = [
        p for p in positions if p.position_type == PositionType.IMBALANCED
    ]
    unusual_positions = [
        p for p in positions if p.position_type == PositionType.UNUSUAL
    ]

    logger.info("📊 Available position type counts:")
    logger.info(f"  🔷 Quiet: {len(quiet_positions):,}")
    logger.info(f"  ⚖️  Imbalanced: {len(imbalanced_positions):,}")
    logger.info(f"  🔸 Unusual: {len(unusual_positions):,}")

    # Shuffle all types
    random.shuffle(quiet_positions)
    random.shuffle(imbalanced_positions)
    random.shuffle(unusual_positions)

    # Create balanced validation set (maintain proportions)
    total_positions = len(positions)
    val_positions = []

    # Calculate proportional validation sizes
    quiet_val_size = int(val_size * len(quiet_positions) / total_positions)
    imbalanced_val_size = int(val_size * len(imbalanced_positions) / total_positions)
    unusual_val_size = val_size - quiet_val_size - imbalanced_val_size

    # Take validation positions
    val_positions.extend(quiet_positions[:quiet_val_size])
    val_positions.extend(imbalanced_positions[:imbalanced_val_size])
    val_positions.extend(unusual_positions[:unusual_val_size])

    # Use all remaining positions for training
    train_positions = (
        quiet_positions[quiet_val_size:]
        + imbalanced_positions[imbalanced_val_size:]
        + unusual_positions[unusual_val_size:]
    )

    # Final shuffle
    random.shuffle(train_positions)
    random.shuffle(val_positions)

    logger.info("✅ Split created:")
    logger.info(
        f"  📈 Training: {len(train_positions):,} positions ({len(train_positions) / 1e6:.1f}M)"
    )
    logger.info(f"  📊 Validation: {len(val_positions):,} positions")
    logger.info(
        f"  📋 Training uses {len(train_positions) / total_positions * 100:.1f}% of all data"
    )

    return train_positions, val_positions


def create_datasets(
    train_positions: list[ChessPosition], val_positions: list[ChessPosition]
) -> tuple[NNUEDataset, NNUEDataset]:
    """Create NNUE datasets from positions"""
    logger.info("🔄 Creating NNUE datasets...")

    feature_extractor = HalfKPFeatureExtractor()
    train_dataset = NNUEDataset(train_positions, feature_extractor)
    val_dataset = NNUEDataset(val_positions, feature_extractor)

    logger.info(
        f"✅ Datasets created - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}"
    )
    return train_dataset, val_dataset


class NNUETrainerOptimized:
    """NNUE trainer optimized for large datasets following best practices"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        optimizer_type: str = "sgd",
        momentum: float = 0.9,
        scheduler_type: str = "step",
        lr_step_size: int = 15,
        lr_gamma: float = 0.1,
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb

        # Create optimizer (SGD preferred for NNUE)
        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        else:  # adam
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # Create scheduler
        if scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_step_size, gamma=lr_gamma
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50,  # Adjust based on epochs
            )
        else:  # plateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=lr_gamma, patience=5, min_lr=1e-6
            )

        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.best_val_loss = float("inf")

        logger.info("🔧 Trainer initialized:")
        logger.info(f"  Optimizer: {optimizer_type.upper()}")
        logger.info(f"  Scheduler: {scheduler_type}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(
            f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader, desc=f"🏃 Epoch {self.epoch + 1}", leave=False
        )

        for batch_idx, (white_features, black_features, targets) in enumerate(
            progress_bar
        ):
            # Move to device
            white_features = white_features.to(self.device, non_blocking=True)
            black_features = black_features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(white_features, black_features)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            current_lr = self.get_lr()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

            # Log every 1000 batches for large datasets
            if batch_idx > 0 and batch_idx % 1000 == 0:
                logger.info(
                    f"  📊 Batch {batch_idx}/{num_batches}: loss={loss.item():.4f}, avg_loss={total_loss / (batch_idx + 1):.4f}"
                )

                # Log to wandb
                if self.use_wandb:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/avg_loss": total_loss / (batch_idx + 1),
                            "train/learning_rate": current_lr,
                            "epoch": self.epoch,
                            "batch": batch_idx + self.epoch * num_batches,
                        }
                    )

        epoch_loss = total_loss / num_batches
        progress_bar.close()
        return epoch_loss

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        progress_bar = tqdm(self.val_loader, desc="📊 Validation", leave=False)

        with torch.no_grad():
            for white_features, black_features, targets in progress_bar:
                # Move to device
                white_features = white_features.to(self.device, non_blocking=True)
                black_features = black_features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                predictions = self.model(white_features, black_features)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        progress_bar.close()
        return total_loss / num_batches

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 5,
        early_stopping_patience: int = 15,
    ):
        """Train the model"""
        Path(checkpoint_dir).mkdir(exist_ok=True)

        patience_counter = 0
        start_time = time.time()

        logger.info(f"🚀 Starting training for {num_epochs} epochs...")
        logger.info(f"📈 Training batches per epoch: {len(self.train_loader):,}")
        logger.info(f"📊 Validation batches: {len(self.val_loader):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Update scheduler
            if hasattr(self.scheduler, "step"):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time

            # Log epoch results
            logger.info(f"⭐ Epoch {epoch + 1}/{num_epochs} completed:")
            logger.info(f"  📈 Train loss: {train_loss:.6f}")
            logger.info(f"  📊 Val loss: {val_loss:.6f}")
            logger.info(f"  🎯 Learning rate: {self.get_lr():.6f}")
            logger.info(
                f"  ⏱️  Epoch time: {epoch_time / 60:.1f}m, Total: {total_time / 60:.1f}m"
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": train_loss,
                        "val/epoch_loss": val_loss,
                        "train/learning_rate": self.get_lr(),
                        "time/epoch_minutes": epoch_time / 60,
                        "time/total_hours": total_time / 3600,
                        "epoch": epoch + 1,
                    }
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                best_path = Path(checkpoint_dir) / "best_model.pth"
                self.save_checkpoint(str(best_path))
                logger.info(f"  💾 New best model saved! Val loss: {val_loss:.6f}")
                if self.use_wandb:
                    wandb.log({"val/best_loss": val_loss})
            else:
                patience_counter += 1
                logger.info(f"  ⏳ No improvement for {patience_counter} epochs")

            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = Path(checkpoint_dir) / f"model_epoch_{epoch + 1}.pth"
                self.save_checkpoint(str(checkpoint_path))
                logger.info(f"  💾 Checkpoint saved: epoch_{epoch + 1}.pth")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"🛑 Early stopping after {patience_counter} epochs without improvement"
                )
                break

            logger.info("")  # Empty line for readability

        total_time = time.time() - start_time
        logger.info("🎉 Training completed!")
        logger.info(f"⏱️  Total time: {total_time / 3600:.1f} hours")
        logger.info(f"🏆 Best validation loss: {self.best_val_loss:.6f}")


def train_with_chunks(args, use_wandb: bool) -> int:
    """Train using chunked data loading for large datasets"""
    logger.info("🔄 === CHUNKED TRAINING MODE ===")
    logger.info(f"📊 Chunk size: {args.chunk_size:,} positions")
    logger.info(f"🔄 Epochs per chunk: {args.epochs_per_chunk}")

    device = get_device()

    # Create model once
    model = create_nnue_model(
        hidden_dim=args.hidden_dim,
        l1_hidden_dim=args.l1_hidden_dim,
        l2_hidden_dim=args.l2_hidden_dim,
        dropout_rate=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model parameters: {total_params:,}")

    # Create validation set from first chunk
    logger.info("🔄 Loading validation set from first chunk...")
    val_chunk, total_positions = load_positions_from_chunks(args.positions_file, 0)

    if not val_chunk:
        logger.error("❌ Failed to load any positions")
        return 1

    # Use a portion of first chunk for validation
    val_positions = val_chunk[: args.val_size]
    logger.info(f"📊 Created validation set: {len(val_positions):,} positions")

    # Calculate total chunks
    total_chunks = (total_positions + args.chunk_size - 1) // args.chunk_size
    logger.info(f"📈 Total chunks to process: {total_chunks}")
    logger.info(f"📊 Total positions: {total_positions:,}")

    # Create validation dataset (reused across chunks)
    feature_extractor = HalfKPFeatureExtractor()
    val_dataset = NNUEDataset(val_positions, feature_extractor)
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Track training across chunks
    trainer = None
    chunk_start_time = time.time()

    for chunk_idx in range(total_chunks):
        logger.info(f"\n🎯 === CHUNK {chunk_idx + 1}/{total_chunks} ===")

        # Load chunk
        chunk_positions, _ = load_positions_from_chunks(args.positions_file, chunk_idx)

        if not chunk_positions:
            logger.info("✅ All chunks processed!")
            break

        # Skip validation positions if this is the first chunk
        if chunk_idx == 0:
            chunk_positions = chunk_positions[args.val_size :]

        logger.info(f"📊 Training positions in this chunk: {len(chunk_positions):,}")

        # Create training dataset for this chunk
        train_dataset = NNUEDataset(chunk_positions, feature_extractor)
        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Critical for proper training
            num_workers=args.num_workers,
            persistent_workers=True,
            drop_last=True,
        )

        # Create or update trainer
        if trainer is None:
            trainer = NNUETrainerOptimized(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                scheduler_type=args.scheduler,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma,
                use_wandb=use_wandb,
            )
        else:
            # Update data loaders for existing trainer
            trainer.train_loader = train_loader

        # Train on this chunk
        logger.info(
            f"🚀 Training on chunk {chunk_idx + 1} for {args.epochs_per_chunk} epochs..."
        )

        trainer.train(
            num_epochs=args.epochs_per_chunk,
            checkpoint_dir=args.checkpoint_dir,
            save_every=1,  # Save after each chunk
            early_stopping_patience=999,  # Don't early stop within chunks
        )

        # Clear chunk data to free memory
        del chunk_positions, train_dataset, train_loader

        logger.info(f"✅ Completed chunk {chunk_idx + 1}")
        logger.info(
            f"⏱️  Chunk time: {(time.time() - chunk_start_time) / 60:.1f} minutes"
        )
        chunk_start_time = time.time()

    total_time = time.time() - chunk_start_time
    logger.info("\n🎉 === CHUNKED TRAINING COMPLETE ===")
    logger.info(f"⏱️  Total training time: {total_time / 3600:.1f} hours")
    logger.info(f"💾 Best model: {args.checkpoint_dir}/best_model.pth")
    logger.info(
        f"📊 Trained on all {total_positions:,} positions across {total_chunks} chunks"
    )

    if use_wandb:
        wandb.log(
            {
                "final/total_hours": total_time / 3600,
                "final/best_val_loss": trainer.best_val_loss if trainer else 0,
                "final/positions_trained": total_positions,
                "final/chunks_processed": total_chunks,
            }
        )
        wandb.finish()

    return 0


def train_with_full_dataset(args, use_wandb: bool) -> int:
    """Train with full dataset loaded into memory"""
    logger.info("🔄 === FULL DATASET TRAINING MODE ===")

    # Load positions
    positions = load_positions_safe(args.positions_file, args.max_positions)

    # Analyze dataset
    logger.info("📊 Analyzing dataset...")
    stats = analyze_positions(positions)

    logger.info("📈 Dataset Statistics:")
    logger.info(f"  Total positions: {stats['total_positions']:,}")
    for pos_type, data in stats["position_type_distribution"].items():
        logger.info(f"  {pos_type}: {data['count']:,} ({data['percentage']:.1f}%)")

    eval_stats = stats["evaluation_stats"]
    logger.info(
        f"  Evaluation: {eval_stats['mean']:.1f} ± {eval_stats['std']:.1f} (range: {eval_stats['min']:.0f} to {eval_stats['max']:.0f})"
    )

    outcome_dist = stats["outcome_distribution"]
    logger.info(
        f"  Outcomes: W={outcome_dist['white_wins']:.1%}, D={outcome_dist['draws']:.1%}, B={outcome_dist['black_wins']:.1%}"
    )
    logger.info("")

    # Create train/val split
    train_positions, val_positions = create_balanced_split(positions, args.val_size)

    # Create datasets
    train_dataset, val_dataset = create_datasets(train_positions, val_positions)

    # Create model
    model = create_nnue_model(
        hidden_dim=args.hidden_dim,
        l1_hidden_dim=args.l1_hidden_dim,
        l2_hidden_dim=args.l2_hidden_dim,
        dropout_rate=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model parameters: {total_params:,}")

    # Create data loaders
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Create trainer
    trainer = NNUETrainerOptimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=get_device(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        scheduler_type=args.scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        use_wandb=use_wandb,
    )

    # Start training
    start_time = time.time()
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience,
    )

    total_time = time.time() - start_time

    # Final summary
    logger.info("🎉 === TRAINING COMPLETE ===")
    logger.info(f"⏱️  Total training time: {total_time / 3600:.1f} hours")
    logger.info(f"💾 Best model: {args.checkpoint_dir}/best_model.pth")
    logger.info(f"📊 Trained on: {len(train_dataset):,} positions")

    if use_wandb:
        wandb.log(
            {
                "final/total_hours": total_time / 3600,
                "final/best_val_loss": trainer.best_val_loss,
                "final/positions_trained": len(train_dataset),
            }
        )
        wandb.finish()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets and train NNUE")

    # Dataset parameters
    parser.add_argument(
        "--positions-file",
        default="full_dataset/all_positional_positions.pkl",
        help="Path to saved positions file",
    )
    parser.add_argument(
        "--val-size", type=int, default=20000, help="Validation set size"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        help="Limit total positions (for memory management)",
    )
    parser.add_argument(
        "--chunked-training",
        action="store_true",
        help="Use chunked training for large datasets",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Positions per chunk (500K for M3 Pro)",
    )
    parser.add_argument(
        "--epochs-per-chunk", type=int, default=4, help="Epochs to train on each chunk"
    )

    # Model parameters (NNUE standard - optimized for speed/strength balance)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension (256=fast, 512=balanced, 1024=slow)",
    )
    parser.add_argument(
        "--l1-hidden-dim", type=int, default=32, help="L1 hidden dimension"
    )
    parser.add_argument(
        "--l2-hidden-dim", type=int, default=32, help="L2 hidden dimension"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (0.0 for NNUE)"
    )

    # Training parameters (NNUE best practices - optimized for 200 epoch training)
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs (200+ for strongest NNUE)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (minimal for M3 Pro memory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.007,
        help="Initial learning rate (conservative for long training)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer (SGD preferred for NNUE)",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--scheduler",
        choices=["step", "cosine", "plateau"],
        default="step",
        help="LR scheduler",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=25,
        help="LR step size for step scheduler (longer for 200 epochs)",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        help="LR decay factor (gentler for long training)",
    )

    # System parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of data loader workers (good for M3 Pro)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )

    # Training control
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Early stopping patience (longer for 200 epochs)",
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Wandb parameters
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chess-eval-optimization",
        help="Wandb project name",
    )
    parser.add_argument("--wandb-name", type=str, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Setup
    set_random_seeds(args.seed)
    device = get_device()

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"nnue-{args.hidden_dim}-{args.epochs}ep",
            config=vars(args),
            tags=["fast-inference", "51M-positions", "positional", "nnue"],
        )
        logger.info("🔗 Wandb logging initialized")

    logger.info("🚀 === NNUE TRAINING FOR FAST INFERENCE ===")
    logger.info(f"💻 Device: {device}")
    logger.info(f"📁 Positions file: {args.positions_file}")
    logger.info(
        f"⚙️  Model: {args.hidden_dim}-{args.l1_hidden_dim}-{args.l2_hidden_dim} (optimized for speed)"
    )
    logger.info(f"🎯 Batch size: {args.batch_size}")
    logger.info(f"📊 Epochs: {args.epochs} (long training for strength)")
    logger.info(f"👷 Workers: {args.num_workers}")
    logger.info("")

    # Check if file exists
    if not Path(args.positions_file).exists():
        logger.error(f"❌ Positions file not found: {args.positions_file}")
        return 1

    if args.chunked_training:
        return train_with_chunks(args, use_wandb)
    else:
        return train_with_full_dataset(args, use_wandb)


if __name__ == "__main__":
    exit(main())
