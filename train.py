"""
NNUE training script with wandb integration
Optimized for grandmaster game datasets
"""

import os
import time
import logging
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import wandb
from tqdm import tqdm

from nnue_model import create_nnue_model
from nnue_dataset import create_balanced_datasets


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NNUETrainer:
    """Clean NNUE trainer with wandb integration"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str = "mps",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        # Loss
        self.criterion = nn.MSELoss()

        # State
        self.epoch = 0
        self.best_val_loss = float("inf")

        logger.info(f"Trainer initialized with device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (white_features, black_features, targets) in enumerate(
            progress_bar
        ):
            # Move to device
            white_features = white_features.to(self.device)
            black_features = black_features.to(self.device)
            targets = targets.to(self.device)

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
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{self.get_lr():.6f}"}
            )

            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.get_lr(),
                        "epoch": self.epoch,
                    }
                )

        # Normalize by dataset size instead of number of batches  
        return total_loss / len(self.train_loader.dataset)

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for white_features, black_features, targets in progress_bar:
                # Move to device
                white_features = white_features.to(self.device)
                black_features = black_features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                predictions = self.model(white_features, black_features)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # Normalize by dataset size instead of number of batches
        return total_loss / len(self.val_loader.dataset)

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
        os.makedirs(checkpoint_dir, exist_ok=True)

        patience_counter = 0
        start_time = time.time()

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": train_loss,
                        "val/epoch_loss": val_loss,
                        "train/learning_rate": self.get_lr(),
                        "epoch": epoch,
                    }
                )

            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                f"Time: {elapsed_time / 60:.1f}m"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(Path(checkpoint_dir) / "best_model.pth")
                logger.info(f"New best model saved: {val_loss:.4f}")

                if self.use_wandb:
                    wandb.log({"val/best_loss": val_loss})
            else:
                patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    Path(checkpoint_dir) / f"model_epoch_{epoch + 1}.pth"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {patience_counter} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")

        if self.use_wandb:
            wandb.log({"training/total_time_minutes": total_time / 60})


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train NNUE Chess Evaluation Model")

    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--l1-hidden-dim", type=int, default=32)
    parser.add_argument("--l2-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Data parameters
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--min-elo", type=int, default=2200)
    parser.add_argument("--quiet-ratio", type=float, default=0.5)

    # System parameters
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Wandb parameters
    parser.add_argument("--wandb-project", type=str, default="chess-eval-optimization")
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # Setup
    set_random_seeds(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=["grandmaster-games", "nnue", "balanced-dataset"],
        )
        logger.info("Initialized wandb logging")

    # Create model
    logger.info("Creating NNUE model...")
    model = create_nnue_model(
        hidden_dim=args.hidden_dim,
        l1_hidden_dim=args.l1_hidden_dim,
        l2_hidden_dim=args.l2_hidden_dim,
        dropout_rate=args.dropout,
    )

    # Get PGN files from data directory
    import glob
    pgn_files = glob.glob("data/*.pgn")
    logger.info(f"Found {len(pgn_files)} PGN files")
    
    # Create datasets
    logger.info("Creating balanced datasets from grandmaster games...")
    train_dataset, val_dataset = create_balanced_datasets(
        data_dir="data",
        train_size=args.train_size,
        val_size=args.val_size,
        pgn_files=pgn_files,
        min_elo=args.min_elo,
        quiet_ratio=args.quiet_ratio,
    )

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Create trainer
    trainer = NNUETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=use_wandb,
    )

    # Watch model with wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=1000)

    # Train
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=5,
        early_stopping_patience=15,
    )

    logger.info("Training completed!")

    # Finish wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
