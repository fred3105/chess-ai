# NNUE Chess AI - Grandmaster Training

High-performance NNUE (Efficiently Updatable Neural Network) chess evaluation trained on your Lichess Elite grandmaster game collection with perfect **50/50 balance** between quiet and complex positions.

## 🎯 Features

- **Balanced Training**: Exactly 50% quiet positions, 50% complex (tactical/imbalanced/unusual)
- **Elite Dataset**: Your 79 Lichess Elite PGN files (2200+ ELO, 2013-2020)
- **Efficient Architecture**: 40,960 HalfKP features, 10.5M parameters
- **M3 Pro Optimized**: MPS acceleration, optimized batch sizes
- **Wandb Integration**: Complete training tracking and visualization
- **Clean Codebase**: Modern Python, type hints, proper error handling

## 🚀 Quick Start

### Setup

```bash
# Install dependencies with uv
uv sync
```

### Training

```bash
# Standard training (100k positions, 50 epochs)
uv run train.py

# Custom training
uv run train.py \
    --train-size 1000000 \
    --val-size 10000 \
    --epochs 100 \
    --quiet-ratio 0.5 \
    --wandb-project my-chess-nnue

# Quick test run
python train.py \
    --train-size 10000 \
    --val-size 2000 \
    --epochs 10 \
    --batch-size 2048
```

## 📊 Dataset Balance

Your system automatically ensures perfect distribution:

- **Quiet (50%)**: Positional play, no tactics, balanced material
- **Complex (50%)**:
  - **Tactical**: Checks, captures, immediate threats
  - **Imbalanced**: Material differences >3 pawns
  - **Unusual**: Endgames, promoted pieces, rare patterns

## 🏗️ Architecture

```
Input: HalfKP Features (40,960 dimensions)
  ↓
Feature Transformer (256 hidden)
  ↓
L1 Layer (32 hidden) + ClippedReLU
  ↓
L2 Layer (32 hidden) + ClippedReLU
  ↓
Output (1 value, centipawns)
```

## 📈 Wandb Integration

All training metrics automatically logged:

- Training/validation loss curves
- Learning rate scheduling
- Model gradients and parameters
- Dataset distribution statistics
- Hardware utilization

## 💡 Usage Examples

```bash
# Research run - small, fast
uv run train.py --train-size 25000 --epochs 20

# Production run - large, comprehensive
uv run train.py --train-size 1000000 --epochs 100

# Balanced experiment - test different ratios
uv run train.py --quiet-ratio 0.3  # 30% quiet, 70% complex
uv run train.py --quiet-ratio 0.7  # 70% quiet, 30% complex

# Hardware optimization
uv run train.py --batch-size 8192 --num-workers 6  # More parallel
uv run train.py --batch-size 2048 --num-workers 2  # Less memory
```

## ✅ System Requirements

- **Hardware**: Apple Silicon (M1/M2/M3) recommended
- **Memory**: 8GB+ RAM for large datasets
- **Storage**: 5GB+ for checkpoints and data
- **Python**: 3.10+
