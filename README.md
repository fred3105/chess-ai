# C++ Chess AI Engine

High-performance chess AI using ONNX Runtime for fast NNUE (Efficiently Updatable Neural Network) evaluation with C++ optimizations.

## 🎯 Features

- **C++ Optimized**: Fast search algorithms with position caching
- **ONNX Runtime**: Efficient neural network evaluation
- **NNUE Architecture**: 40,960 HalfKP features, 10.5M parameters  
- **Interactive GUI**: Pygame-based chess interface

## 🚀 Quick Start

### Setup

```bash
# Install dependencies with uv
uv sync

# Build C++ extension
cd cpp_search && python setup.py build_ext --inplace
```

### Playing Chess

```bash
# Start the chess GUI
uv run chess_ai.py
```

Choose your color (white/black) and search depth (1-8). The AI uses:
- C++ optimized search algorithms
- Position evaluation caching 
- Smart move ordering
- Alpha-beta pruning with iterative deepening

## 🏗️ Architecture

```
C++ Search Engine
  ↓
ONNX Runtime Evaluation
  ↓
NNUE Model (40,960 HalfKP features)
  ↓
Position Score (centipawns)
```

## 🔧 Components

- **`cpp_fast_chess_ai.py`**: Main chess AI engine with C++ optimizations
- **`chess_ai.py`**: Interactive GUI using pygame
- **`cpp_search/`**: C++ extension for fast search algorithms  
- **`nnue_model.py`**: Neural network model definitions
- **`checkpoints/`**: Trained model files

## ⚡ Performance

The C++ implementation provides significant speedups:
- Position caching for repeated evaluations
- Efficient alpha-beta search
- ONNX Runtime optimization
- Batch evaluation for move ordering
