# C++ Chess AI Engine

High-performance chess AI using ONNX Runtime for fast NNUE (Efficiently Updatable Neural Network) evaluation with C++ optimizations.

```bash
# Data pre-processing
uv run training/create_chunked_dataset.py --data-dir data/ --chunk-size 1000000 --output-dir chunked_dataset/

# Training
uv run training/prepare_and_train.py --chunked-training --positions-file chunked_dataset/
```
