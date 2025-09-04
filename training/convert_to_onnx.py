"""
Convert trained PyTorch NNUE model to ONNX format for faster inference
"""

from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.onnx

from nnue_model import create_nnue_model


def convert_model_to_onnx(
    pytorch_model_path: str = "checkpoints/best_model.pth",
    onnx_output_path: str = "checkpoints/nnue_model.onnx",
    device: str = "cpu",
):
    """Convert PyTorch NNUE model to ONNX format"""

    print(f"Loading PyTorch model from {pytorch_model_path}")

    # Load the trained model
    model = create_nnue_model()
    checkpoint = torch.load(pytorch_model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    # Create dummy input tensors matching expected input shape
    batch_size = 1
    feature_dim = 40960  # HalfKP feature dimension

    dummy_white_features = torch.zeros(
        batch_size, feature_dim, dtype=torch.float32, device=device
    )
    dummy_black_features = torch.zeros(
        batch_size, feature_dim, dtype=torch.float32, device=device
    )

    # Set some features to 1 to simulate real input
    dummy_white_features[0, :100] = 1.0
    dummy_black_features[0, 50:150] = 1.0

    print("Converting to ONNX format...")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_white_features, dummy_black_features),
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["white_features", "black_features"],
        output_names=["evaluation"],
        dynamic_axes={
            "white_features": {0: "batch_size"},
            "black_features": {0: "batch_size"},
            "evaluation": {0: "batch_size"},
        },
    )

    print(f"Model successfully converted to ONNX: {onnx_output_path}")

    # Verify the ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)

    # Test inference with ONNX Runtime
    print("Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(onnx_output_path)

    # Convert PyTorch tensors to numpy for ONNX Runtime
    white_np = dummy_white_features.cpu().numpy()
    black_np = dummy_black_features.cpu().numpy()

    # Run inference
    ort_inputs = {"white_features": white_np, "black_features": black_np}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare with PyTorch output
    with torch.no_grad():
        pytorch_output = model(dummy_white_features, dummy_black_features)

    print(f"PyTorch output: {pytorch_output.item():.4f}")
    print(f"ONNX output: {ort_outputs[0][0][0]:.4f}")
    print(f"Difference: {abs(pytorch_output.item() - ort_outputs[0][0][0]):.6f}")

    print("ONNX conversion completed successfully!")
    return onnx_output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "checkpoints/best_model.pth"

    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        sys.exit(1)

    convert_model_to_onnx(model_path)
