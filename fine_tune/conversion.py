import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel

# Define the face recognition model wrapper class
class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Convert Fine-Tuned PyTorch ArcFace weights back to ONNX")
    parser.add_argument(
        "--onnx-base",
        type=str,
        default=str(repo_root / "src" / "models" / "weights" / "w600k_r50.onnx"),
        help="Path to base pretrained ArcFace ONNX model"
    )
    parser.add_argument(
        "--pth-weights",
        type=str,
        default=str(script_dir / "arcface_finetuned_final.pth"),
        help="Path to fine-tuned PyTorch state_dict (.pth)"
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=str(script_dir / "updated_model.onnx"),
        help="Path to save the updated ONNX model"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.onnx_base):
        raise FileNotFoundError(f"Base ONNX model not found: {args.onnx_base}")

    print(f"Loading base ONNX model from: {args.onnx_base}")
    onnx_model = onnx.load(args.onnx_base)
    pytorch_model = ConvertModel(onnx_model)  # Convert ONNX -> PyTorch

    # Wrap in FaceRecognitionModel
    model = FaceRecognitionModel(pytorch_model)

    if os.path.exists(args.pth_weights):
        print(f"Loading fine-tuned PyTorch weights from: {args.pth_weights}")
        try:
            state_dict = torch.load(args.pth_weights, map_location=torch.device("cpu"))
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Notice: Missing keys in state_dict: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Notice: Unexpected keys in state_dict: {len(unexpected_keys)}")

            print("PyTorch model weights loaded successfully!")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Warning: {args.pth_weights} not found. Exporting base converted model.")

    # Convert back to ONNX
    os.makedirs(os.path.dirname(os.path.abspath(args.output_onnx)), exist_ok=True)
    dummy_input = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_onnx,
        export_params=True,
        opset_version=args.opset_version,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"Updated ONNX model successfully saved at: {args.output_onnx}")


if __name__ == "__main__":
    main()
