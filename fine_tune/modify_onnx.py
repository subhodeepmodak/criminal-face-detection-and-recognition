import os
import argparse
from pathlib import Path
import onnx

def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Modify ONNX model to support dynamic batch size")
    parser.add_argument(
        "--input-onnx",
        type=str,
        default=str(repo_root / "src" / "models" / "weights" / "w600k_r50.onnx"),
        help="Input ONNX model path"
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default=str(repo_root / "src" / "models" / "weights" / "w600k_r50_dynamic.onnx"),
        help="Output ONNX model path"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.input_onnx):
        raise FileNotFoundError(f"Input ONNX model not found: {args.input_onnx}")

    print(f"Loading ONNX model from: {args.input_onnx}")
    onnx_model = onnx.load(args.input_onnx)

    # Modify batch size to be dynamic
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_onnx)), exist_ok=True)
    onnx.save(onnx_model, args.output_onnx)
    print(f"✅ Modified ONNX model with dynamic batch size saved at: {args.output_onnx}")

if __name__ == "__main__":
    main()
