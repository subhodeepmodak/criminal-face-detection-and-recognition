import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel

# Define the face recognition model class
class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# Load ONNX model first
onnx_model_path = "C:\\Users\\ssmod\\Downloads\\finalization\\face-reidentification\\src\\models\\weights\\w600k_r50.onnx"

# ✅ Load ONNX correctly
onnx_model = onnx.load(onnx_model_path)  
pytorch_model = ConvertModel(onnx_model)  # ✅ Convert ONNX → PyTorch

# Wrap in FaceRecognitionModel
model = FaceRecognitionModel(pytorch_model)

# Load fine-tuned PyTorch weights
fine_tuned_model_path = "C:\\Users\\ssmod\\Downloads\\finalization\\face-reidentification\\fine_tune\\arcface_finetuned_final.pth"
try:
    state_dict = torch.load(fine_tuned_model_path, map_location=torch.device("cpu"))

    # Handle missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Convert back to ONNX
updated_onnx_path = "C:\\Users\\ssmod\\Downloads\\finalization\\face-reidentification\\fine_tune\\updated_model.onnx"

dummy_input = torch.randn(1, 3, 112, 112)  # Example input (batch_size=1, 3 color channels, 112x112 image)
torch.onnx.export(model, dummy_input, updated_onnx_path, 
                  export_params=True, opset_version=11, 
                  input_names=["input"], output_names=["output"])

print(f"Updated ONNX model saved at {updated_onnx_path}")
