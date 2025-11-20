import onnx

# Load the ONNX model
onnx_model_path = "C:\\Users\\ssmod\\Downloads\\finalization\\face-reidentification\\src\\models\\weights\\w600k_r50.onnx"
onnx_model = onnx.load(onnx_model_path)

# Modify batch size to be dynamic
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"

# Save the modified ONNX model
modified_onnx_path = "C:\\Users\\ssmod\\Downloads\\finalization\\face-reidentification\\src\\models\\weights\\w600k_r50_dynamic.onnx"
onnx.save(onnx_model, modified_onnx_path)

print("✅ ONNX model modified to support dynamic batch size.")
