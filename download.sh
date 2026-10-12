#!/bin/bash

# Target directory
TARGET_DIR="src/models/weights"

if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
fi

echo "Downloading SCRFD and ArcFace ONNX weights into $TARGET_DIR..."

wget -nc -O "$TARGET_DIR/det_2.5g.onnx" https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx
wget -nc -O "$TARGET_DIR/det_500m.onnx" https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx
wget -nc -O "$TARGET_DIR/det_10g.onnx" https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx
wget -nc -O "$TARGET_DIR/w600k_mbf.onnx" https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx
wget -nc -O "$TARGET_DIR/w600k_r50.onnx" https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx

echo "Weights download complete!"
