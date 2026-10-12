import os
import urllib.request
from pathlib import Path

WEIGHTS = {
    "det_10g.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx",
    "det_2.5g.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
    "det_500m.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx",
    "w600k_mbf.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx",
    "w600k_r50.onnx": "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx"
}

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"✓ Already exists: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            print(f"\r  Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end="")
        else:
            print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB", end="")

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=report_progress)
        print(f"\n✓ Downloaded: {dest_path.name}")
    except Exception as e:
        print(f"\n✗ Error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()

def main():
    root = Path(__file__).resolve().parent
    weights_dir = root / "src" / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {weights_dir}")
    for filename, url in WEIGHTS.items():
        download_file(url, weights_dir / filename)

    print("\nAll model weights check complete!")

if __name__ == "__main__":
    main()
