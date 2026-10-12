import os
import argparse
from pathlib import Path
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from onnx2pytorch import ConvertModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Enable CUDA optimizations if available
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Fine-tune ArcFace for custom identities")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(script_dir / "dataset"),
        help="Path to training dataset with identity subfolders"
    )
    parser.add_argument(
        "--onnx-base",
        type=str,
        default=str(repo_root / "src" / "models" / "weights" / "w600k_r50.onnx"),
        help="Path to base pretrained ArcFace ONNX model"
    )
    parser.add_argument(
        "--output-pth",
        type=str,
        default=str(script_dir / "arcface_finetuned_best.pth"),
        help="Path to save best fine-tuned PyTorch model"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total if total > 0 else 0


def set_bn_eval(module):
    """Set BatchNorm layers to eval mode to preserve pretrained statistics."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.eval()
    for child in module.children():
        set_bn_eval(child)


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if not os.path.exists(args.onnx_base):
        raise FileNotFoundError(f"Base ONNX model not found: {args.onnx_base}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset path not found: {args.dataset}")

    # Load ONNX Model and convert to PyTorch
    print(f"Loading base ArcFace model from {args.onnx_base}...")
    onnx_model = onnx.load(args.onnx_base)
    arcface_pytorch = ConvertModel(onnx_model, experimental=True)
    arcface_pytorch.to(device)
    set_bn_eval(arcface_pytorch)

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load Dataset
    dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} images from {num_classes} identity classes: {dataset.classes}")

    # Modify last layer for custom identities
    last_layer_name = list(arcface_pytorch._modules.keys())[-1]
    arcface_pytorch._modules[last_layer_name] = nn.Linear(512, num_classes)
    arcface_pytorch.to(device)

    # Freeze backbone, train only new classification head
    for name, param in arcface_pytorch.named_parameters():
        param.requires_grad = last_layer_name in name

    optimizer = optim.Adam(arcface_pytorch._modules[last_layer_name].parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        arcface_pytorch.train()
        set_bn_eval(arcface_pytorch)
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = arcface_pytorch(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        scheduler.step(avg_loss)

        accuracy = evaluate_model(arcface_pytorch, dataloader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs(os.path.dirname(os.path.abspath(args.output_pth)), exist_ok=True)
            torch.save(arcface_pytorch.state_dict(), args.output_pth)
            print(f"Saved best checkpoint to {args.output_pth} ({best_accuracy:.2f}%)")

    print(f"Fine-tuning completed! Best Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
