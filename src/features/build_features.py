"""
Image preprocessing for model inference.

Two modes:
  - "dino":     518x518 center crop — required by DINOv2 ViT-Large (M1)
  - "standard": 224x224 resize     — used by EfficientNet (M3) and ResNet50 (M2 feature extractor)

Both apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""
import torch
from torchvision import transforms
from PIL import Image


def preprocess_image(img_path, mode="standard"):
    """
    Load an image and return a normalized tensor ready for inference.

    Args:
        img_path: Path to the image file.
        mode: "dino" (518px) or "standard" (224px).

    Returns:
        Tensor of shape (1, 3, H, W) with ImageNet normalization applied.
    """
    img = Image.open(img_path).convert('RGB')

    if mode == "dino":
        # DINOv2 ViT-Large expects 518x518 center-cropped input
        tr = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard CNN input: 224x224
        tr = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return tr(img).unsqueeze(0)
