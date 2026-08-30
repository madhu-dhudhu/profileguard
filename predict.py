"""
AIGC Image Detector - Inference Script

Takes a directory of images and outputs a JSON file with a confidence score
for each image, indicating the likelihood that it is AI-generated.

Usage:
    python predict.py --image_dir path/to/images --model_path baseline_model.pth --output predictions.json

Output format (predictions.json):
    [
        {"image_path": "path/to/images/img1.jpg", "pred": 0.87},
        {"image_path": "path/to/images/img2.jpg", "pred": 0.12},
        ...
    ]
    where "pred" is the probability the image is AI-generated (0.0 = real, 1.0 = fake).
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


# Must match the class order used during training.
# If your training used ImageFolder, check dataset.classes to confirm this order.
# Example: if classes == ['fake', 'real'], then FAKE_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 0  # <-- CONFIRM THIS MATCHES YOUR TRAINING SETUP

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Standard ImageNet normalization, matching what the model was trained with
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, device):
    """Load the trained EfficientNet-B0 model from a checkpoint file."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def find_images(image_dir):
    """Recursively find all supported image files in a directory."""
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                image_paths.append(os.path.join(root, fname))
    return sorted(image_paths)


def predict_single_image(model, image_path, device):
    """Run inference on a single image, returning a fake-probability score."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  Warning: could not open {image_path} ({e}), skipping.", file=sys.stderr)
        return None

    img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        fake_prob = probs[0, FAKE_CLASS_INDEX].item()

    return fake_prob


def main():
    parser = argparse.ArgumentParser(description="AIGC Image Detector - batch inference script")
    parser.add_argument('--image_dir', type=str, required=True,
                         help='Path to a directory of images to classify')
    parser.add_argument('--model_path', type=str, default='baseline_model.pth',
                         help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default='predictions.json',
                         help='Path to write the output JSON file')
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: image directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.model_path):
        print(f"Error: model checkpoint not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(args.model_path, device)

    print(f"Scanning {args.image_dir} for images...")
    image_paths = find_images(args.image_dir)
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        print("No supported images found. Exiting.", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, image_path in enumerate(image_paths, 1):
        fake_prob = predict_single_image(model, image_path, device)
        if fake_prob is not None:
            results.append({"image_path": image_path, "pred": round(fake_prob, 4)})

        if i % 50 == 0 or i == len(image_paths):
            print(f"  Processed {i}/{len(image_paths)} images...")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Wrote {len(results)} predictions to {args.output}")


if __name__ == '__main__':
    main()
