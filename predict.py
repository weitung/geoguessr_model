"""Single-image inference for GeoGuessr model."""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from data.dataset import CountryMapper
from models.geoclip_finetune import GeoGuessrModel


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model and country mapper."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    mapper_path = os.path.join(cfg["checkpoint"]["save_dir"], "country_mapper.json")
    mapper = CountryMapper.load(mapper_path)

    model = GeoGuessrModel(
        num_classes=mapper.num_classes,
        freeze_clip=cfg["model"]["freeze_clip"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, mapper, cfg["data"]["image_size"]


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess a single image."""
    transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(image_path: str, checkpoint_path: str, top_k: int = 5) -> list[dict]:
    """Predict country from a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapper, image_size = load_model(checkpoint_path, device)

    image = preprocess_image(image_path, image_size).to(device)
    result = model.predict_topk(image, top_k=top_k)

    predictions = []
    for label, prob in zip(
        result["top_k_labels"][0].cpu().numpy(),
        result["top_k_probs"][0].cpu().numpy(),
    ):
        lat, lon = mapper.get_centroid(int(label))
        predictions.append(
            {
                "country": mapper.decode(int(label)),
                "latitude": lat,
                "longitude": lon,
                "confidence": float(prob),
            }
        )
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict country from image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt", help="Model checkpoint"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions")
    args = parser.parse_args()

    print(f"Predicting location for: {args.image}")
    predictions = predict(args.image, args.checkpoint, args.top_k)

    print(f"\nTop {args.top_k} predictions:")
    for i, pred in enumerate(predictions, 1):
        print(
            f"  {i}. {pred['country']:20s} | "
            f"({pred['latitude']:.1f}, {pred['longitude']:.1f}) | "
            f"confidence: {pred['confidence']:.4f}"
        )


if __name__ == "__main__":
    main()
