"""GeoCLIP wrapper with country classification head for finetuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoclip import GeoCLIP


class GeoGuessrModel(nn.Module):
    """GeoCLIP-based model with a country classification head.

    Architecture:
    - Frozen CLIP ViT-L/14 image encoder (via GeoCLIP) → 768-dim
    - GeoCLIP's pretrained MLP (768 → 512), finetuned
    - Classification head: 512 → num_classes (countries)

    GeoCLIP already freezes CLIP internally. We finetune the MLP + classifier.
    """

    def __init__(
        self,
        num_classes: int = 55,
        freeze_clip: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained GeoCLIP — it freezes CLIP ViT-L/14 internally
        self.geoclip = GeoCLIP()

        # Optionally freeze the MLP too (only train classifier head)
        if freeze_clip:
            for param in self.geoclip.image_encoder.CLIP.parameters():
                param.requires_grad = False

        # Country classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def to(self, device):
        """Override to properly move GeoCLIP submodules."""
        self.geoclip = self.geoclip.to(device)
        return super().to(device)

    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim embeddings using GeoCLIP's image encoder (CLIP + MLP)."""
        # transformers >=5 returns BaseModelOutputWithPooling from get_image_features
        clip_out = self.geoclip.image_encoder.CLIP.get_image_features(pixel_values=images)
        if not isinstance(clip_out, torch.Tensor):
            clip_out = clip_out.pooler_output
        embeddings = self.geoclip.image_encoder.mlp(clip_out)
        return F.normalize(embeddings, dim=-1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass → logits and embeddings."""
        image_embeds = self.get_image_embeddings(images)
        logits = self.classifier(image_embeds)
        return {"logits": logits, "image_embeds": image_embeds}

    def compute_loss(
        self, output: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """CrossEntropy loss on country labels."""
        return F.cross_entropy(output["logits"], labels)

    def predict_topk(self, images: torch.Tensor, top_k: int = 5) -> dict:
        """Predict top-k countries for a batch of images."""
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
            probs = F.softmax(output["logits"], dim=-1)
            top_probs, top_labels = probs.topk(top_k, dim=-1)
        return {"top_k_labels": top_labels, "top_k_probs": top_probs}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
