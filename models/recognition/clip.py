from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading CLIP on: {device}")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


ASTRONOMY_LABELS = [
    "Thor's Helmet Nebula",
    "Horsehead Nebula",
    "Carina Nebula",
    "Eagle Nebula",
    "Orion Nebula",
    "Great Nebula in Cepheus",
    "Andromeda Galaxy",
    "Whirlpool Galaxy",
    "Crab Nebula",
    "Helix Nebula",
    "Ring Nebula",
    "Lagoon Nebula",
    "Trifid Nebula",
    "Tarantula Nebula",
    "Sombrero Galaxy",
    "Triangulum Galaxy",
    "Pleiades",
    "Milky Way Galaxy",
    "Jupiter",
    "Saturn",
    "Mars",
    "Moon",
    "Sun",
]


def recognize_object(image_path, top_k=5):
    image = Image.open(image_path).convert("RGB")

    prompts = [
        f"an astronomical photograph of {label}"
        for label in ASTRONOMY_LABELS
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    ranked = sorted(
        zip(ASTRONOMY_LABELS, probabilities.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "label": label,
            "confidence": round(score, 4),
        }
        for label, score in ranked[:top_k]
    ]
