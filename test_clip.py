import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

image_path = "sample_images/images (3).jpeg"

labels = [
    "This is an image of the Carina Nebula, NGC 3372.",
    "This is an image of the Eagle Nebula, M16.",
]
image = Image.open(image_path).convert("RGB")

inputs = processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = outputs.logits_per_image.softmax(dim=1)[0]

for label, probability in sorted(
    zip(labels, probabilities),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label}: {probability.item():.4f}")