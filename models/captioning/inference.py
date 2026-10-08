from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "fine_tuned_blip"

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f'Using device: {device}')

if not (MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        f"Fine-tuned BLIP checkpoint not found at: {MODEL_DIR}\n"
        "Train it (models/captioning/train.py, e.g. in Colab) and extract "
        "fine_tuned_blip.zip so that models/fine_tuned_blip/config.json exists."
    )

processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image,
                       return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs,
                                max_new_tokens=30,
                                num_beams=5,
                                early_stopping=True)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


if __name__ == '__main__':
    image_path = PROJECT_ROOT / 'sample_images' / 'images (1).jpeg'
    caption = generate_caption(image_path)
    print('Generated Caption:')
    print(caption)
