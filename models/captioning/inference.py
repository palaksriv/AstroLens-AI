from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
device=torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f'Using device: {device}')
processor=BlipProcessor.from_pretrained("models/fine_tuned_blip")
model=BlipForConditionalGeneration.from_pretrained("models/fine_tuned_blip")
model.to(device)
model.eval()
def generate_caption(image_path):
    image=Image.open(image_path).convert('RGB')
    inputs=processor(images=image,
                 return_tensors='pt')
    inputs={key : value.to(device) for key,value in inputs.items()}
    with torch.no_grad():
        output=model.generate(**inputs,
                              max_new_tokens=30,
                              num_beams=5,
                              early_stopping=True)
    caption=processor.decode(output[0],skip_special_tokens=True)
    return caption
if __name__=='__main__':
    image_path='sample_images/images (4).jpeg'
    caption=generate_caption(image_path)
    print('Generated Caption:')
    print(caption)
