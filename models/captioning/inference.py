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
image=Image.open('sample_images/images (3).jpeg').convert('RGB')
inputs=processor(images=image,
                 return_tensors='pt')
inputs={key : value.to(device) for key,value in inputs.items()}
with torch.no_grad():
    output=model.generate(**inputs)
caption=processor.decode(output[0],skip_special_tokens=True)
print('Generated caption:')
print(caption)