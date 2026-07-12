import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor #processor only
from dataset import AstroLensDataset
from transformers import BlipForConditionalGeneration #neural net

BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_EPOCHS=5
PRINT_EVERY=25

CHECKPOINT_DIR=Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

dataset=AstroLensDataset()

processor=BlipProcessor.from_pretrained(
    'Salesforce/blip-image-captioning-base')
model=BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
model.to(device)
optimizer=AdamW(model.parameters(),lr=LEARNING_RATE)

def collate_fn(batch):
    images,captions=zip(*batch) #batch is [('img1','jupiter')] so *batch unpacks the list and zip groups together elements w the same position as tuples
    inputs=processor(images=images,text=captions,padding=True,return_tensors='pt')
    labels=inputs['input_ids'].clone()
    labels[labels==processor.tokenizer.pad_token_id]=-100
    inputs['labels']=labels
    return inputs

dataloader=DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

model.train()
for epoch in range(NUM_EPOCHS):
    print(f'\n Epoch {epoch+1}/{NUM_EPOCHS}')
    total_loss=0
    for batch_idx,batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch={
                key:value.to(device) for key, value in batch.items()
              }
        outputs=model(**batch)
        loss=outputs.loss
        loss.backward()
        optimizer.step()
    
        total_loss+=loss.item()
        if batch_idx % 25==0:
            print(f'Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}')
    
average_loss=total_loss/len(dataloader)
print("Average training loss:",average_loss)
checkpoint_path=CHECKPOINT_DIR/f'blip_poch_{epoch+1}.pth'
torch.save(
    {
        'epoch':epoch+1,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':average_loss,
    },
    checkpoint_path
)
print(f'Checkpoint saved to {checkpoint_path}')