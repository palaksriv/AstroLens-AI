import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor #processor only
from dataset import AstralVisionDataset
from transformers import BlipForConditionalGeneration #neural net

BATCH_SIZE=4
LEARNING_RATE=1e-5
NUM_EPOCHS=5
PRINT_EVERY=25

CHECKPOINT_DIR=Path('checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset=AstralVisionDataset(csv_file='train.csv')
val_dataset=AstralVisionDataset(csv_file='val.csv')

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

train_dataloader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()

    print(f'\n Epoch {epoch+1}/{NUM_EPOCHS}')
    total_loss=0
    for batch_idx,batch in enumerate(train_dataloader):
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
            print(f'Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}')
    
    average_loss=total_loss/len(train_dataloader)
    print(f"Average training loss: {average_loss:.4f}")

    model.eval()
    val_loss=0
    with torch.no_grad():
        for batch in val_dataloader:
            batch={
                key:value.to(device) for key, value in batch.items()
            }
            outputs=model(**batch)
            val_loss+=outputs.loss.item()
        avg_val_loss=val_loss/len(val_dataloader)
        print(f'Validation loss: {avg_val_loss:.4f}')
        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            model.save_pretrained('models/fine_tuned_blip')
            processor.save_pretrained('models/fine_tuned_blip')
            print('Best model saved!')

