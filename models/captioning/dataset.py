import pandas as pd 
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image 
class AstroLensDataset(Dataset): #to be able to use dataloader 
    def __init__(self,data_dir=None,csv_file='metadata.csv'):
        if data_dir is None:
            project_root=Path(__file__).resolve().parent.parent.parent
            self.processed_blip_dir=(project_root/'data'/'processed'/'blip')
        else:
            self.processed_blip_dir=Path(data_dir)
        self.images_dir=self.processed_blip_dir/'images'
        self.metadata_path=self.processed_blip_dir/csv_file
        self.metadata=pd.read_csv(self.metadata_path)
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self,index):
        sample=self.metadata.iloc[index]
        image_name=sample['file']
        caption=sample['title']
        image_path=self.images_dir/image_name
        image=Image.open(image_path).convert('RGB')
        return image,caption
#testing
if __name__=='__main__':
    dataset=AstroLensDataset(data_dir=r"C:\Users\HomePC1\Documents\prep\AstroLens-AI\data\processed\blip")
    print(dataset.metadata.head())
    print(f'Dataset size: {len(dataset)}')
    image,caption=dataset[0]
    print(type(image))
    print(caption)
    