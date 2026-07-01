import pandas as pd 
from pathlib import Path
import shutil
PROJECT_ROOT=Path(__file__).resolve().parent.parent #for project directory
RAW_BLIP_DIR=PROJECT_ROOT/'data'/'raw'/'blip'
RAW_IMAGES_DIR=RAW_BLIP_DIR/'images'
METADATA_PATH=RAW_BLIP_DIR/'metadata.csv'
PROCESSED_BLIP_DIR=PROJECT_ROOT/'data'/'processed'/'blip'
PROCESSED_METADATA_PATH=PROCESSED_BLIP_DIR/'metadata.csv'
PROCESSED_IMAGES_DIR=PROCESSED_BLIP_DIR/'images'
metadata=pd.read_csv(METADATA_PATH)
print('Metadata Loaded Successfully')
print(f'No. of Entries/Records: {len(metadata)}')
print()
print(metadata.head())
def clean_title(title):
    return ' '.join(str(title).split())
metadata['title']=metadata['title'].apply(clean_title)
print(metadata['title'].head())

PROCESSED_IMAGES_DIR.mkdir(parents=True,exist_ok=True)
for image_path in RAW_IMAGES_DIR.iterdir():
    if image_path.is_file():
        shutil.copy2(image_path,PROCESSED_IMAGES_DIR)
print(f'Copied {len(list(PROCESSED_IMAGES_DIR.iterdir()))} images.' )
metadata.to_csv(PROCESSED_METADATA_PATH,index=False) #no index bc pandas stores an index in its dataframe
print("Processed metadata saved successfully")