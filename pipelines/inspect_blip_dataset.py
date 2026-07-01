import os 
import pandas as pd
from pathlib import Path
METADATA_PATH="data/raw/blip/metadata.csv"
IMAGES_FOLDER=Path("data/raw/blip/images")
def main():
    print('AstroLens AI- BLIP dataset inspection')
    metadata=pd.read_csv(METADATA_PATH)
    print(f'Metadata Entries:{len(metadata)}')
    images=list(IMAGES_FOLDER.iterdir())
    print(f'Images Count:{len(images)}')
    metadata_files=set(metadata['file'])
    image_filenames={image.name for image in images}
    missing_images=metadata_files-image_filenames
    print(f'No. of missing images:{len(missing_images)}')
    missing_titles=metadata['title'].isna().sum()
    print(f'Missing titles: {missing_titles}')
if __name__=="__main__":
    main()   #only executes main() if this file is run directly