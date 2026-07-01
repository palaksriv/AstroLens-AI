import pandas as pd
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent #goes to project root and then finds the folders
PROCESSED_METADATA_PATH=PROJECT_ROOT/'data'/'processed'/'blip'/'metadata.csv'
PROCESSED_IMAGES_DIR=PROJECT_ROOT/'data'/'processed'/'blip'/'images'
def main():
    metadata=pd.read_csv(PROCESSED_METADATA_PATH)
    metadata_files=set(metadata['file'])
    print('Processed Data Validation:')
    print(f'No. of Records in Metadata: {len(metadata)}')
    images=list(image for image in PROCESSED_IMAGES_DIR.iterdir() if image.is_file())
    image_names={image.name for image in images}
    print(f'No. of Images: {len(image_names)}')
    missing_images = metadata_files - image_names
    extra_images = image_names - metadata_files
    print(f'Missing Images: {len(missing_images)}')
    print(f'Extra Images: {len(extra_images)}')
if __name__=='__main__':
    main()