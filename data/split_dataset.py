import pandas as pd 
from pathlib import Path
from sklearn.model_selection import train_test_split
BLIP_DIR=Path(__file__).resolve().parent/'processed'/'blip'
metadata=pd.read_csv(BLIP_DIR/'metadata.csv')
train_df,temp_df=train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
val_df,test_df=train_test_split(temp_df,
                                test_size=0.5,
                                random_state=42,
                                shuffle=True)
train_df.to_csv(BLIP_DIR/'train.csv',index=False)
val_df.to_csv(BLIP_DIR/'val.csv',index=False)
test_df.to_csv(BLIP_DIR/'test.csv',index=False)
print('Dataset split completed')

