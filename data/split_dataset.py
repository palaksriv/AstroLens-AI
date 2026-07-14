import pandas as pd 
from sklearn.model_selection import train_test_split
metadata=pd.read_csv('data/processed/blip/metadata.csv')
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
train_df.to_csv('data/processed/blip/train.csv',index=False)
val_df.to_csv('data/processed/blip/val.csv',index=False)
test_df.to_csv('data/processed/blip/test.csv',index=False)
print('Dataset split completed')

