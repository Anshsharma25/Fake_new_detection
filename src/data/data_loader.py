import pandas as pd 
from src.config.logger import logger

def load_data():
    # Load data from CSV
    fake_df = pd.read_csv('src\data\Fakenews.csv')
    true_df = pd.read_csv('src\data\Truenews.csv')
    
    # Add labels: Fake News → 1, Real News → 0
    
    fake_df['label'] = 1
    true_df['label'] = 0
    
    # Concatenate the dataframes
    df = pd.concat([fake_df, true_df],ignore_index=True)
    
    # Shuffle the data
    df = df.sample(frac=1 ,random_state=42).reset_index(drop=True)
    
    # Remove missing values
    df = df.dropna()
    logger.info(f"Dataset loaded with {len(df)} samples.")
    return df