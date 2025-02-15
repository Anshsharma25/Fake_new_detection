import pandas as pd
from src.config.logger import logger

def load_data():
    # Use processed dataset instead of raw datasets
    dataset_path = "src/data/processed_data.csv"  
    df = pd.read_csv(dataset_path)

    if df.empty:
        logger.error("Processed dataset is empty.")
        return None, dataset_path

    logger.info(f"Dataset loaded from: {dataset_path} with {len(df)} samples.")
    return df, dataset_path
