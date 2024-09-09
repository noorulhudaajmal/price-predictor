from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def data_splitter_step(
        df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets using the DataSplitter
    """
    logging.info("Starting data splitting step...")

    # Initialize the DataSplitter with the specified strategy
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())

    # Perform the split
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    logging.info(f"Data split completed. Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Log the split details to MLflow
    mlflow.log_param("target_column", target_column)
    mlflow.log_param("X_train_shape", X_train.shape)
    mlflow.log_param("X_test_shape", X_test.shape)
    mlflow.log_param("y_train_shape", y_train.shape)
    mlflow.log_param("y_test_shape", y_test.shape)

    return X_train, X_test, y_train, y_test
