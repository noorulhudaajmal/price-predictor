import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    StandardScaling,
    MinMaxScaling,
    OneHotEncoding
)

import mlflow

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_feature_engineering(df: pd.DataFrame, strategy: str, features: list) -> pd.DataFrame:
    logging.info(f"Starting feature engineering with strategy: {strategy} on features: {features}")

    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)

    logging.info("Feature engineering completed.")

    return transformed_df
