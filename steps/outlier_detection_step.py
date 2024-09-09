import pandas as pd
from src.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection
)
import mlflow

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def outlier_detection_step(df: pd.DataFrame, strategy: str, column_name: str):
    logging.info(f"Starting outlier detection step with DataFrame column {column_name}")

    if df is None:
        logging.error("Received a Non-Type DataFrame.")
        raise ValueError("Input must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)}")
        raise ValueError("Input must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not present exist in DataFrame.")
        raise ValueError(f"Column '{column_name}' does not present exist in DataFrame.")

    df_numeric = df.select_dtypes(include=[int, float])

    if strategy == "zscore":
        outlier_detector = OutlierDetector(ZScoreOutlierDetection())
    elif strategy == "iqr":
        outlier_detector = OutlierDetector(IQROutlierDetection())
    else:
        raise ValueError(f"Unsupported outlier detection strategy: {strategy}")

    # detected_outliers = outlier_detector.detect_outliers(df)
    cleaned_df = outlier_detector.handle_outliers(df_numeric)
    return cleaned_df


