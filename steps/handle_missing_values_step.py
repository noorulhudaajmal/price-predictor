import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValuesHandler
)
import mlflow

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def handle_missing_values(df: pd.DataFrame, strategy:str = "mean"):
    """
    Initiate the respective handler based on strategy
    specified and use it to handle missing values in data

    :param df: dataframe with missing values
    :param strategy: name of strategy
    :return: cleaned dataframe with imputed/cleaned missing value rows
    """
    if strategy == "drop":
        handler = MissingValuesHandler(DropMissingValuesStrategy())
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValuesHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df

