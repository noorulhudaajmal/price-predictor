import mlflow
import os
from dotenv import load_dotenv
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.handle_missing_values_step import handle_missing_values
from steps.feature_engineering_step import apply_feature_engineering

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

def preprocessing_pipeline():
    """
    End-to-end data preprocessing pipeline using MLflow
    """
    # Start an MLflow run
    with mlflow.start_run(run_name="preprocessing_pipeline"):

        # Data Ingestion
        raw_data = data_ingestion_step(file_path=os.getenv('FILE_PATH'))
        mlflow.log_param("file_path", os.getenv('FILE_PATH'))

        # Handling Missing Values step
        cleaned_data = handle_missing_values(df=raw_data)

        # Feature Engineering
        transformed_data = apply_feature_engineering(df=cleaned_data,
                                                     strategy="log",
                                                     features=["Gr Liv Area", "SalePrice"])
        mlflow.log_param("feature_engineering_strategy", "log")

        # Data splitting step
        X_train, X_test, y_train, y_test = data_splitter_step(df=transformed_data, target_column="SalePrice")

        # Log datasets and their shapes
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("X_test_shape", X_test.shape)

        # mlflow.log_artifact(<file_path>, "processed_data")

        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocessing_pipeline()
