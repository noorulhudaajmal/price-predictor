import mlflow
import os
from dotenv import load_dotenv
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.handle_missing_values_step import handle_missing_values
from steps.feature_engineering_step import apply_feature_engineering
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()

def training_pipeline():
    """
    End-to-end data training pipeline using MLflow
    """

    # Set the experiment name
    mlflow.set_experiment("House Price Prediction")

    # Start an MLflow run
    with mlflow.start_run(run_name="linear_regression") as run:

        run_id = run.info.run_id

        mlflow.set_tag("model_type", "lr")
        mlflow.set_tag("author", "Huda")
        mlflow.set_tag("version", "1.0")
        mlflow.set_tags({
            "data_version": "v1.0",
            "project": "house_price_prediction",
            "purpose": "train model with basic preprocessing and Linear Regression"
        })
        # Log experiment description
        mlflow.set_tag("description", "An experiment to predict house prices using linear regression and log strategy for feature engineering.")

        # 1. Data Ingestion
        raw_data = data_ingestion_step(file_path=os.getenv('FILE_PATH'))
        mlflow.log_param("file_path", os.getenv('FILE_PATH'))

        # 2. Handling Missing Values step
        cleaned_data = handle_missing_values(df=raw_data)

        # 3. Feature Engineering
        transformed_data = apply_feature_engineering(df=cleaned_data,
                                                     strategy="log",
                                                     features=["Gr Liv Area", "SalePrice"])
        mlflow.log_param("feature_engineering_strategy", "log")

        # 4. Data splitting step
        X_train, X_test, y_train, y_test = data_splitter_step(df=transformed_data, target_column="SalePrice")

        # Log datasets and their shapes
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("X_test_shape", X_test.shape)

        # 5. Model building step
        trained_model = model_building_step(X_train, y_train)
        logging.info("Model training and logging completed.")

        # 6. Model Evaluation step
        evaluation_metrics = model_evaluation_step(trained_model, X_test, y_test)
        logging.info("Model evaluation completed.")

        # Logging evaluation metrics
        mlflow.log_metric("Mean Squared Error",
                          evaluation_metrics.get("Mean Squared Error"),
                          run_id=run_id)
        mlflow.log_metric("R-squared",
                          evaluation_metrics.get("R-squared"),
                          run_id=run_id)
        logging.info("Model metrics logged.")

        # Logging model pipeline
        mlflow.sklearn.log_model(
            sk_model=trained_model,
            artifact_path="model",
            registered_model_name="house_price_prediction_lr",
            input_example=X_train.head(1)
        )

        # Register the model in the Model Registry
        # mlflow.register_model(f"runs:/{run_id}/model", "house_price_prediction_lr")
        logging.info(f"Model registered in MLflow Model Registry with run ID {run_id}.")

        return trained_model

if __name__ == "__main__":
    model = training_pipeline()
