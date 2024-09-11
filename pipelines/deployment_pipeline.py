import mlflow
import pandas as pd
import json
from steps.dynamic_importer import dynamic_importer
import logging
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_uri):
    """
    Loads the deployed model using MLflow model registry.
    """
    logging.info(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("Model loaded successfully")
    return model

def run_inference(model, batch_data):
    """
    Runs inference on the batch data using the loaded model.
    """
    logging.info("Running inference on batch data")
    # preprocessor = model.named_steps["preprocessor"]
    # batch_data_processed = preprocessor.transform(batch_data)
    # estimator = model.named_steps["model"]
    logging.info(f"model is: {model}")
    preds = model.predict(batch_data)
    logging.info(f"Predictions: {preds}")
    return preds

def deployment_pipeline():
    """
    Pipeline that imports data dynamically, loads the
    deployed model, and runs predictions in real time.
    """

    # Step 1: Fetch dynamic data (equivalent to a real-world API call)
    logging.info("Fetching dynamic batch data")
    json_data = dynamic_importer()
    batch_data = pd.read_json(json_data, orient="split")

    # Step 2: Load the deployed model from MLflow
    model_uri = "models:/house_price_prediction_lr/8"
    model = load_model(model_uri)

    # Step 3: Run predictions on the fetched batch data
    predictions = run_inference(model, batch_data)

    return predictions

if __name__ == "__main__":
    predictions = deployment_pipeline()
    print("Predictions:", predictions)
