import mlflow
import pandas as pd
from steps.dynamic_importer import dynamic_importer
import logging
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InferencePipeline:
    def __init__(self, model_uri):
        """
        Initialize the inference pipeline by loading the model.

        :param model_uri: The URI of the model to load.
        """
        self.model_uri = model_uri
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the deployed model using MLflow model registry.

        :return: Loaded model from model registry.
        """
        logging.info(f"Loading model from {self.model_uri}")
        model = mlflow.sklearn.load_model(self.model_uri)
        logging.info("Model loaded successfully")
        return model

    def run_inference(self, batch_data):
        """
        Runs inference on the batch data using the loaded model.

        :param batch_data: The data to run inference on.

        :return: Model inference.
        """
        logging.info("Running inference on batch data")
        preds = self.model.predict(batch_data)
        logging.info(f"Predictions: {preds}")
        return preds

if __name__ == "__main__":
    # for testing
    json_data = dynamic_importer()  # to get test data
    test_data = pd.read_json(json_data, orient="split")
    model_uri = "models:/house_price_prediction_lr/4"

    # Create an instance of the InferencePipeline class
    pipeline = InferencePipeline(model_uri)

    # Run inference
    predictions = pipeline.run_inference(test_data)
    print("Predictions:", predictions)
