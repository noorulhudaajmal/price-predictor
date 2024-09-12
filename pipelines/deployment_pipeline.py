import pandas as pd
import logging
from flask import Flask, request, jsonify
from inference_pipeline import InferencePipeline


# Initialize Flask app
app = Flask(__name__)

# inference pipeline instance
inference_pipeline = None


# API route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the incoming JSON data
        data = request.get_json()

        # 2. Convert JSON data to pandas DataFrame
        batch_data = pd.DataFrame(data)

        # 3. Make predictions using the inference pipeline
        predictions = inference_pipeline.run_inference(batch_data)

        # 4. Return predictions as a JSON response
        response = {"predictions": predictions.tolist()}
        return jsonify(response)

    except Exception as e:
        # error log
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


def load_app():
    # Model loading
    model_uri = "models:/house_price_prediction_lr/4"

    # Initialize the InferencePipeline
    global inference_pipeline
    inference_pipeline = InferencePipeline(model_uri)

    # Start the Flask app on port 8000
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    load_app()
