# House Price Prediction
This project is designed to predict house prices using an end-to-end machine learning 
(ML) pipeline. It incorporates MLOps principles to streamline the development, deployment, 
and maintenance of the model.

The application consists of three main components:

1. **Model Training:** A training pipeline that builds and registers a model using MLflow.
2. **Model Deployment:** A Flask API that serves predictions from the trained model.
3. **Frontend:** A Streamlit app that allows users to input data and get predictions from the Flask API.


## Key Features
1. **End-to-End Machine Learning Pipeline**
   This project implements an end-to-end ML pipeline, starting from data ingestion to 
   model deployment.

2. **MLOps Framework**
   The project uses MLflow to manage the machine learning lifecycle:
   - **Experiment tracking:** Tracks various experiments and stores metrics, parameters, and artifacts.
   - **Model registration:** The trained model is logged and registered in MLflow, making it available for deployment. 
   - **Model versioning:** New versions of the model can be tracked and compared.
   
3. **Pipeline Architecture**
   - **Pipeline Pattern:** The project uses a pipeline architecture to process data through various stages (e.g., preprocessing, model training, evaluation).
   - **Dependency Injection:** Components like the inference pipeline, feature engineering, and preprocessing are injected into the training and deployment pipelines, enhancing flexibility and allowing for easier testing.

4. **Model Deployment via Flask API**
   The model is deployed using a Flask API, which exposes an endpoint `/predict` to serve real-time predictions. The API uses an inference pipeline to load the model from MLflow.

5. **Streamlit Web App for Predictions**
   The project includes a Streamlit app to provide a user-friendly interface for making predictions. Users can input data via the app, which then communicates with the Flask API to get the price predictions.

## Getting Started
1. **Install Dependencies**

   ```shell
   pip install -r requirements.txt
    ```
   
2. **Train the Model**

   Run the training_pipeline.py to train the model and register it in MLflow:
   ```shell
   python pipelines/training_pipeline.py
   ```
   
3. **Start Flask API**

   After training the model, start the Flask API to serve the model:
   ```shell
   python pipelines/deployment_pipeline.py
    ```
   This starts the API at http://localhost:8000


4. **Run the Streamlit App**
      
    Launch the Streamlit app:
   ```shell
   streamlit run app.py
    ```
   The Streamlit app will run at http://localhost:8501



## Docker Setup
1. **Build Docker Image**

   To build the Docker image, run the following command in the project root:
   ```shell
   docker build -t price-predictor-app .
    ```
2. **Run Docker Container**
   
    After the image is built, you can run the container:
   ```shell
    docker run -p 8000:8000 -p 8501:8501 price-predictor-app
    ```

---