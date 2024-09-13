# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire content of the directory to /app in the container
COPY . .

# Expose port 8000 for Flask API and 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# run the training pipeline to register the model
RUN python pipelines/training_pipeline.py

# run both Flask and Streamlit concurrently
CMD ["sh", "-c", "python pipelines/deployment_pipeline.py & streamlit run app.py"]
