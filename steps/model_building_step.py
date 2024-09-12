import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import logging

from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def model_building_step(
        X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """
    Builds and trains Regression Model using scikit-learn.

    :param X_train: feature set for Model training.
    :param y_train: target/label set for Model training.

    :return: trained scikit-learn Pipeline.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_columns = X_train.select_dtypes(exclude=['object', 'category']).columns

    logging.info(f"Categorical columns: {categorical_columns}")
    logging.info(f"Numerical columns: {numerical_columns}")

    # Creating the preprocessing pipelines
    logging.info("Initializing the pre-processing pipeline for column transformation.")
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_columns),
            ("categorical", categorical_transformer, categorical_columns)
        ]
    )

    # Creating the training pipeline
    logging.info("Initializing the Model training pipeline with column transformation preprocessing.")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    logging.info("Starting model training pipeline with MLflow...")


    mlflow.sklearn.autolog(log_models=False)  # auto-logging for sklearn for logging training metrics

    try:

        logging.info("Building and training Regression Model.")
        pipeline.fit(X_train, y_train)  # training the pipeline
        logging.info("Model training completed.")

        #Logging the expected column names
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_columns])

        expected_columns = numerical_columns.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_columns)
        )

        processed_data = pipeline.named_steps["preprocessor"].fit_transform(X_train)
        logging.info(f"Processed_ train data shape: {processed_data.shape}")

        logging.info(f"Model expects the following columns: {expected_columns}")
        # mlflow.log_param("expected_columns", expected_columns)


    except Exception as e:
        logging.error(f"Error occurred while training the Model: {e}")
        raise e

    logging.info("Model training pipeline with MLflow completed.")

    return pipeline