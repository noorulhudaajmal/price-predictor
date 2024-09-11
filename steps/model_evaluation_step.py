import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def model_evaluation_step(trained_model: Pipeline,
                          X_test: pd.DataFrame,
                          y_test: pd.Series) -> dict:
    """
    Evaluate the regression model using sckit-learn
    R-squared and Mean Square Error.

    :param trained_model: the trained model pipeline.
    :param X_test: feature set.
    :param y_test: target/label set.

    :return: dictionary containing evaluation metrics.
    """

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying preprocessing to features test-set.")

    preprocessor = trained_model.named_steps["preprocessor"]
    # transforming the X_test data
    X_test_processed = preprocessor.transform(X_test)

    # # Check if the processed data is in a sparse matrix format
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()  # Convert sparse matrix to dense array

    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must e returned as python dict.")

    return evaluation_metrics

