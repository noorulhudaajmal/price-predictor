from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluationStrategy(ABC):
    """
    Abstract base class for Model evaluation strategy.
    """
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin,
                       X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Abstract method to evaluate the Model.

        :param model: the trained model to evaluate.
        :param X_test: feature set for testing the model.
        :param y_test: target/label set for testing the model.

        :return: dictionary containing evaluation metrics.
        """
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    """
    Concrete class for implementation ModelEvaluationStrategy
    using scikit Linear evaluation metrics.
    """

    def evaluate_model(self, model: RegressorMixin,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> dict:
        """
        Evaluate the regression model using sckit-learn
        R-squared and Mean Square Error.

        :param model: the trained model to evaluate.
        :param X_test: feature set for testing the model.
        :param y_test: target/label set for testing the model.

        :return: dictionary containing R^2 and Mean Square  error values.
        """

        # if not isinstance(X_test, pd.DataFrame):
        #     raise TypeError("X_test must be a pandas DataFrame.")
        # if not isinstance(y_test, pd.Series):
        #     raise TypeError("y_test must be a pandas Series.")

        logging.info("Predicting using trained Model.")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics.")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        eval_metrics = {
            "Mean Squared Error": mse,
            "R-squared" : r2
        }

        logging.info(f"Model evaluation completed: {eval_metrics}")

        return eval_metrics


class ModelEvaluator:
    """
    Context class for Model Evaluation.
    """

    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initialized the model evaluator with specified strategy.

        :param strategy: strategy method for Model evaluation.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Setter function for updating the Model evaluation strategy.

        :param strategy: the new strategy method to be used
                         for Model evaluation
        """
        logging.info("Switching Model evaluation strategy.")
        self.strategy = strategy

    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the Model evaluation using the current strategy.

        :param model: the trained model to be evaluated.
        :param X_test: The feature set for Model testing.
        :param y_test: The target/label set for Model testing.

        :return: dictionary containing metrics(MSE & R^2).
        """

        logging.info("Evaluating the Model using the current strategy.")
        return self.strategy.evaluate_model(model, X_test, y_test)