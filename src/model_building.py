from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import logging

from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelBuildingStrategy(ABC):
    """
    Abstract base class for Model building strategy.
    """
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to build and train ML Model.

        :param X_train: The feature set for Model training.
        :param y_train: The target/label set for Model training.

        :return: trained scikit learn Model instance.
        """
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    """
    Concrete class for implementation ModelBuildingStrategy
    using scikit Linear regression.
    """

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and train Linear Regression Model using scikit-learn.

        :param X_train: The feature set for Model training.
        :param y_train: The target/label set for Model training.

        :return: a scikit-learn pipeline with trained Linear Regression Model.
        """

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing the Linear Regression Model with scaling.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]
        )

        logging.info("Training Linear Regression Model.")
        pipeline.fit(X_train, y_train)

        logging.info("Model Training completed.")

        return pipeline



class DecisionTreeRegressionStrategy(ModelBuildingStrategy):
    """
    Concrete class for implementation ModelBuildingStrategy
    using scikit Linear regression.
    """

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and train Linear Regression Model using scikit-learn.

        :param X_train: The feature set for Model training.
        :param y_train: The target/label set for Model training.

        :return: a scikit-learn pipeline with trained Linear Regression Model.
        """

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing the Linear Regression Model with scaling.")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeRegressor())
            ]
        )

        logging.info("Training Linear Regression Model.")
        pipeline.fit(X_train, y_train)

        logging.info("Model Training completed.")

        return pipeline

class ModelBuilder:
    """
    Context class for Model Building.
    """

    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initialized the model builder with specified strategy.

        :param strategy: strategy method for building Model
        """
        self.strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Setter function for updating the Model building strategy.

        :param strategy: the new strategy method to be used
                         for building Model
        """
        logging.info("Switching Model building strategy.")
        self.strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Executes the Model building and Training using the
        current strategy.

        :param X_train: The feature set for Model training.
        :param y_train: The target/label set for Model training.

        :return: a trained scikit-learn Model instance.
        """

        logging.info("Building and training Model using the current strategy.")
        return self.strategy.build_and_train_model(X_train, y_train)