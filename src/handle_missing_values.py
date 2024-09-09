import pandas as pd

from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MissingValuesHandlingStrategy(ABC):
    """
    Abstract base class for handling missing values strategies
    """

    @abstractmethod
    def handle(self, df: pd.DataFrame):
        """
        abstract method for handling missing values in the provided dataframe
        """
        pass


class DropMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    Concrete implementation of MissingValuesHandlingStrategy
    to handle missing values in the data by dropping them
    """

    def __init__(self, axis=0, threshold=None):
        """
        Initializes the strategy with specified parameters
        """
        self.axis = axis
        self.threshold = threshold

    def handle(self, df: pd.DataFrame):
        """
        Drops rows or columns based on specified axis
        and threshold from the provided dataframe
        """
        logging.info(f"Dropping missing values with axis={self.axis}.")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.threshold)
        logging.info("Missing Values dropped.")
        return df_cleaned


class FillMissingValuesStrategy(MissingValuesHandlingStrategy):
    """
    Concrete implementation of MissingValuesHandlingStrategy
    to fill missing values in the data
    """

    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with
        specified method and constant value to fill in
        missing values in the data
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame):
        """
        Fill in missing values in the data with
        specified method or fill value
        """
        logging.info(f"Filling missing values using the method: {self.method}")

        data = df.copy()

        if self.method=="mean":
            numeric_columns = data.select_dtypes(include="number").columns
            data[numeric_columns] = data[numeric_columns].fillna(
                data[numeric_columns].mean()
            )
        elif self.method=="median":
            numeric_columns = data.select_dtypes(include="number").columns
            data[numeric_columns] = data[numeric_columns].fillna(
                data[numeric_columns].median()
            )
        elif self.method=="mode":
            numeric_columns = data.select_dtypes(include="number").columns
            data[numeric_columns] = data[numeric_columns].fillna(
                data[numeric_columns].mode()
            )
        elif self.method=="constant":
            data = data.fillna(value=self.fill_value)
        else:
            logging.warning(f"Unknown method {self.method}. No missing values handled.")
            return data

        logging.info("Missing Values filled.")
        return data


class MissingValuesHandler:
    """
    Context class for using Missing Values handling strategies
    """

    def __init__(self, strategy: MissingValuesHandlingStrategy):
        """
        Initialized the handler with specified strategy
        """
        self.strategy = strategy


    def set_strategy(self, strategy: MissingValuesHandlingStrategy):
        """
        Setter function for updating strategy
        """
        logging.info("Switching handling missing value strategy.")
        self.strategy = strategy


    def handle_missing_values(self, df: pd.DataFrame):
        """
        Execute the current strategy to handle
        missing values in the provided data
        """
        logging.info("Handling missing values using current strategy.")
        return self.strategy.handle(df)