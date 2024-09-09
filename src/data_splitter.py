import pandas as pd

from abc import ABC, abstractmethod
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataSplittingStrategy(ABC):
    """
    Abstract base class for data splitting strategies
    """

    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        abstract method for splitting the provided dataframe
        into train and test splits
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Inplements the simple train test splits
    """
    def __init__(self, test_size: float = 0.2, random_state: int = 101):
        """
        Initializes the SimpleTrainTestSplitStrategy
        with specified parameters
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the provided data into train and tests set
        using sklearn framework
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")

        return X_train, X_test, y_train, y_test


class DataSplitter:
    """
    Context class for using data splitting strategies
    """

    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialized the splitter with specified strategy
        """
        self.strategy = strategy


    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Setter function for updating strategy
        """
        logging.info("Switching data splitting strategy.")
        self.strategy = strategy


    def split(self, df: pd.DataFrame, target_column: str):
        """
        Execute the current strategy to split
        the provided data into train-test split
        """
        logging.info("Splitting data using current strategy.")
        return self.strategy.split_data(df, target_column)