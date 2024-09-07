import pandas as pd
from abc import ABC, abstractmethod


class DataInspectionStrategy(ABC):
    """Interface for data inspect strategies"""
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """performs data inspection"""
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    """Concrete DataInspectionStrategy for Data types inspection"""
    def inspect(self, df: pd.DataFrame):
        """prints of the datatypes of the provided data columns"""

        print("Data types and Non-null counts:")
        return df.info()


class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    """Concrete DataInspectionStrategy for summary statistics inspection"""
    def inspect(self, df: pd.DataFrame):
        """prints of the summary statistics of the provided data"""

        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())

        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))


class DataInspector:
    """Context class for using Data Inspection strategies"""
    def __init__(self, strategy: DataInspectionStrategy):
        """initializes data inspection with specified Data Inspection Strategy"""
        self.strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        """setter function for updating data inspection strategy"""
        self.strategy = strategy

    def execute_inspection(self, df:pd.DataFrame) -> None:
        """performs inspection associated with selected strategy"""
        self.strategy.inspect(df)

