import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from abc import ABC, abstractmethod


class BivariateAnalysisStrategy(ABC):
    """Abstract class to have a common interface for bivariate analysis strategies"""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Performs bivariate analysis to analyze the relationship
        between the specified features of the dataframe
        """
        pass


class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """
    Concrete implementation of BivariateAnalysisStrategy
    for analyzing relationship between two numerical features
    """

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Performs bivariate analysis of the specified
        numerical features using a scatter plot
        """

        if not is_numeric_dtype(df[feature1]) and not is_numeric_dtype(df[feature2]):
            raise ValueError(f"The datatype of provided column(s) is not numerical.")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()


class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    """
    Concrete implementation of BivariateAnalysisStrategy for
    analyzing relationship between categorical and numerical features
    """

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Performs bivariate analysis of the specified features using
        a boxplot to plot numerical distribution of feature2 against
        the categories of feature1
        """

        # if not is_object_dtype(df[feature1]):
        #     raise ValueError(f"The datatype of {feature1} is not categorical")
        #
        # if not is_numeric_dtype(df[feature2]):
        #     raise ValueError(f"The datatype of {feature2} is not numerical")

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.xticks(rotation=45)
        plt.show()


class BivariateInspector:
    """Context class for using Bivariate Analysis strategies"""

    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the bivariate inspector with
        specified Bivariate Analysis strategy
        """
        self.strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy) -> None:
        """setter function for updating bivariate analysis strategy"""
        self.strategy = strategy

    def perform_analysis(self, df:pd.DataFrame, feature_1: str, feature_2: str) -> None:
        """performs bivariate analysis associated with selected strategy"""
        self.strategy.analyze(df, feature_1, feature_2)



