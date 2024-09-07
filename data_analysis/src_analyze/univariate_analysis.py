import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_object_dtype, is_numeric_dtype

from abc import ABC, abstractmethod


class UnivariateAnalysisStrategy(ABC):
    """Abstract class to have a common interface for univariate analysis strategies"""

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """Performs univariate analysis on the specified feature of the dataframe"""
        pass


class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """
    Concrete implementation of UnivariateAnalysisStrategy
    for performing Univariate analysis on Numerical data features
    """

    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Performs univariate analysis of the numerical feature
        by plotting the histogram of it with KDE line
        """

        if not is_numeric_dtype(df[feature]):
            raise ValueError(f"The datatype of {feature} is not numerical.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(f"{feature}")
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """
    Concrete implementation of UnivariateAnalysisStrategy for
    performing Univariate analysis on Categorical data features
    """

    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Performs univariate analysis of the specified
        categorical feature by plotting frequency distribution
        using a bar plot
        """
        if not is_object_dtype(df[feature]):
            raise ValueError(f"The datatype of {feature} is not categorical")

        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, data=df)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(f"{feature}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateInspector:
    """Context class for using Univariate Analysis strategies"""

    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the univariate inspector with
        specified Univariate Analysis strategy
        """
        self.strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        """setter function for updating univariate analysis strategy"""
        self.strategy = strategy

    def perform_analysis(self, df:pd.DataFrame, feature: str) -> None:
        """performs univariate analysis associated with selected strategy"""
        self.strategy.analyze(df, feature)



