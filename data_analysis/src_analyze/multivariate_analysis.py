import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod

class MultivariateAnalysisTemplate(ABC):
    """
    Abstract class for providing a template
    to perform multivariate analysis on
    the features of the data
    """
    def analyze(self, df: pd.DataFrame):
        """
        Performs comprehensive multivariate analysis
        of the provided data
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)


    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates the heatmap of the correlation
        matrix of the data
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate the pairplot to show the relationship
        between the features of the data
        """
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    """
    Concrete implementation of the MultivariateAnalysisTemplate that
    implementing the methods to generate the heatmap and pairplot
    """

    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate the heatmap to show the correlation between the
        features of that specified data
        """

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates the pairplot for the specified data
        """

        sns.pairplot(df)
        plt.suptitle("Pair Plot", y=1.02)
        plt.show()