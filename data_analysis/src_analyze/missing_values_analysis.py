import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abc import ABC, abstractmethod


class MissingValuesAnalysisTemplate(ABC):
    """Abstract base class for missing values analysis"""

    def analyze(self, df: pd.DataFrame):
        """Performs missing value analysis in a certain order"""
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """Identifies missing values in the dataframe"""
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """Visualizes the missing values in the dataframe"""
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    """Concrete class for performing simply the missing value identification and visualization"""

    def identify_missing_values(self, df: pd.DataFrame):
        """prints out the count of missing value for each column"""
        print("Missing Values count:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        """creates a heatmap to visualize the missing values in the data"""
        print("\n\nVisualizing the missing values...")
        plt.figure(figsize=(20,10))
        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()



