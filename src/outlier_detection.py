import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame):
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold=threshold

    def detect_outliers(self, df: pd.DataFrame):
        logging.info("Detecting outliers with z-score method.")
        z_scores = np.abs((df - df.mean())/df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with z-score threshold: {self.threshold}")
        return outliers


class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame):
        logging.info("Detecting outliers using IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3-Q1
        outliers = (df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))
        logging.info(f"Outliers detected using the IQR method.")
        return outliers


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame):
        logging.info("Executing the detect outliers strategy.")
        return self.strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method: str = "remove"):
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing the outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping the outliers from the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.info(f"Unknown method: {self.strategy}. Outliers are not handled.")
        logging.info("Outliers handling completed.")

        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualize outliers for features: {features}.")
        for feature in features:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=feature, data=df)
            plt.title(f"Box plot of {feature}")
            plt.show()

