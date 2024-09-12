import numpy as np
import pandas as pd
import logging

from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame):
        pass


class LogTransformation(FeatureEngineeringStrategy):

    def __init__(self, features: list):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame):
        logging.info(f"Applying log transformation to features: {self.features}")

        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )

        logging.info("Log transformation completed.")

        return df_transformed


class StandardScaling(FeatureEngineeringStrategy):

    def __init__(self, features: list):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame):
        logging.info(f"Applying standard scaling to features: {self.features}")

        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])

        logging.info("Standard scaling completed.")

        return df_transformed


class MinMaxScaling(FeatureEngineeringStrategy):

    def __init__(self, features: list):
        self.features = features
        self.scaler = MinMaxScaler()

    def apply_transformation(self, df: pd.DataFrame):
        logging.info(f"Applying min-max scaling to features: {self.features}")

        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])

        logging.info("Min-Max scaling completed.")

        return df_transformed


class OneHotEncoding(FeatureEngineeringStrategy):

    def __init__(self, features: list):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')

    def apply_transformation(self, df: pd.DataFrame):
        logging.info(f"Applying one-hot encoding to features: {self.features}")

        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df_transformed[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )

        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

        logging.info("One-hot encoding completed.")

        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy, noise_features: list = None):
        self.strategy = strategy
        self.noise_features = noise_features

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy.")
        self.strategy = strategy

    def drop_noise_features(self, df: pd.DataFrame):
        logging.info("Dropping Noise Features...")
        df = df.drop(columns=self.noise_features)
        return df

    def apply_feature_engineering(self, df: pd.DataFrame):
        logging.info("Applying feature engineering strategy.")
        if self.noise_features:
            df = self.drop_noise_features(df)
        return self.strategy.apply_transformation(df)


