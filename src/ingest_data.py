import os
import zipfile
import pandas as pd

from abc import ABC, abstractmethod
from dotenv import load_dotenv
load_dotenv()

class DataIngestor(ABC):
    """Abstract class for data ingestor"""
    @abstractmethod
    def ingest(self, file_path: str)->pd.DataFrame:
        """Method to ingest data from provided file path"""
        pass


class ZipDataIngestor(DataIngestor):
    """concrete class for ZIP ingestion"""

    def ingest(self, file_path: str) -> pd.DataFrame:

        # Checking for valid zip file
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file path does not belong to a valid zip file.")

        # extracting data from zip file
        with zipfile.ZipFile(file_path, 'r') as zip_reader:
            zip_reader.extractall("extracted_data")

        # list the extracted file
        extracted_files = os.listdir(os.getenv("EXTRACTED_DIR"))
        # get the csv files from extracted folder
        csv_files = [file for file in extracted_files if file.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in extracted data.")
        elif len(csv_files) > 1:
            raise ValueError("More than one CSV files found. Specify the file to use.")
        else:
            # Reads in data from avb csv file
            csv_file_data = os.path.join(os.getenv("EXTRACTED_DIR"), csv_files[0])
            data = pd.read_csv(csv_file_data)
            return data


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Provide appropriate data ingestor based on provided file extension"""
        if file_extension == ".zip":
            return ZipDataIngestor()
        raise ValueError(f"No ingestor available for {file_extension} file extension.")


# if __name__ == "__main__":
#     # Path to zip file
#     file_path = "..."
#     # file extension
#     file_extension = os.path.splitext(file_path)[1]
#
#     # get ingestor
#     data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
#
#     # ingest and load data
#     data = data_ingestor.ingest(file_path)
#
#     print(data.head())




