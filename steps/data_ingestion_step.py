import os
from src.ingest_data import (
    DataIngestorFactory
)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def data_ingestion_step(file_path: str):
    """
    Initiate the respective data ingestor based on the
    extension of the file from the provided file path
    and reads in data using the ingestor

    :param file_path: path to the data file
    :return: dataframe containing data from file
    """

    logging.info(f"Loading data from path: {file_path}")

    file_extension = os.path.splitext(file_path)[1]
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)

    logging.info("Data loaded")

    return data

