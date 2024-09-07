from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values
from zenml import Model, pipeline, step

import os
from dotenv import load_dotenv


load_dotenv()

@pipeline(
    model=Model(
        name="price_predictor"
    )
)
def preprocessing_pipeline():
    """
    End to end data preprocessing pipeline
    :return:
    """

    # Data Ingestion
    raw_data = data_ingestion_step(file_path=os.getenv('FILE_PATH'))

    # Handling Missing Values step
    cleaned_data = handle_missing_values(df=raw_data)

    return cleaned_data


if __name__ == "__main__":
    run = preprocessing_pipeline()

