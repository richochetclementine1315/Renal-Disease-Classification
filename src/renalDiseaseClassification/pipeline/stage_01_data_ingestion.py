from src.renalDiseaseClassification.config.configuration import ConfigurationManager
from src.renalDiseaseClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.renalDiseaseClassification.entity.config_entity import DataIngestionConfig
from src.renalDiseaseClassification.components.data_ingestion import DataIngestion
from src.renalDiseaseClassification import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")


        config = ConfigurationManager(
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
        )
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============================x")
    except Exception as e:
        logger.exception(e)
        raise e        
