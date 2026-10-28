from src.renalDiseaseClassification.config.configuration import ConfigurationManager
from src.renalDiseaseClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.renalDiseaseClassification.entity.config_entity import PrepareBaseModelConfig
from src.renalDiseaseClassification.components.prepare_base_model import PrepareBaseModel
from src.renalDiseaseClassification import logger


STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")


        config = ConfigurationManager(
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
        )
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============================x")
    except Exception as e:
        logger.exception(e)
        raise e           