from src.renalDiseaseClassification import logger
from src.renalDiseaseClassification.config.configuration import ConfigurationManager, CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.renalDiseaseClassification.components.model_evaluation_mlflow import Evaluation


STAGE_NAME = "Model Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH
        )
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx===============================x")
    except Exception as e:
        logger.exception(e)
        raise e        