import sys
from src.logger import logging
from src.exception import CustomException
from src.components.model_experimentation import ModelTrainerConfig, ModelExperimentation

class ExperimentationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config= ModelTrainerConfig()
        model_exp= ModelExperimentation(config)

        #logging.info("Starting model trainig")
        model_exp.initiate_model_experimentation()
        
     
if __name__=='__main__':
    try:
        logging.info("Starting trainig pipeline")
        obj= ExperimentationPipeline()
        obj.main()
        logging.info("Model training completed.")
        
    except Exception as e:
        raise CustomException(e,sys)