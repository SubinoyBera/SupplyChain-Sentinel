import sys
from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config= ModelTrainerConfig()
        model_trainer= ModelTrainer(config)

        #logging.info("Starting model trainig")
        model_trainer.initiate_model_training()
        
     
if __name__=='__main__':
    try:
        logging.info("Starting trainig pipeline")
        obj= ModelTrainerPipeline()
        obj.main()
        logging.info("Model training completed.")
        
    except Exception as e:
        raise CustomException(e,sys)