import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformationConfig, DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config= DataTransformationConfig()
        data_transformation= DataTransformation(config)

        logging.info("Data transformation started")
        data_transformation.get_transformation()
        
        logging.info("Splitting data")
        data_transformation.train_test_split()

        
if __name__=='__main__':
    try:
        logging.info("Starting data transformation pipeline")
        obj= DataTransformationPipeline()
        obj.main()
        logging.info("Data transformation completed.")
        
    except Exception as e:
        raise CustomException(e,sys)