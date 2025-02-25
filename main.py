import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.experimentation_pipeline import ExperimentationPipeline

try:
    logging.info("STAGE:1 Data ingestion stage initiated")
    
    data_ingestion= DataIngestionPipeline()
    data_ingestion.main()
    logging.info("Data Ingestion stage completed\n\n")
    
except Exception as e:
    logging.error(f"Data Ingestion Failed: {e}", exc_info=True)
    raise CustomException(e,sys)

try:
    logging.info("STAGE:2 Data transformation stage initiated")
    
    data_transformation= DataTransformationPipeline()
    data_transformation.main()
    logging.info("Data Transformation stage completed\n\n")
    
except Exception as e:
    logging.error(f"Data Transformation Failed: {e}", exc_info=True)
    raise CustomException(e,sys)

try:
    logging.info("STAGE:3 Experimentation stage (model training-evaluation) initiated")
    
    model_trainer= ExperimentationPipeline()
    model_trainer.main()
    logging.info("Experimentation stage completed\n\n")
    
except Exception as e:
    logging.error(f"Model Experiments Failed: {e}", exc_info=True)
    raise CustomException(e,sys)