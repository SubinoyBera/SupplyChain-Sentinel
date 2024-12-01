import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline

try:
    logging.info("Data ingestion stage initiated")
    
    data_ingestion= DataIngestionPipeline()
    data_ingestion.main()
    logging.info("Data Ingestion stage completed")
    
except Exception as e:
    logging.error(f"Data Ingestion Failed: {e}", exc_info=True)
    raise CustomException(e,sys)