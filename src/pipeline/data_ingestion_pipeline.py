import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestionConfig, DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config= DataIngestionConfig()
        data_ingestion= DataIngestion(config)

        logging.info("Connecting MySQL server")
        data_ingestion.connect_mysql_server()

        logging.info("Fetching data from server")
        data_ingestion.fetch_data()

        logging.info("Closing server connection")
        data_ingestion.close_mysql_connection()
        
if __name__=='__main__':
    try:
        logging.info("Starting data ingestion pipeline")
        obj= DataIngestionPipeline()
        obj.main()
        logging.info("Data ingestion completed.")
        
    except Exception as e:
        raise CustomException(e,sys)