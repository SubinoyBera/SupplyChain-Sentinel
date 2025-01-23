import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

import pymysql
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()

class DataIngestionConfig:
    def __init__(self):
        self.config= read_yaml(Path("config.yaml"))
        logging.info("Reading config.yaml")
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> Path:
        config=self.config.data_ingestion
        try:
            create_directories([config.root_dir])
            data_dir= config.root_dir
            file_name= config.file_name
            data_ingestion_config= Path(data_dir)/file_name
            logging.info("Created data_ingestion path")
            
            return data_ingestion_config
        
        except Exception as e:
            logging.error(e, exc_info=True)
            raise CustomException(e,sys)
        

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.engine= None
        self.data_path= config.get_data_ingestion_config()
        
    def connect_mysql_server(self):
        try:
            user= os.getenv('user')
            password= os.getenv('password')
            host= os.getenv('host')
            database= os.getenv('database')
            
            self.engine=create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
            logging.info("Connection established with server")
        
        except Exception as e:
            logging.error(f"Server Connection Error: {e}", exc_info=True)
            raise CustomException(e,sys)
        
        
    def fetch_data(self):
        table_name= os.getenv('table')
        try:
            query=f"SELECT * FROM {table_name}"
            df= pd.read_sql_query(query, self.engine)
            
            logging.info("Reading data as DataFrame")
            df.to_csv(self.data_path, index=False, header=True)
            logging.info("Data read from database")
        
            return self.data_path
            
        except Exception as e:
            logging.error(f"Error in fetching data: {e}", exc_info=True)
            raise CustomException(e,sys)
        
        
    def close_mysql_connection(self):
        try:
            if self.engine:
                self.engine.dispose()
                logging.info("Connection closed")
            else:
                logging.info("No server connection found")
                
        except Exception as e:
            logging.error(f"Failed to close server connection: {e}", exc_info=True)
