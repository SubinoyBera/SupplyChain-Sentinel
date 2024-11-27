import os
import sys
from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
load_dotenv()

import pymysql
from sqlalchemy import create_engine
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path= os.path.join("artifacts", "SupplyChain_Data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            user= os.getenv('user')
            password= os.getenv('password')
            host= os.getenv('host')
            database='Logistics_Management'
            table_name='supplychain_table'
            
            engine=create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
            query=f"SELECT * FROM {table_name}"
            df= pd.read_sql_query(query, engine)
            logging.info("Read data from MySQL server as DataFrame.")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Data ingestion is completed!")
            
            return(
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj= DataIngestion()
    obj.initiate_data_ingestion()