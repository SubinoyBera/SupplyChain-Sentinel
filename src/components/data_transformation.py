import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek

class DataTransformationConfig:
    def __init__(self):
        self.config= read_yaml(Path("config.yaml"))
        logging.info("Reading config.yaml")
    
    def get_data_transformation_config(self) -> Path:
        config=self.config.data_transformation
        try:
            create_directories([config.root_dir])
            root_dir= config.root_dir
            data_path= config.file_name
            data_transformation_config= Path(root_dir, data_path)
            logging.info("Created data_ingestion path")
            
            return data_transformation_config
        
        except Exception as e:
            logging.error(e, exc_info=True)
            raise CustomException(e,sys)

    
class DataTransformation:
    def __init__(self, config=DataTransformationConfig):
        self.config = config
       
    def LabelEncoding(self, x):
        le=LabelEncoder()
        x=le.fit_transform(x)
        return x
    
    
    def get_data_balancing(self, train):
        pass
    
    def get_transformation(self):
        
        data=pd.read_csv(self.config.data_path)
        
        data['Suspected_Fraud']= np.where(['Order_Status']=='SUSPECTED_FRAUD', 1, 0)
        
        df= data['Type', 'Customer_Id', 'Order_Customer_Id', 'Order_Region', 'Customer_Country', 
                        'Customer_City', 'Customer_Segment', 'Order_City', 'Order_State', 
                        'Order_Country', 'Late_DeliveryRisk', 'Shipping_Mode', 'Suspected_Fraud']
            
        df= df.apply(self.LabelEncoding())
        
        train, test= train_test_split(df, test_size=0.2, random_state=42)
        
        test.to_csv(os.path.join(self.config.root_dir), index=False)
        
        self.get_data_balancing(train)