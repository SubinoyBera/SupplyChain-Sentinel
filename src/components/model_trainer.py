import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn. metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek


class ModelTrainerConfig:
    def __init__(self):
        self.config= read_yaml(Path("config.yaml"))
        self.params_config= read_yaml(Path("params.yaml"))
        
    
    def get_model_trainer_config(self) -> Path:
        config= self.config.model_training
        params= self.params_config
        
        create_directories([config.root_dir])
        root_dir=config.root_dir
        train_data_path=config.train_data_path
        test_data_path=config.test_data_path
        
        return (root_dir, 
                train_data_path,
                test_data_path,
                params)
        

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.root_dir, self.train_data_path, self.test_data_path, self.params= config.get_model_trainer_config()


    def train(self, X_train, y_train):
        hyperparams= self.params.models

        models={
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC()
        }
        
        try: 
            for model_name, model in models.items():
                logging.info(f"Tuning {model_name}...")
                grid_search= GridSearchCV(model, hyperparams[model_name], cv=5,
                                           scoring="roc_auc", n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_params= grid_search.best_params_
                best_model= model.set_params(**best_params)
                best_model.fit(X_train, y_train)

                joblib.dump(best_model, os.path.join(self.root_dir, model_name))
                logging.info(f"{model_name} successfull trained")
        
        except Exception as e:
            logging.error(f"Error occured during tuning and trining model: {e}", exc_info=True)
            raise CustomException(e,sys)
            

    def initiate_model_training(self):
        train_data= pd.read_csv(self.train_data_path)
        test_data= pd.read_csv(self.test_data_path)
        
        X_train, X_test= train_data.drop(["Suspected_Fraud"],axis=1), test_data.drop(["Suspected_Fraud"],axis=1)
        y_train, y_test= train_data["Suspected_Fraud"], test_data["Suspected_Fraud"]
        
        logging.info("Model training initiated")
        self.train(X_train, y_train)