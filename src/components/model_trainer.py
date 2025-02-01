import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek


class ModelTrainerConfig:
    def __init__(self):
        self.config= read_yaml(Path("config.yaml"))
        self.params= read_yaml(Path("params.yaml"))
        
    
    def get_model_training_config(self) -> Path:
        config= self.config.model_training
        params= self.params.models
        
        create_directories([config.root_dir])
        root_dir=config.root_dir
        train_data_path=config.train_data_path
        test_data_path=config.test_data_path
        
        return params
        
    
class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.params
    
    def initiate_model_training(self):
        try:
            X_train= df.drop(['Suspected_Fraud'], axis=1)
            y_train= df['Suspected_Fraud']
            
            model= {
                "svm": SVC(),
                "logistic_regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "randon_forest": RandomForestClassifier(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Naive Bayes": GaussianNB(),
                "lightGBM": LGBMClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }
        
        except:
            pass