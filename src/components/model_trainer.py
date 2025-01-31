import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek


class ModelTrainerConfig:
    def __init__(self):
        self.config= read_yaml(Path("config.yaml"))
        self.params= read_yaml(Path("params.yaml"))
        
    
    def get_model_training_config(self) -> Path:
        config= self.config.model_training
        params= self.params
        
        create_directories([config.root_dir])
        root_dir=config.root_dir
        train_data_path=config.train_data_path
        test_data_path=config.test_data_path
    
    
class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        pass
    
    try:
        pass
    except:
        pass