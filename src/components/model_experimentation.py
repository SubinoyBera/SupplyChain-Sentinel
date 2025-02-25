import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories, evaluate_model, track_experiment

import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
import mlflow


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

     
class DataBalancer():
    def __init__(self):
        pass
            
    def smote_tomek(self, X_train, y_train):
        sm= SMOTETomek(0.75)
        X_train_sm= sm.fit_resample(X_train)
        y_train_sm= sm.fit_resample(y_train)  
        return (X_train_sm,
                y_train_sm)
    
    def adasyn(self, X_train, y_train):
        sm= ADASYN()
        X_train_ads= sm.fit_resample(X_train)
        y_train_ads= sm.fit_resample(y_train)
        return (X_train_ads,
                y_train_ads)
        

class ModelExperimentation:
    def __init__(self, config:ModelTrainerConfig):
        self.root_dir, self.train_data_path, self.test_data_path, self.params= config.get_model_trainer_config()

    def model_trainer(self, X_train, y_train, X_test, y_test):
        hyperparams= self.params.models
        mlflow.set_experiment("Fraud Detection")
        mlflow.set_tracking_uri(uri="")
        
        #Train models on the imbalanced data
        models_set1={
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XgBoost": XGBClassifier(), 
        }
        
        for model_name, model in models_set1.items():
            try:
                logging.info(f"Tuning {model_name} model...")
                grid_search= GridSearchCV(model, hyperparams[model_name], cv=5,
                                           scoring="roc_auc", n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_params= grid_search.best_params_
                
            except Exception as e:
                logging.error(f"Error in tuning {model_name}: {e}", exc_info=True)
                raise CustomException(e,sys)

            try:
                logging.info(f"Started training: {model_name}")
                best_model= model.set_params(**best_params)
                best_model.fit(X_train, y_train)
                logging.info("Training completed")
                try:
                    report= evaluate_model(model=best_model, X_test=X_test, y_test=y_test)
                    logging.info("Model evaluation completed")
                    
                    track_experiment(model_name=model_name, model=best_model,
                                     model_params=best_params, report=report)
                    logging.info("Experiment successfully tracked")
                    
                except Exception as e:
                    logging.error(f"Error in experiment tracking/evaluation: {e}", exc_info=True)
                    raise CustomException(e,sys)

                joblib.dump(best_model, os.path.join(self.root_dir, model_name))
                logging.info(f"{model_name} model successfully saved in dir")
            
            except Exception as e:
                logging.error(f"Error in model training: {e}", exc_info=True)
                raise CustomException(e,sys)
        
        #Train models with applying data balancing 
        models_set2={
            "RandomForest": RandomForestClassifier(),
            "LightGBM": LGBMClassifier(),
            "XgBoost": XGBClassifier(), 
        }
        
        balancers=['smote_tomek', 'adasyn']    
        obj= DataBalancer()
        for model_name, model in models_set2.items():
            for i in range(len(balancers)):
                X_train_res, y_train_res= obj[balancers[i]](X_train, y_train)
                try:
                    logging.info(f"Tuning {model_name} with {balancers[i]}...")
                    grid_search= GridSearchCV(model, hyperparams[model_name], cv=5,
                                               scoring="roc_auc", n_jobs=-1)
                    grid_search.fit(X_train_res, y_train_res)
                    best_params= grid_search.best_params_
                except Exception as e:
                    logging.error(f"Error in tuning {model_name} with {balancers[i]}): {e}", exc_info=True)
                    raise CustomException(e,sys)

                try:
                    logging.info(f"Started training: {model_name} with {balancers[i]}")
                    best_model= model.set_params(**best_params)
                    best_model.fit(X_train_res, y_train_res)
                    logging.info("Training completed")
                    
                    model_rename= f"{model_name} with {balancers[i]}"
                    try:
                        report= evaluate_model(model=best_model, X_test=X_test, y_test=y_test)
                        logging.info("Model evaluation completed")
                        
                        track_experiment(model_name=model_rename, model=best_model,
                                     model_params=best_params, report=report)
                        logging.info("Experiment successfully tracked")
                        
                    except Exception as e:
                        logging.error(f"Error in experiment tracking/evaluation: {e}", exc_info=True)
                        raise CustomException(e,sys)
                
                    joblib.dump(best_model, os.path.join(self.root_dir, model_rename))
                    logging.info(f"{model_rename} successfull saved in dir")
        
                except Exception as e:
                    logging.error(f"Error occured during tuning and training model: {e}", exc_info=True)
                    raise CustomException(e,sys)
            

    def initiate_model_experimentation(self):
        train_data= pd.read_csv(self.train_data_path)
        test_data= pd.read_csv(self.test_data_path)
        
        X_train, X_test= train_data.drop(["Suspected_Fraud"],axis=1), test_data.drop(["Suspected_Fraud"],axis=1)
        y_train, y_test= train_data["Suspected_Fraud"], test_data["Suspected_Fraud"]
        
        logging.info("Model training initiated")
        self.train(X_train, y_train, X_test, y_test)
