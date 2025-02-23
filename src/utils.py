import os
import sys
import yaml
from src.logger import logging
from src.exception import CustomException
import box
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
import sklearn
from sklearn.metrics import classification_report
import mlflow

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content= yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully.")
            
            return ConfigBox(content)
        
    except Exception as e:
        raise CustomException(e,sys)
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory: {path}")
            
            
def evaluate_model(model, X_test, y_test):
    try: 
        y_pred= model.predict(X_test)
        report= classification_report(y_test, y_pred, output_dict=True)
        return report

    except Exception as e:
        logging.error(f"Error in {model} model evaluation: {e,sys}")
        

def track_experiment(model_name, model, model_params, report):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param('model_name', model_params)
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1-score_macro', report['macro avg']['f1-score'])
        
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")