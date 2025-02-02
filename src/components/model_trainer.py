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
        self.params_config= read_yaml(Path("params.yaml"))
        
    
    def get_model_training_config(self) -> Path:
        config= self.config.model_training
        params= self.params_config
        
        create_directories([config.root_dir])
        root_dir=config.root_dir
        train_data_path=config.train_data_path
        test_data_path=config.test_data_path
        
        return (root_dir, 
                train_data_path,
                params)
                
        
    
class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.params

    def objective(trial):
    model_name = trial.suggest_categorical("model", list(models.keys()))
    params = config["optuna_params"][model_name]
    
    model = models[model_name](**{key: trial.suggest_float(key, *val) for key, val in params.items()})
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

    def objective(trial):
    model_name = trial.suggest_categorical("model", list(models.keys()))
    params = config["optuna_params"][model_name]
    
    model = models[model_name](**{key: trial.suggest_float(key, *val) for key, val in params.items()})
    
    with mlflow.start_run(nested=True):  # Start a nested MLflow run
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        # Log model details
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_name", model_name)
        mlflow.sklearn.log_model(model, f"model_{model_name}")

    return metrics["f1_score"]  # Return F1-score for Optuna optimization


    def objective(trial, model_name):
    model_params = self.params["models"][model_name]

    hyperparams = {}
    for param, config in model_params.items():
        if config["type"] == "loguniform":
            hyperparams[param] = trial.suggest_loguniform(param, config["low"], config["high"])
        elif config["type"] == "int":
            hyperparams[param] = trial.suggest_int(param, config["low"], config["high"], config.get("step", 1))
        elif config["type"] == "uniform":
            hyperparams[param] = trial.suggest_float(param, config["low"], config["high"])
        elif config["type"] == "categorical":
            hyperparams[param] = trial.suggest_categorical(param, config["choices"])

    
    model_mapping = {
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
    }
    
    model = model_mapping[model_name](**hyperparams)

    # Perform cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score

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
            import mlflow


# Load params from YAML
with 

# Optuna Objective Function
def objective(trial):
    model_name = trial.suggest_categorical("model", list(models.keys()))
    params = config["optuna_params"][model_name]
    
    model = models[model_name](**{key: trial.suggest_float(key, *val) for key, val in params.items()})
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Get Best Params
best_params = study.best_params
best_model_name = best_params.pop("model")
best_model = models[best_model_name](**best_params)

# Train Best Model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluate Metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

# 
print("Best model:", best_model_name, "Metrics:", metrics)
            pass



import yaml
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Generate sample data (replace with your dataset)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna objective function for hyperparameter tuning
def objective(trial):
    model_name = trial.suggest_categorical("model", ["logistic_regression", "random_forest", "svm"])

    if model_name == "logistic_regression":
        C = trial.suggest_float("C", params["models"]["logistic_regression"]["C"]["low"], 
                                params["models"]["logistic_regression"]["C"]["high"], log=True)
        penalty = trial.suggest_categorical("penalty", params["models"]["logistic_regression"]["penalty"]["choices"])
        solver = trial.suggest_categorical("solver", params["models"]["logistic_regression"]["solver"]["choices"])
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
    
    elif model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", params["models"]["random_forest"]["n_estimators"]["low"], 
                                         params["models"]["random_forest"]["n_estimators"]["high"], 
                                         step=params["models"]["random_forest"]["n_estimators"]["step"])
        max_depth = trial.suggest_int("max_depth", params["models"]["random_forest"]["max_depth"]["low"], 
                                      params["models"]["random_forest"]["max_depth"]["high"])
        min_samples_split = trial.suggest_int("min_samples_split", params["models"]["random_forest"]["min_samples_split"]["low"], 
                                              params["models"]["random_forest"]["min_samples_split"]["high"])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    elif model_name == "svm":
        C = trial.suggest_float("C", params["models"]["svm"]["C"]["low"], params["models"]["svm"]["C"]["high"], log=True)
        kernel = trial.suggest_categorical("kernel", params["models"]["svm"]["kernel"]["choices"])
        gamma = trial.suggest_categorical("gamma", params["models"]["svm"]["gamma"]["choices"])
        model = SVC(C=C, kernel=kernel, gamma=gamma)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print best parameters
print("Best hyperparameters:", study.best_params)
