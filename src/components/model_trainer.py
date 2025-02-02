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
import optuna
import joblib
from box import ConfigBox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Load parameters using ConfigBox
params = ConfigBox.from_yaml(filename="params.yaml")

# Generate sample data (Replace with your actual dataset)
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model trainer class
class ModelTrainer:
    def __init__(self, params):
        self.params = params.models

    def suggest_param(self, trial, param_name, param_info):
        """Dynamically handle different parameter types in Optuna"""
        param_type = param_info.type

        if param_type == "int":
            return trial.suggest_int(param_name, param_info.low, param_info.high)
        elif param_type == "float":
            return trial.suggest_float(param_name, param_info.low, param_info.high)
        elif param_type == "loguniform":
            return trial.suggest_float(param_name, param_info.low, param_info.high, log=True)
        elif param_type == "uniform":
            return trial.suggest_float(param_name, param_info.low, param_info.high)
        elif param_type == "categorical":
            return trial.suggest_categorical(param_name, param_info.choices)

    def train(self, trial):
        model_name = trial.suggest_categorical("model", list(self.params.keys()))
        model_params = self.params[model_name]

        if model_name == "logistic_regression":
            model = LogisticRegression(
                C=self.suggest_param(trial, "C", model_params.C),
                penalty=self.suggest_param(trial, "penalty", model_params.penalty),
                solver=self.suggest_param(trial, "solver", model_params.solver),
                max_iter=1000
            )

        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.suggest_param(trial, "n_estimators", model_params.n_estimators),
                max_depth=self.suggest_param(trial, "max_depth", model_params.max_depth),
                min_samples_split=self.suggest_param(trial, "min_samples_split", model_params.min_samples_split),
                random_state=42
            )

        elif model_name == "svm":
            model = SVC(
                C=self.suggest_param(trial, "C", model_params.C),
                kernel=self.suggest_param(trial, "kernel", model_params.kernel),
                gamma=self.suggest_param(trial, "gamma", model_params.gamma)
            )

        elif model_name == "xgboost":
            model = XGBClassifier(
                learning_rate=self.suggest_param(trial, "learning_rate", model_params.learning_rate),
                n_estimators=self.suggest_param(trial, "n_estimators", model_params.n_estimators),
                max_depth=self.suggest_param(trial, "max_depth", model_params.max_depth),
                subsample=self.suggest_param(trial, "subsample", model_params.subsample),
                use_label_encoder=False, eval_metric="logloss"
            )

        elif model_name == "lightgbm":
            model = LGBMClassifier(
                learning_rate=self.suggest_param(trial, "learning_rate", model_params.learning_rate),
                num_leaves=self.suggest_param(trial, "num_leaves", model_params.num_leaves)
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

# Run Optuna hyperparameter tuning
trainer = ModelTrainer(params)
study = optuna.create_study(direction="maximize")
study.optimize(trainer.train, n_trials=20)

# Save the best model
best_model = trainer.train(study.best_trial)
joblib.dump(best_model, "best_model.pkl")

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)




