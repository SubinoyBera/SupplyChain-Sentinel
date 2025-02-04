import os
import sys
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from utils import read_yaml, create_directories

import pandas as pd
import joblib
import optuna
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
from sklearn. metrics import accuracy_score

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


class DataBalancer():
    def __init__():
        pass
    
    def smote(X_train, y_train):
        sm= SMOTE(random_state=42)
        X_train_sm, y_train_sm= sm.fit_resample(X_train, y_train)
        return (X_train_sm,
                y_train_sm)
    
    def adasyn(X_train, y_train):
        ada= ADASYN(random_state=42)
        X_train_ada, y_train_ada= ada.fit_resample(X_train, y_train)
        return (X_train_ada,
                y_train_ada)
    
    def enn(X_train, y_train):
        en= EditedNearestNeighbours()
        X_train_en, y_train_en= en.fit_resample(X_train, y_train)
        return (X_train_en,
                y_train_en)
    
    def smote_tomek(X_train, y_train):
        smt= SMOTETomek(0.75)
        X_train_smt, y_train_smt= smt.fit_resample(X_train, y_train)
        return (X_train_smt,
                y_train_smt)
        

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.root_dir, self.train_data_path, 
        self.test_data_path, self.params= config.get_model_trainer_config()

    def suggest_param(self, trial, param_name, param_info):
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


    def train(self, trial, X_train, y_train, X_test, y_test):
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
    
    
    def initiate_model_training(self):
        train_data= pd.read_csv(self.train_data_path)
        test_data= pd.read_csv(self.test_data_path)
        
        X_train, X_test= train_data.drop(["Suspected_Fraud"],axis=1), test_data.drop(["Suspected_Fraud"],axis=1)
        y_train, y_test= train_data(["Suspected_Fraud"]), test_data(["Suspected_Fraud"])
        
        balancers=["smote", "adasyn", "enn", "smote_tomek"]
        obj= DataBalancer()
        
        for i in balancers:
            if i=="smote":
                X_train_res, y_train_res= obj.smote(X_train, y_train)
                study = optuna.create_study(direction="maximize")
                study.optimize(self.train(trial=20, X_train=X_train_res, y_train=y_train_res,
                                          X_test=X_test, y_test=y_test))
    
    
    