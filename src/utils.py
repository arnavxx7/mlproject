import numpy as np
import pandas as pd
import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Dictionary containing all model names as keys and their respective r2 scores as values
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            #Hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(estimator=model, param_grid=para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)   #Training the model

            y_train_pred = model.predict(X_train)  #Predictions for the training data

            y_test_pred = model.predict(X_test)  #Predictions for the test data

            train_model_score = r2_score(y_train, y_train_pred)   #Model accuracy on the training data

            test_model_score = r2_score(y_test, y_test_pred)  #Model accuracy on test data

            report[list(models.keys())[i]] = test_model_score

        return report


 
    except Exception as e:
        raise CustomException(e,sys)