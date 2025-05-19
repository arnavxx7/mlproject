import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
                )
            models = {
                        "Linear Regression": LinearRegression(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor(),
                        "Gradient Boosting Regressor":GradientBoostingRegressor()
                    }
            
            params = {
                "Linear Regression": {},
                "K-Neighbors Regressor":{
                    "n_neighbors":[5,7,9,11]
                },
                "Decision Tree": {
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Random Forest Regressor": {
                    "n_estimators":[8,16,24,48,64]
                },
                "XGBRegressor":{
                    "learning_rate":[0.1, 0.01, 0.05, 0.001]
                },
                "CatBoosting Regressor":{
                    "depth":[6,8,10]
                },
                "AdaBoost Regressor":{
                    "learning_rate":[0.1, 0.01, 0.05, 0.001]
                },
                "Gradient Boosting Regressor": {
                    "learning_rate":[0.1, 0.01, 0.05, 0.001]
                },
            }
            logging.info("Training all models and evaluating them")
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Get the best model score from dict

            best_model_score = max(sorted(list(model_report.values())))

            # Get the best model name

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Get the best model

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found.")
            
            logging.info("Best model found on both training and test data")

            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
            )

            #Returning the output and accuracy of the best model

            y_pred = best_model.predict(X_test)

            best_r2_score = r2_score(y_test, y_pred)

            return best_r2_score, best_model_name
        except Exception as e:
            raise CustomException(e, sys)