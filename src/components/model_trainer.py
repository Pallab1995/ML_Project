import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utilis import save_object,evaluate_models

@dataclass

class ModelTraningConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")   #to save the model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTraningConfig()    


    def Initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            # features: all columns except last ; target: last column
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # ensure feature matrices are 2D and targets are 1D
            if X_train.ndim > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if X_test.ndim > 2:
                X_test = X_test.reshape(X_test.shape[0], -1)
            y_train = np.ravel(y_train)
            y_test = np.ravel(y_test)
            
            models={

            "Random Forest":RandomForestRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Gradient Boosting":GradientBoostingRegressor(),
            "Linear Regression":LinearRegression(),
            "K-Neighbors Regression":KNeighborsRegressor(),
            "XGB Regression":XGBRegressor(),
            "CatBoosting Regression":CatBoostRegressor(),
            "AdaBoost Regression":AdaBoostRegressor()
            
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score= max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[

                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both traning and testing dataset")

            save_object(

                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)
            return r2_square


		
        except Exception as e:
                raise CustomException(e,sys)