import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e :
        raise CustomException(e, sys)
    
    # src/utils.py



def evaluate_models(x_train, y_train, x_test, y_test, models,params):
    model_report = {}

    for name, model in models.items():
        if name in params:
            param_grid = params[name]
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)
            score = r2_score(y_test, y_pred)
            model_report[name] = score

    return model_report

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
