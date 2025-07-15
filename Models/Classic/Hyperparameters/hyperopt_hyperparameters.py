from time import time

import pandas as pd
import xgboost
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error

from Models.variables import HYPERPARAMETERS_DATASET, HYPEROPT_EVALUATIONS
from Models.xgboost_model import TrainXgBoost


class HyperOptXgboost(TrainXgBoost):
    def __init__(self, data, prints=True, hyper_details=False, no_of_evals=HYPEROPT_EVALUATIONS):
        super().__init__(data, test_range=[2024], prints=prints)
        self.create_regression_model()

        self.hyper_details = hyper_details
        self.no_of_evals = no_of_evals

        self.hyper_params = {
            "colsample_bylevel": hp.uniform("colsample_bylevel", 0.1, 1),
            "colsample_bynode": hp.uniform("colsample_bynode", 0.1, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1),
            "gamma": hp.uniform("gamma", 0, 6),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "max_leaves": hp.quniform("max_leaves", 5, 100, 5),
            "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
            "n_estimators": hp.quniform("n_estimators", 100, 400, 1),
            "reg_alpha": hp.quniform("reg_alpha", 10, 100, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "subsample": hp.uniform("subsample", 0.1, 1)
        }

        self.hyperopt_results = None
        self.perform_hyperopt()

    def objective(self, space):
        params = {
            "enable_categorical": True,
            "colsample_bylevel": space['colsample_bylevel'],
            "colsample_bynode": space['colsample_bynode'],
            "colsample_bytree": space['colsample_bytree'],
            "gamma": space['gamma'],
            "learning_rate": space['learning_rate'],
            "max_depth": int(space['max_depth']),
            "max_leaves": int(space['max_leaves']),
            "min_child_weight": int(space['min_child_weight']),
            "n_estimators": int(space['n_estimators']),
            "reg_alpha": int(space['reg_alpha']),
            "reg_lambda": space['reg_lambda'],
            "subsample": space['subsample']
        }

        model_for_trial = xgboost.XGBRegressor(**params)
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        model_for_trial.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=0)
        pred = model_for_trial.predict(self.X_test)
        scoring = -mean_absolute_error(self.y_test, pred)

        return {'loss': -scoring, 'status': STATUS_OK}

    def perform_hyperopt(self):
        start_time = time()

        trials = Trials()

        best_hyper_params = fmin(
            fn=self.objective,
            space=self.hyper_params,
            algo=tpe.suggest,
            max_evals=self.no_of_evals,
            trials=trials
        )
        self.hyperopt_results = best_hyper_params

        best_params = {
            "enable_categorical": True,
            "colsample_bylevel": best_hyper_params['colsample_bylevel'],
            "colsample_bynode": best_hyper_params['colsample_bynode'],
            "colsample_bytree": best_hyper_params['colsample_bytree'],
            "gamma": best_hyper_params['gamma'],
            "learning_rate": best_hyper_params['learning_rate'],
            "max_depth": int(best_hyper_params['max_depth']),
            "max_leaves": int(best_hyper_params['max_leaves']),
            "min_child_weight": int(best_hyper_params['min_child_weight']),
            "n_estimators": int(best_hyper_params['n_estimators']),
            "reg_alpha": int(best_hyper_params['reg_alpha']),
            "reg_lambda": best_hyper_params['reg_lambda'],
            "subsample": best_hyper_params['subsample']
        }

        hyperopt_model = xgboost.XGBRegressor(**best_params)
        hyperopt_model.fit(self.X_train, self.y_train)

        end_time = time()
        elapsed = end_time - start_time

        if self.hyper_details:
            print(f"Time elapsed: {elapsed:.2f}")
            print("Best hyperparameters found:")

            for key, value in self.hyperopt_results.items():
                print(f"\t{key}: {value}")

        self.model = hyperopt_model

        self.evaluate_regression()


if __name__ == '__main__':
    dataset = pd.read_parquet(f'../Datasets/{HYPERPARAMETERS_DATASET}')

    HyperOptXgboost(dataset, hyper_details=True)
