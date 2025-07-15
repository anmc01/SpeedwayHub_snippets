from time import time

import optuna
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error

from Models.variables import HYPERPARAMETERS_DATASET, OPTUNA_TRIALS, OPTUNA_TIMEOUT
from Models.xgboost_model import TrainXgBoost


class OptunaXgboost(TrainXgBoost):
    def __init__(self, data, prints=True, hyper_details=False, no_of_trials=OPTUNA_TRIALS):
        super().__init__(data, test_range=[2024], prints=prints)
        self.create_regression_model()

        self.hyper_details = hyper_details
        self.no_of_trials = no_of_trials

        self.dtrain = xgboost.DMatrix(self.X_train, label=self.y_train, enable_categorical=True)
        self.dtest = xgboost.DMatrix(self.X_test, label=self.y_test, enable_categorical=True)

        self.optuna_results = None
        self.perform_optuning()

    def objective(self, trial):
        hyper_params = {
            "enable_categorical": True,
            "verbosity": 0,
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 6.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "max_leaves": trial.suggest_int("max_leaves", 2, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "reg_alpha": trial.suggest_int("reg_alpha", 10, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        }
        best_hyper_params = xgboost.train(hyper_params, self.dtrain)
        pred = best_hyper_params.predict(self.dtest)

        scoring = -mean_absolute_error(self.y_test, pred)

        return scoring

    def perform_optuning(self):
        start_time = time()

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.no_of_trials, timeout=OPTUNA_TIMEOUT, show_progress_bar=True)

        trial = study.best_trial
        best_params = trial.params.items()
        best_params = dict(best_params)
        self.optuna_results = best_params

        optuna_model = xgboost.XGBRegressor(**best_params, enable_categorical=True)
        optuna_model.fit(self.X_train, self.y_train)

        elapsed = time() - start_time

        if self.hyper_details:
            print(f"Time elapsed: {elapsed:.2f}")

            print(f"Number of finished trials: {len(study.trials)}")
            print("Best trial:")
            print(f"\tValue: {trial.value}\n\tParams:")
            for key, value in trial.params.items():
                print(f"\t\t{key}: {value}")

        self.model = optuna_model

        self.evaluate_regression()


if __name__ == '__main__':
    dataset = pd.read_parquet(f'../Datasets/{HYPERPARAMETERS_DATASET}')

    OptunaXgboost(dataset, hyper_details=True)
