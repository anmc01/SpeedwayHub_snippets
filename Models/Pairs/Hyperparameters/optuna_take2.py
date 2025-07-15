import optuna
import optuna.visualization.matplotlib as opplt
import matplotlib.pyplot as plt
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

from Models.variables import OPTUNA_TRIALS, OPTUNA_TIMEOUT, PAIRS_DATASET
from Models.Pairs.pairs_xgboost_model import TrainXgBoost


class Optuna2Xgboost(TrainXgBoost):
    def __init__(self, data, prints=True, hyper_details=False, no_of_trials=OPTUNA_TRIALS, optuna_plots=False):
        super().__init__(data, test_range=[2024], prints=prints)

        self.create_classification_model()

        self.hyper_details = hyper_details
        self.no_of_trials = no_of_trials
        self.optuna_plots = optuna_plots

        self.optuna_results = None

        plt.style.use('default')
        self.perform_optuning()

    def objective(self, trial):
        hyper_params = {
            "enable_categorical": True,
            "verbosity": 0,
            "objective": "binary:logistic",
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

        # Train the model
        model = xgboost.XGBClassifier(**hyper_params)
        model.fit(self.X_train, self.y_train)

        # Validate the model
        preds = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, preds)
        return accuracy

    def perform_optuning(self):
        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.no_of_trials, show_progress_bar=True)

        trial = study.best_trial
        best_params = trial.params.items()
        best_params = dict(best_params)
        self.optuna_results = best_params

        if self.hyper_details:
            print("Best hyperparameters:", study.best_params)
            print("Best accuracy:", study.best_value)

        if self.optuna_plots:
            opplt.plot_param_importances(study)
            plt.show()

        optuna_model = xgboost.XGBClassifier(**study.best_params, enable_categorical=True, verbosity=0)

        self.model = optuna_model
        self.model.fit(self.X_train, self.y_train)
        self.evaluate_classification()


if __name__ == '__main__':
    dataset = pd.read_parquet(f'../../../Dataset/Datasets/dataset_pairs_0522_full.parquet')

    # OptunaXgboost(dataset, hyper_details=True, no_of_trials=100)

    results = list()
    for i in range(5):
        hyper = Optuna2Xgboost(dataset, prints=False, hyper_details=True, optuna_plots=True)
        results.append([i + 1, hyper.pairs, hyper.accuracy, hyper.accuracy_with_bonuses, hyper.optuna_results])
        print(results[-1][:-1])
