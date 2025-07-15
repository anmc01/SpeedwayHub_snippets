from time import time

import optuna
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

from Models.Pairs.pairs_xgboost_model import TrainXgBoost
from Models.variables import OPTUNA_TRIALS, PAIRS_DATASET


class OptunaXgboost(TrainXgBoost):
    def __init__(self, data, prints=True, hyper_details=False, no_of_trials=OPTUNA_TRIALS, n_jobs=4, accuracy_threshold=0.709):
        super().__init__(data, test_range=[2024], prints=prints)
        self.create_classification_model()

        self.hyper_details = hyper_details
        self.no_of_trials = no_of_trials
        self.n_jobs = n_jobs
        self.accuracy_threshold = accuracy_threshold

        self.optuna_results = None
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
        model = xgboost.XGBClassifier(**hyper_params, n_jobs=1, tree_method="gpu_hist", gpu_id=0)
        model.fit(self.X_train, self.y_train)

        # Validate the model
        preds = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, preds)
        return accuracy

    def prune_study(self, study):
        """Filtruje próby poniżej progu accuracy, oznaczając je jako FAIL"""
        complete_trials = 0
        failed_trials = 0

        for trial in study.get_trials(deepcopy=False):
            if trial.value is not None and trial.value >= self.accuracy_threshold:
                trial.state = optuna.trial.TrialState.COMPLETE
                complete_trials += 1
            else:
                trial.state = optuna.trial.TrialState.FAIL
                failed_trials += 1

        if self.hyper_details:
            print(f"Próby oznaczone jako COMPLETE: {complete_trials}")
            print(f"Próby oznaczone jako FAIL: {failed_trials}")
            print(f"Sampler będzie używał {complete_trials} prób do próbkowania")

        return study

    def perform_optuning(self):
        start_time = time()

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.no_of_trials, n_jobs=self.n_jobs, show_progress_bar=True)

        # Po optymalizacji filtrujemy próby
        study = self.prune_study(study)

        trial = study.best_trial
        best_params = trial.params.items()
        best_params = dict(best_params)
        self.optuna_results = best_params

        elapsed = time() - start_time

        if self.hyper_details:
            print(f"Time elapsed: {elapsed:.2f}")
            print(f"Number of parallel jobs: {self.n_jobs}")
            print(f"Accuracy threshold: {self.accuracy_threshold}")

            print(f"Best hyperparameters: {study.best_params}")
            print(f"Best accuracy: {study.best_value}")

            print("Best trial:\n\tParams:")
            for key, value in trial.params.items():
                print(f"\t\t{key}: {value}")

        optuna_model = xgboost.XGBClassifier(**study.best_params, enable_categorical=True, verbosity=0, n_jobs=-1,
                                             tree_method="gpu_hist", gpu_id=0)

        self.model = optuna_model
        self.model.fit(self.X_train, self.y_train)
        self.evaluate_classification()


if __name__ == '__main__':
    dataset = pd.read_parquet(f'../../../Dataset/Datasets/{PAIRS_DATASET}')

    OptunaXgboost(dataset, hyper_details=True, n_jobs=4, accuracy_threshold=0.709)
