from time import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from Models.xgboost_model import TrainXgBoost


class RandomSearchXgBoost(TrainXgBoost):
    def __init__(self, data, prints=True, hyper_details=False):
        super().__init__(data, test_range=[2024], prints=prints)
        self.create_regression_model()

        self.hyper_details = hyper_details

        self.hyper_params = {
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5, 7, 10],
            'subsample': [0.25, 0.5, 0.75, 1.0],
            'colsample_bytree': [0.25, 0.5, 0.75, 1.0],
            'colsample_bylevel': [0.25, 0.5, 0.75, 1.0],
            'colsample_bynode': [0.25, 0.5, 0.75, 1.0],
            'n_estimators': [100, 200, 300, 400, 500],
            'reg_alpha': [0, 10, 50, 100],
            'reg_lambda': [0, 0.5, 1, 2]
        }

        self.grid_search_results = None
        self.perform_random_search()

    def perform_random_search(self):
        hyperparameter_combinations = 20
        cross_validation_folds = 3

        scoring = 'neg_mean_absolute_error'

        start_time = time()

        cross_validation_sets = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True)
        random_search = RandomizedSearchCV(self.model, param_distributions=self.hyper_params,
                                           n_iter=hyperparameter_combinations,
                                           scoring=scoring, n_jobs=4,
                                           cv=cross_validation_sets.split(self.X_train, self.y_train),
                                           verbose=2)

        random_search.fit(self.X_train, self.y_train)

        elapsed = time() - start_time

        self.grid_search_results = random_search.best_params_
        self.model = random_search.best_estimator_

        if self.hyper_details:
            print(f"Time elapsed: {elapsed:.2f}")
            print("Best hyperparameters found:")

            for key, value in self.grid_search_results.items():
                print(f"\t{key}: {value}")

        self.evaluate_regression()


if __name__ == '__main__':
    dataset = pd.read_pickle('../Datasets/dataset_full_20250407.pkl')

    RandomSearchXgBoost(dataset, hyper_details=True)
