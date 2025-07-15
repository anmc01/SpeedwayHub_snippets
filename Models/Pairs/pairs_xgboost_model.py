from time import time

import pandas as pd
import xgboost as xgb

from Models.Pairs.parent import PairsModel
from Models.utils import drop_pairs_columns
from Models.utils import prepare_pairs_evaluation_df
from Models.variables import PAIRS_DATASET


class TrainXgBoost(PairsModel):
    def __init__(self, data, train_range=None, test_range=None, hyperparameters=False, prints=True, plots=False, details=False):
        super().__init__(data, train_range, test_range, hyperparameters, prints, plots, details)

        self.X_train = drop_pairs_columns(self.X_train)
        self.X_test = drop_pairs_columns(self.X_test)

    def create_classification_model(self):
        if self.hyperparameters:
            start_time = time()

            self.model = xgb.XGBClassifier(
                enable_categorical=True,
                verbosity=0,
                colsample_bylevel=0.49573232857956046,
                colsample_bynode=0.4295081671605477,
                colsample_bytree=0.8619628634893782,
                gamma=1.4019922727498564,
                learning_rate=0.28913881001542213,
                max_depth=5,
                max_leaves=84,
                min_child_weight=2,
                n_estimators=327,
                reg_alpha=12,
                reg_lambda=0.5929197455797863,
                subsample=0.885798184206069,
                n_jobs=-1,
                device='cuda'
            )
        else:
            start_time = time()

            self.model = xgb.XGBClassifier(
                enable_categorical=True,
                verbosity=0,
                n_jobs=-1,
                device='cuda'
            )

        self.model.fit(self.X_train, self.y_train)

        end_time = time()
        self.elapsed = end_time - start_time

        # ficik = self.model.fit(self.X_train, self.y_train).feature_importances_
        # print(ficik)

    def evaluate_classification(self):
        predictions = self.model.predict(self.X_test)
        predictions_probs = self.model.predict_proba(self.X_test)[:, 1]

        self.predicted_df['Pair_prediction'] = predictions
        self.predicted_df['Pair_prediction_prob'] = predictions_probs

        self.evaluation_df = prepare_pairs_evaluation_df(self.predicted_df)

        self.evaluation_metrics()
        self.evaluation_plots()
        self.evaluation_details()


if __name__ == '__main__':
    dataset = pd.read_parquet(f'../../Dataset/Datasets/{PAIRS_DATASET}')

    print("CLASSIFICATION".center(40, " "))
    pairs = TrainXgBoost(dataset, test_range=[2024], hyperparameters=True, plots=False, details=False)
    pairs.create_classification_model()
    pairs.evaluate_classification()
