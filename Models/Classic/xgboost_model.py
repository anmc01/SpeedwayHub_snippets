from time import time

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from Models.Classic.parent import RegressionModel
from Models.utils import prepare_evaluation_df, drop_columns
from Models.variables import MODELS_DATASET


class TrainXgBoost(RegressionModel):
    def __init__(self, data, train_range=None, test_range=None, hyperparameters=False, prints=True, plots=False, details=False):
        super().__init__(data, train_range, test_range, hyperparameters, prints, plots, details)

        self.X_train = drop_columns(self.X_train)
        self.X_test = drop_columns(self.X_test)

    def create_regression_model(self):
        if self.hyperparameters:
            start_time = time()

            self.model = xgb.XGBRegressor(
                enable_categorical=True,
                verbosity=0,
                colsample_bylevel=0.9187990540642073,
                colsample_bynode=0.656015091250099,
                colsample_bytree=0.7165775490100128,
                gamma=0.9774351044135563,
                learning_rate=0.03244716831851254,
                max_depth=7,
                max_leaves=15,
                min_child_weight=8,
                n_estimators=393,
                reg_alpha=12,
                reg_lambda=1.2602576487181387,
                subsample=0.9999809370993155,
            )
        else:
            start_time = time()

            self.model = xgb.XGBRegressor(enable_categorical=True, verbosity=0)

        self.model.fit(self.X_train, self.y_train)

        end_time = time()
        self.elapsed = end_time - start_time

    def evaluate_regression(self):
        self.predictions = self.model.predict(self.X_test)
        self.predicted_df_regression['Predicted_points'] = self.predictions
        self.evaluation_df = prepare_evaluation_df(self.predicted_df_regression)

        if self.plots:
            fig, ax = plt.subplots(figsize=(20, 14))
            xgb.plot_importance(
                self.model,
                title="Feature importances - XGBoost",
                ax=ax
            )
            plt.tight_layout()
            plt.show()

        self.evaluation_metrics()
        self.evaluation_plots()
        self.evaluation_details()


if __name__ == '__main__':
    dataset = pd.read_parquet(f"../../Dataset/Datasets/{MODELS_DATASET}")

    print("CLASSIFIED REGRESSION".center(40, " "))
    regression = TrainXgBoost(dataset, test_range=[2024], hyperparameters=False, plots=False, details=False)
    regression.create_regression_model()
    regression.evaluate_regression()
