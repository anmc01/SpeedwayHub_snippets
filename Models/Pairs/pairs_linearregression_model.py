from time import time

import pandas as pd
from sklearn.linear_model import LogisticRegression

from Models.Pairs.parent import PairsModel
from Models.utils import drop_pairs_columns
from Models.utils import prepare_pairs_evaluation_df
from Models.variables import PAIRS_DATASET


class TrainLinearRegression(PairsModel):
    def __init__(self, data, train_range=None, test_range=None, hyperparameters=False, prints=True, plots=False, details=False):
        super().__init__(data, train_range, test_range, hyperparameters, prints, plots, details)

        self.X_train = self.handle_categoricals(self.X_train)
        self.X_test = self.handle_categoricals(self.X_test)

        nan_rows_train = self.X_train[self.X_train.isna().any(axis=1)].index
        nan_rows_test = self.X_test[self.X_test.isna().any(axis=1)].index

        self.X_train = self.X_train.drop(index=nan_rows_train)
        self.y_train = self.y_train.drop(index=nan_rows_train)
        self.X_test = self.X_test.drop(index=nan_rows_test)
        self.y_test = self.y_test.drop(index=nan_rows_test)

    @staticmethod
    def handle_categoricals(df):
        df = drop_pairs_columns(df)

        categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_columns)

        return df

    def create_classification_model(self):
        start_time = time()
        self.model = LogisticRegression()

        self.model.fit(self.X_train, self.y_train.values.ravel())

        end_time = time()
        self.elapsed = end_time - start_time


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
    pairs = TrainLinearRegression(dataset, test_range=[2024])
    pairs.create_classification_model()
    pairs.evaluate_classification()
