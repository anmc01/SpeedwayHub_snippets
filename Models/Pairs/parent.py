import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import Models.classification_evaluation as ceval


class PairsModel:
    def __init__(self, data, train_range=None, test_range=None, hyperparameters=False, prints=True, plots=False, details=False):
        self.df = data

        self.test_range = test_range

        if train_range is None and test_range is not None:
            years_in_df = self.df['Year'].unique().tolist()
            self.train_range = [year for year in years_in_df if year not in self.test_range]
        else:
            self.train_range = train_range

        self.hyperparameters = hyperparameters

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()
        self.predicted_df = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)

        self.model, self.predictions, self.evaluation_df = None, None, None

        self.prints, self.plots, self.details = prints, plots, details

        self.elapsed = 0
        self.pairs, self.accuracy, self.accuracy_with_bonuses, self.competition_prediction_accuracy = None, None, None, None
        self.importance_weight, self. importance_gain, self.cm = None, None, None

        plt.style.use('dark_background')

    def split_dataset(self):
        if self.train_range is None and self.test_range is None:
            X = self.df.drop(axis=1, columns=['Result'])
            y = self.df['Result']

            return train_test_split(X, y, test_size=0.25, random_state=42)

        elif self.train_range is not None and self.test_range is not None:
            X = self.df.drop(axis=1, columns=['Result'])
            y = self.df[['Year', 'Result']]

            X_train = X[X['Year'].isin(self.train_range)]
            X_test = X[X['Year'].isin(self.test_range)]
            y_train = y[y['Year'].isin(self.train_range)]
            y_test = y[y['Year'].isin(self.test_range)]

            y_train = y_train.loc[:, y_train.columns != 'Year']
            y_test = y_test.loc[:, y_test.columns != 'Year']

            return X_train, X_test, y_train, y_test

    def evaluation_metrics(self):
        if self.prints:
            print(f"Elapsed time: {self.elapsed:.2f}s")

        self.pairs = ceval.classification_accuracy(
            self.predicted_df, 'Result', 'Pair_prediction', 'plain pairs', self.prints
        )
        ceval.classification_accuracy(
            self.evaluation_df, 'Result', 'Pair_prediction', 'summed pairs', self.prints
        )
        self.accuracy = ceval.classification_accuracy(
            self.evaluation_df, 'Rider_points', 'Predicted_points_integer', "", self.prints
        )

        self.accuracy_with_bonuses = ceval.classification_accuracy(
            self.evaluation_df, 'Rider_points', 'Predicted_points_integer_with_bonus', 'with bonuses', self.prints
        )

    def evaluation_plots(self):
        self.importance_weight = ceval.classification_importance(self.model, "weight", self.plots)
        self.importance_gain = ceval.classification_importance(self.model, "gain", self.plots)
        self.cm = ceval.classification_cm(self.evaluation_df, self.plots)

    def evaluation_details(self):
        if self.details:
            ceval.analyze_classification_based_on_elo(self.evaluation_df, print_details=False)
            ceval.analyze_classification_based_on_match(self.evaluation_df)
