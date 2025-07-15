import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import Models.regression_evaluation as reval


class RegressionModel:
    def __init__(self, data, train_range=None, test_range=None, hyperparameters=False, prints=True, plots=False, details=False, drop_nans=False):
        self.df = data
        self.df = self.df[self.df['Letter'].isna()]

        self.test_range = test_range

        if train_range is None and test_range is not None:
            years_in_df = self.df['Year'].unique().tolist()
            self.train_range = [year for year in years_in_df if year not in self.test_range]
        else:
            self.train_range = train_range

        self.hyperparameters = hyperparameters

        if drop_nans:
            self.df = self.df.dropna(axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_dataset()
        self.predicted_df_regression = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)

        self.model, self.predictions, self.evaluation_df = None, None, None

        self.prints, self.plots, self.details = prints, plots, details

        self.elapsed = 0
        self.mae, self.rps = None, None
        self.accuracy, self.accuracy_with_bonuses, self.competition_prediction_accuracy = None, None, None
        self.cm = None

        plt.style.use('dark_background')

    def split_dataset(self):
        if self.train_range is None and self.test_range is None:
            X = self.df.drop(axis=1, columns=['Rider_points'])
            y = self.df['Rider_points']

            return train_test_split(X, y, test_size=0.25, random_state=42)

        elif self.train_range is not None and self.test_range is not None:
            X = self.df.drop(axis=1, columns=['Rider_points'])
            y = self.df[['Year', 'Rider_points']]

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

        self.mae = reval.classified_regression_mae(self.y_test, self.predictions, self.prints)
        self.rps = reval.classified_regression_rps(self.evaluation_df, self.prints)

        self.accuracy = reval.classified_regression_accuracy(self.evaluation_df, self.prints)
        self.accuracy_with_bonuses = reval.classified_regression_accuracy_with_bonuses(self.evaluation_df, self.prints)
        self.competition_prediction_accuracy = reval.classified_regression_competition_accuracy(self.evaluation_df, self.prints)

    def evaluation_plots(self):
        self.cm = reval.classified_regression_cm(self.evaluation_df, self.plots)

    def evaluation_details(self):
        if self.details:
            reval.classified_regression_report(self.evaluation_df)

            reval.analyze_classified_regression_based_on_match(self.evaluation_df)
            reval.analyze_classified_regression_based_on_elo(self.evaluation_df, False)
            reval.analyze_classified_regression_based_on_rider(self.evaluation_df)

            reval.analyze_classified_regression(self.evaluation_df, self.plots)
