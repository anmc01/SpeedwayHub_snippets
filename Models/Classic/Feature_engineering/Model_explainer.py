import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from Models.variables import MODELS_DATASET
from Models.Classic.xgboost_model import TrainXgBoost

# import matplotlib
# matplotlib.use('TkAgg')


class ExplainModel(TrainXgBoost):
    def __init__(self, data, model_with_hyperparameters=False):
        super().__init__(data, test_range=[2024], hyperparameters=model_with_hyperparameters, prints=False)
        self.create_regression_model()

        plt.style.use("default")

        # Importance metrics
        print("Proceeding to importance metrics...")
        self.importance_weight, self.importance_gain = self.extract_feature_importance()
        self.fill_missing_features()
        self.normalize_importance_types()

        self.importances_df = self.combine_importance_types_into_dataframe()
        self.importances_heatmap = self.get_importances_heatmap()

        # Shap features
        print("Proceeding to shap features...")
        self.shap_summary = self.get_shap_summary()
        self.prediction_charts = self.analyze_single_prediction()

        # Correlation heatmap
        print("Proceeding to correlations...")
        self.correlation_heatmap = self.get_correlation_heatmap()

        # Permutation importance
        print("Proceeding to permutations...")
        self.permutation_importance = self.get_permutation_importance()

    def extract_feature_importance(self):
        """
        weight — liczba podziałów ze względu na daną cechę
        gain — średni wzrost precyzji predykcji po podziale ze wzgledu na daną cechę
        """
        weight = self.model.get_booster().get_score(importance_type='weight')
        gain = self.model.get_booster().get_score(importance_type='gain')

        return weight, gain

    def fill_missing_features(self):
        """
        Na wypadek wystąpienia cechy, której weight == 0
        """
        all_features = list(self.df.drop(columns=['Rider_points', 'Year']).columns)
        if all_features == set(self.importance_weight.keys()) and all_features == set(self.importance_gain.keys()):
            return
        else:
            missing_features_weight = set(all_features).difference(set(self.importance_weight))
            missing_features_gain = set(all_features).difference(set(self.importance_gain))
            for feature in missing_features_weight:
                self.importance_weight[feature] = 0
            for feature in missing_features_gain:
                self.importance_gain[feature] = 0

    def normalize_importance_types(self):
        """
        Liniowa normalizacja ważności do skali od 0 do 100.
        max(weight) = 100 - reszta jest procentową wartoscią max (tak samo dla gain)
        """
        max_weight = max(self.importance_weight.values())
        max_gain = max(self.importance_gain.values())

        for feature in self.importance_weight:
            self.importance_weight[feature] = self.importance_weight[feature] / max_weight * 100
        for feature in self.importance_gain:
            self.importance_gain[feature] = self.importance_gain[feature] / max_gain * 100

    def combine_importance_types_into_dataframe(self, sort_by='Gain'):
        """
        geometric mean - całkowity wklad cechy do modelu. Wzór: sqrt(weight*gain)
        """
        importances_df = pd.DataFrame({
            'Feature': list(self.importance_weight.keys()),
            'Weight': list(self.importance_weight.values()),
            'Gain': list(self.importance_gain.values())
        })

        importances_df['Geometric_mean'] = (importances_df['Weight'] * importances_df['Gain']) ** 0.5
        importances_df.sort_values(by=[sort_by], ascending=False, inplace=True)

        return importances_df

    def get_importances_heatmap(self):
        fig, ax = plt.subplots(figsize=(8, 18))
        sns.heatmap(
            self.importances_df.set_index('Feature'),
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidth=.1,
            annot_kws={"size": 8, "ha": "center", "va": "center"},
            cbar=True,
            cbar_kws={"location": "right", "orientation": "vertical", "aspect": 50},
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )

        ax.set_aspect(0.2, anchor='E')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='center')

        fig.tight_layout()

        return fig

    def get_shap_summary(self):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X_train)

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=80, show=False, plot_size=(20, 30))
        fig.tight_layout()

        return fig

    def analyze_single_prediction(self, number_of_predictions=1):
        """
        Wpływ wartości cech na konkretną predykcję.
        """

        #  TODO: Dodać
        #   - prawdziwej wartosci points na wykresie,
        #   - wynik klasyfikacji regresyjnej,
        #   - Comp_ID,
        #   - Heat_ID,
        #   - Name & Surname,
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_train)

        charts = []

        for i in np.random.choice(self.X_train.shape[0], size=number_of_predictions):
            print(f'Actual Points: {self.y_train.iloc[i]}')

            # Extract SHAP values and feature names
            feature_names = self.X_train.columns
            shap_values_for_instance = shap_values[i].values
            instance_data = self.X_train.iloc[i]

            # Prepare custom labels with new lines
            custom_labels = [
                f"{feature}\n{value:.2f}" for feature, value in zip(feature_names, shap_values_for_instance)
            ]

            # Create a force plot with custom labels
            charts.append(
                shap.force_plot(
                    explainer.expected_value,
                    shap_values_for_instance,
                    instance_data,
                    matplotlib=True,
                    show=False,
                    feature_names=custom_labels  # Pass new labels here
                )
            )

        return charts

    def get_correlation_heatmap(self):
        heatmap_df = self.X_train.copy()
        # Zamiana zmiennej kategorycznej 'Gate' na liczby w lokalnym df
        heatmap_df['In_home'] = heatmap_df['In_home'].map({'1': 1, '0': 0})
        heatmap_df['Gate'] = heatmap_df['Gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        heatmap_df['Teammate_gate'] = heatmap_df['Teammate_gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        heatmap_df['Rival_1_gate'] = heatmap_df['Rival_1_gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        heatmap_df['Rival_2_gate'] = heatmap_df['Rival_2_gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})

        fig, ax = plt.subplots(figsize=(35, 35))
        sns.heatmap(
            heatmap_df.corr(),
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidth=.1,
            annot_kws={"size": 8, "ha": "center", "va": "center"},
            cbar=False,
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )

        ax.set_aspect('auto', anchor='C')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        return fig

    def get_permutation_importance(self):
        """
        Wartości zmiennej są przetasowane po całym zbiorze uzącym, np. dla Rider_ELO kazdy zawodnik ma ELO losowego zawodnika.
        I tak dla każdej cechy osobno, liczona jest różnica jakości modelu przed tasowaniem i po (5 razy się liczy i uśrednia wynik).
        """
        result = permutation_importance(self.model, self.X_train, self.y_train, scoring='r2', n_jobs=4)
        importance_df = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': result.importances_mean})
        importance_df.sort_values(by="Importance", ascending=False, inplace=True)

        fig, ax = plt.subplots(figsize=(24, 18))

        sns.barplot(data=importance_df, x="Importance", y="Feature")
        ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, color='0')

        return fig


if __name__ == '__main__':
    dataset = pd.read_parquet(f"../../../Dataset/Datasets/{MODELS_DATASET}")

    em = ExplainModel(dataset, model_with_hyperparameters=True)

    em.importances_heatmap.show()
    em.shap_summary.show()
    em.correlation_heatmap.show()
    em.permutation_importance.show()

    for chart in em.prediction_charts:
        chart.show()

    plt.show()
