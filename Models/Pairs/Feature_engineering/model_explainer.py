import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from Models.Pairs.pairs_xgboost_model import TrainXgBoost
from Models.variables import PAIRS_DATASET


class ExplainModel:
    def __init__(self, trained_model):
        self.model_obj = trained_model

        plt.style.use("default")

        # Importance metrics
        print("Proceeding to importance metrics...")
        self.importance_weight, self.importance_gain = self.extract_feature_importance()
        self.fill_missing_features()

        self.importances_df = self.combine_importance_types_into_dataframe()
        self.importances_heatmap = self.get_importances_heatmap()

        # Shap features
        print("Proceeding to shap summary...")
        self.shap_summary = self.get_shap_summary()

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
        weight = self.model_obj.model.get_booster().get_score(importance_type='weight')
        gain = self.model_obj.model.get_booster().get_score(importance_type='gain')

        return weight, gain

    def fill_missing_features(self):
        """
        Na wypadek wystąpienia cechy, której weight == 0
        """
        all_features = list(self.model_obj.X_train.columns)
        if all_features == set(self.importance_weight.keys()) and all_features == set(self.importance_gain.keys()):
            return
        else:
            missing_features_weight = set(all_features).difference(set(self.importance_weight))
            missing_features_gain = set(all_features).difference(set(self.importance_gain))
            for feature in missing_features_weight:
                self.importance_weight[feature] = 0
            for feature in missing_features_gain:
                self.importance_gain[feature] = 0

    def combine_importance_types_into_dataframe(self, sort_by='Geometric_mean'):
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
        fig, ax = plt.subplots(figsize=(10, 15))
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
        explainer = shap.Explainer(self.model_obj.model)
        shap_values = explainer(self.model_obj.X_train)

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=80, show=False, plot_size=(10, 15))
        fig.tight_layout()

        return fig

    def get_correlation_heatmap(self):
        heatmap_df = self.model_obj.X_train.copy()
        # Zamiana zmiennej kategorycznej 'Gate' na liczby w lokalnym df
        heatmap_df['Rider_in_home'] = heatmap_df['Rider_in_home'].map({'1': 1, '0': 0})
        heatmap_df['Opponent_in_home'] = heatmap_df['Opponent_in_home'].map({'1': 1, '0': 0})
        heatmap_df['Rider_gate'] = heatmap_df['Rider_gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        heatmap_df['Opponent_gate'] = heatmap_df['Opponent_gate'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})

        fig, ax = plt.subplots(figsize=(20, 20))
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
        result = permutation_importance(self.model_obj.model, self.model_obj.X_train, self.model_obj.y_train, scoring='r2', n_jobs=4)
        importance_df = pd.DataFrame({'Feature': self.model_obj.X_train.columns, 'Importance': result.importances_mean})
        importance_df.sort_values(by="Importance", ascending=False, inplace=True)

        fig, ax = plt.subplots(figsize=(18, 10))

        sns.barplot(data=importance_df, x="Importance", y="Feature")
        ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, color='0')

        return fig


if __name__ == '__main__':
    dataset = pd.read_parquet(f"../../../Dataset/Datasets/{PAIRS_DATASET}")
    model = TrainXgBoost(dataset, test_range=[2024], hyperparameters=True, prints=False, plots=False, details=False)
    model.create_classification_model()

    em = ExplainModel(trained_model=model)

    em.importances_heatmap.show()
    em.shap_summary.show()
    em.correlation_heatmap.show()
    em.permutation_importance.show()
