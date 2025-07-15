import matplotlib.pyplot as plt
import pandas as pd
import shap

from Models.Classic.xgboost_model import TrainXgBoost
from Models.variables import MODELS_DATASET


class DependenciesPlots(TrainXgBoost):
    def __init__(self, data, model_with_hyperparameters=False):
        super().__init__(data, test_range=[2024], hyperparameters=model_with_hyperparameters, prints=False)
        self.create_regression_model()

        plt.style.use("default")

        # Dependencies
        print("Proceeding to dependencies...")
        self.dependencies_plots = self.get_dependencies_plots()

    def get_dependencies_plots(self):
        X = pd.concat([self.X_train, self.X_test], axis=0)
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X)

        # features_to_plot = list(X.columns)  # could be a list with chosen attribute names
        features_to_plot = ["Rider_ELO", "Heat_number", "Gate", "In_home", "3_heat_avg", "5_heat_avg",
                            "10_heat_avg", "Season_heats_no", "Overall_heats_no", "Comp_avg", "Comp_sum",
                            "Season_avg", "Season_sum", "Gate_avg_ovr", "Gate_avg_year", "Track_avg_ovr",
                            "3_match_avg", "5_match_avg", "Home_away_track_avg_year", "Home_away_track_avg_ovr",
                            "Previous_start", "Previous_home_away_start", "Previous_track_start",
                            "Team_pts_sum", "Teams_pts_diff", "Heats_since_track_equation",
                            "Ovr_track_gate_avg_ovr", "Ovr_track_gate_avg_year", "Win_ratio_track_gate",
                            "Win_ratio_rider_gate", "Win_ratio_rider_track_gate"]

        figures = []

        for i in range(0, len(features_to_plot), 6):
            subset = features_to_plot[i:i + 6]

            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            for j, feature in enumerate(subset):
                shap.plots.scatter(shap_values[:, feature], ax=axs[j], show=False, color=shap_values[:, feature])
                axs[j].set_title(feature)
                axs[j].set_ylim([-0.5, 0.5])

            for k in range(len(subset), len(axs)):
                axs[k].axis('off')

            plt.tight_layout()
            figures.append(fig)

        return figures


if __name__ == "__main__":
    dataset = pd.read_parquet(f"../../../Dataset/Datasets/{MODELS_DATASET}")
    plots = DependenciesPlots(dataset)
    for chart in plots.dependencies_plots:
        chart.show()

    plt.show()
