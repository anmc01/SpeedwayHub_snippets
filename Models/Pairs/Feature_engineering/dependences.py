import matplotlib.pyplot as plt
import pandas as pd
import shap

from Models.Pairs.pairs_xgboost_model import TrainXgBoost
from Models.variables import PAIRS_DATASET


class DependenciesPlots:
    def __init__(self, trained_model, features_to_plot):
        self.model_obj = trained_model

        print("Creating explainer...")
        X = pd.concat([self.model_obj.X_train, self.model_obj.X_test], axis=0)
        explainer = shap.Explainer(self.model_obj.model)
        self.shap_values = explainer(X)

        plt.style.use("default")
        self.features_to_plot = features_to_plot

        print("Proceeding to dependencies...")
        self.dependencies_plots = self.get_dependencies_plots()

    def get_dependencies_plots(self):
        figures = []

        print("Proceeding to plotting...")

        for i, feature in enumerate(self.features_to_plot):
            print(f"Creating plot {i + 1}/{len(self.features_to_plot)} for feature: {feature}...")

            fig, ax = plt.subplots(figsize=(6, 5))

            shap_vals = self.shap_values[:, feature]
            shap.plots.scatter(shap_vals, ax=ax, show=False, color=shap_vals)
            ax.set_title(feature)
            ax.set_ylim([-0.7, 0.7])

            max_val = shap_vals.max(axis=0).values
            min_val = shap_vals.min(axis=0).values

            max_color = 'green' if max_val >= 0.1 else 'blue'
            min_color = 'green' if min_val <= -0.1 else 'red'

            ax.axhline(max_val, color=max_color, linestyle='--', label=f'Max: {max_val:.4f}')
            ax.axhline(min_val, color=min_color, linestyle='--', label=f'Min: {min_val:.4f}')

            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

            plt.tight_layout()
            figures.append(fig)

        return figures


if __name__ == "__main__":
    dataset = pd.read_parquet(f'../../../Dataset/Datasets/{PAIRS_DATASET}')
    print(dataset.columns.values.tolist())

    features = [
        'Rider_ELO', 'Opponent_ELO', 'Rider_in_home', 'Opponent_in_home', 'Rider_gate', 'Opponent_gate',
        'Rider_season_avg', 'Opponent_season_avg', 'Diff_season_avg', 'Diff_overall_heats_no', 'Diff_season_heats_no', 'Diff_season_sum',
        'Rider_ovr_avg', 'Opponent_ovr_avg', 'Rider_comp_sum', 'Opponent_comp_sum', 'Rider_ovr_sum', 'Opponent_ovr_sum',
        'Rider_track_avg_ovr', 'Opponent_track_avg_ovr', 'Rider_ovr_track_gate_avg_ovr', 'Opponent_ovr_track_gate_avg_ovr', 'Rider_ovr_track_gate_avg_year', 'Opponent_ovr_track_gate_avg_year',
        'Rider_win_ratio_rider_gate', 'Opponent_win_ratio_rider_gate', 'Rider_win_ratio_rider_track_gate', 'Opponent_win_ratio_rider_track_gate', 'Rider_gate_avg_ovr', 'Opponent_gate_avg_ovr',
        'Rider_3_match_avg', 'Opponent_3_match_avg', 'Rider_5_match_avg', 'Opponent_5_match_avg', 'Rider_won_duels_ovr', 'Opponent_won_duels_ovr',
        'Rider_home_away_track_avg_ovr', 'Opponent_home_away_track_avg_ovr', 'Diff_home_away_track_avg_year', 'Duels_diff_ovr', 'Rider_heats_since_track_equation', 'Opponent_heats_since_track_equation',
        'Diff_3_heat_avg', 'Diff_5_heat_avg', 'Diff_10_heat_avg', 'Diff_27_heat_avg'
    ]

    model = TrainXgBoost(dataset, test_range=[2024], hyperparameters=True, prints=False)
    model.create_classification_model()
    model.evaluate_classification()

    plots = DependenciesPlots(model, features_to_plot=features)

    for chart in plots.dependencies_plots:
        chart.show()

    plt.show()
