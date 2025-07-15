import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score)

plt.style.use("dark_background")


def classification_accuracy(predicted_df, result, prediction, comment="", display=True):
    accuracy = accuracy_score(predicted_df[result], predicted_df[prediction])
    all_records = len(predicted_df[result])

    if len(str(comment)) > 0:
        comment = " (" + str(comment) + ")"

    if display:
        print(f"Classification accuracy{comment}: {accuracy * 100:.3f}% ({all_records * accuracy:.0f}/{all_records})")

    return accuracy


def classification_importance(model, importance_type='gain', display=True):
    fig, ax = None, None

    if display:
        fig, ax = plt.subplots(figsize=(20, 14))
        xgb.plot_importance(
            model,
            title=f"Feature importances ({importance_type}) - XGBoost",
            ax=ax,
            importance_type=importance_type
        )
        plt.tight_layout()
        plt.show()

    return fig


def classification_cm(predicted_df, display=True):
    cm = confusion_matrix(predicted_df["Rider_points"], predicted_df["Predicted_points_integer_with_bonus"])

    if display:
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()

    return cm


def analyze_classification_based_on_elo(predicted_df, print_details=False):
    df = predicted_df

    bins = [0, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, float('inf')]
    labels = [
        '[0, 1000)', '[1000, 1100)', '[1100, 1200)',
        '[1200, 1300)', '[1300, 1400)', '[1400, 1500)',
        '[1500, 1600)', '[1600, 1700)', '[1700, 1800)', '[1800+)'
    ]

    df['ELO_bin'] = pd.cut(df['Rider_ELO'], bins=bins, labels=labels, right=False)

    accuracy_by_ELO = df.groupby('ELO_bin', observed=True).agg(
        duels=('Rider_points', 'size'),
        accuracy=('Is_accurate_sum', 'mean')
    )

    print(f"{accuracy_by_ELO}\n")

    if print_details:
        for elo_interval, group in df.groupby('ELO_bin', observed=True):
            duels = group['Rider_points'].size
            accuracy = group['Is_accurate_sum'].mean()

            y_true = group['Rider_points']
            y_pred = group['Predicted_points_integer_with_bonus']
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

            print(f"ELO range: {elo_interval:11} | Duels: {duels:4} | Accuracy: {accuracy:.2%}\nConfusion Matrix: \n{cm}\n")


def analyze_classification_based_on_match(predicted_df):
    df = predicted_df.groupby(["Competition_ID"], as_index=False).agg(
        duels=("Competition_ID", "size"), accuracy=("Is_accurate_sum", "mean")
    )

    df = df.sort_values(by="accuracy", ascending=False)

    print(f"Accuracy by match:\n{df.to_string()}")
