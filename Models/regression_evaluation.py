import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report, mean_absolute_error,
)

plt.style.use("dark_background")


def classified_regression_mae(results, predictions, display=True):
    mae = mean_absolute_error(results, predictions)

    if display:
        print(f"Absolute mean error of classic regression: {mae:.4f}")

    return mae


def classified_regression_rps(predicted_df, display=True):
    classes = [0, 1, 2, 3]
    rps_list = []

    for idx, row in predicted_df.iterrows():
        pred_val = row['Predicted_points_integer']
        logits = [-((pred_val - c) ** 2) for c in classes]
        pred_probs = softmax(logits)
        #  print(f"pred_val: {pred_val}, logits: {logits}, pred_probs: {pred_probs}")

        actual = [0, 0, 0, 0]
        actual[int(row['Rider_points'])] = 1

        cum_pred = np.cumsum(pred_probs)
        cum_actual = np.cumsum(actual)

        rps = np.sum((cum_pred - cum_actual) ** 2)
        rps_list.append(rps)

    overall_rps = np.mean(rps_list)

    if display:
        print(f"RPS based on classified regression: {overall_rps:.4f}")

    return overall_rps


def classified_regression_accuracy(predicted_df, display=True):
    accuracy = accuracy_score(
        predicted_df["Rider_points"], predicted_df["Predicted_points_integer"]
    )

    if display:
        print(
            f"Classified regression accuracy: {accuracy * 100:.3f}% ({len(predicted_df['Rider_points']) * accuracy:.0f}/{len(predicted_df['Rider_points'])})"
        )

    return accuracy


def classified_regression_accuracy_with_bonuses(predicted_df, display=True):
    accuracy = accuracy_score(
        predicted_df["Rider_points"],
        predicted_df["Predicted_points_integer_with_bonus"],
    )

    if display:
        print(
            f"Classified regression accuracy (with bonuses): {accuracy * 100:.3f}% ({len(predicted_df['Rider_points']) * accuracy:.0f}/{len(predicted_df['Rider_points'])})"
        )

    return accuracy


def classified_regression_competition_accuracy(predicted_df, display=True):
    df = (
        predicted_df.groupby(["Competition_ID", "Team_ID", "In_home"], observed=False)
        .agg({"Rider_points": "sum", "Predicted_points_integer": "sum"})
        .reset_index()
    )

    team1_df = df[df['In_home'] == 1].copy()
    team2_df = df[df['In_home'] == 0].copy()

    team1_df = team1_df.rename(columns={
        'Team_ID': 'Team1_ID',
        'Rider_points': 'Team1_actual_score',
        'Predicted_points_integer': 'Team1_predicted_score'
    })

    team2_df = team2_df.rename(columns={
        'Team_ID': 'Team2_ID',
        'Rider_points': 'Team2_actual_score',
        'Predicted_points_integer': 'Team2_predicted_score'
    })

    competition_df = pd.merge(team1_df[['Competition_ID', 'Team1_ID', 'Team1_actual_score', 'Team1_predicted_score']],
                              team2_df[['Competition_ID', 'Team2_ID', 'Team2_actual_score', 'Team2_predicted_score']],
                              on='Competition_ID')

    def get_actual_winner(row):
        if row['Team1_actual_score'] > row['Team2_actual_score']:
            return row['Team1_ID']
        elif row['Team1_actual_score'] < row['Team2_actual_score']:
            return row['Team2_ID']
        else:
            return 'Draw'

    def get_predicted_winner(row):
        if row['Team1_predicted_score'] > row['Team2_predicted_score']:
            return row['Team1_ID']
        elif row['Team1_predicted_score'] < row['Team2_predicted_score']:
            return row['Team2_ID']
        else:
            return 'Draw'

    competition_df['Actual_winner'] = competition_df.apply(get_actual_winner, axis=1)
    competition_df['Predicted_winner'] = competition_df.apply(get_predicted_winner, axis=1)
    competition_df['Is_accurate'] = competition_df['Actual_winner'] == competition_df['Predicted_winner']

    #  print(competition_df.to_string())
    accuracy = competition_df['Is_accurate'].mean()

    if display:
        print(f"Classified regression competition winner prediction accuracy: {accuracy * 100:.3f}%")

    return accuracy


def classified_regression_cm(predicted_df, display=True):
    cm = confusion_matrix(
        predicted_df["Rider_points"],
        predicted_df["Predicted_points_integer_with_bonus"],
    )

    if display:
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()

    return cm


def classified_regression_report(predicted_df):
    report = classification_report(
        predicted_df["Rider_points"],
        predicted_df["Predicted_points_integer_with_bonus"],
    )

    print(f"Classified regression report:\n{report}")

    return report


def analyze_classified_regression_based_on_match(predicted_df):
    df = predicted_df.groupby(["Competition_ID", "Game_time"], as_index=False).agg(
        duels=("Competition_ID", "size"), accuracy=("Is_accurate_sum", "mean")
    )

    df = df.sort_values(by="accuracy", ascending=False)

    print(f"Accuracy by match:\n{df.to_string()}")


def analyze_classified_regression_based_on_elo(predicted_df, print_details=False):
    df = predicted_df

    bins = [0, 1000, 1200, 1400, 1600, 1800, float('inf')]
    labels = ['[0,1000)', '[1000,1200)', '[1200,1400)', '[1400,1600)', '[1600,1800)', '[1800+)']
    df['ELO_bin'] = pd.cut(
        df['Rider_ELO'],
        bins=bins, labels=labels, right=False)

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


def analyze_classified_regression_based_on_rider(predicted_df):
    df = predicted_df.groupby(['Rider_ID', 'Name', 'Surname'], as_index=False).agg(
        duels=('Rider_points', 'size'),
        accuracy=('Is_accurate_sum', 'mean')
    )

    df = df.sort_values(by='accuracy', ascending=False)
    print(f"Accuracy by rider:\n{df.to_string()}")


def analyze_classified_regression(predicted_df, display_plots=False):
    """
    Ta metoda istnieje tylko w celu debugowania, weryfikacji i zrozumienia wyników. Pozniej do usuniecia.
    """

    # print(f"Number of values in Predicted_points_integer: (should be almost even distributed) {predicted_heats['Predicted_points_integer'].value_counts()}")
    # print(f"Number of values in Points: (should be almost even distributed) {predicted_heats['Points'].value_counts()}")

    predicted_df['Rider_points'] = predicted_df.pop('Rider_points')

    columns_to_remove = [
        'Game_time', 'Year', 'Track_ID', 'Rider_ID', 'Heat_ID', 'Heat_run_no',
        'Result_ID', 'Bonus_point', 'Letter', 'Substituted', 'In_home',
        'ELO_ranking_ID', 'Heats_since_track_equation'
    ]
    riders = ['Teammate', 'Rival_1', 'Rival_2']

    attributes = [
        ('ELO', 'ELO'),
        ('ID', 'ID'),
        ('3_heat_avg', '3_heat_avg'),
        ('5_heat_avg', '5_heat_avg'),
        ('10_heat_avg', '10_heat_avg'),
        ('season_heats_no', 'Season_heats_no'),
        ('overall_heats_no', 'Overall_heats_no'),
        ('comp_avg', 'Comp_avg'),
        ('comp_sum', 'Comp_sum'),
        ('season_avg', 'Season_avg'),
        ('season_sum', 'Season_sum'),
        ('gate_avg_ovr', 'Gate_avg_ovr'),
        ('gate_avg_year', 'Gate_avg_year'),
        ('track_avg_ovr', 'Track_avg_ovr')
    ]

    for output_name, ds_column in attributes:
        if ds_column not in ["ID", "ELO"]:
            columns_to_remove.append(str(f"{ds_column}"))
        for rider_name in riders:
            columns_to_remove.append(str(f"{rider_name}_{output_name}"))

    predicted_df = predicted_df.drop(columns=columns_to_remove)

    grouped = list(predicted_df.groupby(['Competition_ID', 'Heat_number']))

    print_random_predicted_heats(grouped, 5)

    if display_plots:
        predicted_points_histogram(predicted_df)
        predicted_points_in_group_avg(predicted_df)


def print_random_predicted_heats(grouped_df, no_of_prints):
    groups_to_print = random.sample(grouped_df, no_of_prints)

    for (competition_id, heat_number), group in groups_to_print:
        print(f"\nCompetition ID: {competition_id} | Heat number: {heat_number}")
        print(group.drop(columns=['Competition_ID', 'Heat_number']).to_string(index=False))


def predicted_points_histogram(predicted_df):
    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        predicted_df["Predicted_points"],
        bins=20,
        color="blue",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set(
        xlabel="Predicted_Points",
        ylabel="Count",
        title="Histogram wartości Predicted Points",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def predicted_points_in_group_avg(predicted_df):
    avg_predicted_points_by_group = predicted_df.groupby(
        ["Competition_ID", "Heat_number"]
    )["Predicted_points"].mean()

    _, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        avg_predicted_points_by_group,
        bins=20,
        color="green",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set(
        xlabel="Średnia Predicted Points w grupie",
        ylabel="Liczba grup",
        title="Histogram średnich wartości Predicted Points w biegu",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    average_predicted_points = predicted_df['Predicted_points'].mean()
    print(f"\nŚrednia wartość Predicted_points: {average_predicted_points:.2f}\n")
