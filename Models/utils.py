import numpy as np
import pandas as pd

from Models.variables import COLUMNS_TO_DROP, PAIRS_COLUMNS_TO_DROP, PAIRS_COLUMNS_TO_KEEP, EVALUATION_COLUMN


def drop_columns(df):
    df = df.drop(axis=1, columns=COLUMNS_TO_DROP, errors='ignore')

    return df


def drop_pairs_columns(df):
    df = df.drop(axis=1, columns=PAIRS_COLUMNS_TO_DROP, errors='ignore')

    return df


def classify_points(predicted_group):
    evaluated_group = predicted_group.sort_values('Predicted_points', ascending=False)
    ranks = [3, 2, 1, 0, 0, 0]

    evaluated_group['Predicted_points_integer'] = ranks[:len(evaluated_group)]

    return evaluated_group


def count_bonuses_as_accurate(predicted_group):
    evaluated_group = predicted_group.sort_values('Predicted_points_integer', ascending=False)
    values = [None] * len(evaluated_group)
    rider_count = len(evaluated_group)

    if rider_count >= 2:
        riders = [evaluated_group.iloc[i] for i in range(min(3, rider_count))]

        # Check for 2-3 points bonus pair (positions 1-2)
        if riders[0]['Team_ID'] == riders[1]['Team_ID']:
            if riders[0]['Rider_points'] == 2 and riders[1]['Rider_points'] == 3:
                values[0], values[1] = True, True

        # Check for 1-2 points bonus pair (positions 2-3)
        if rider_count >= 3 and riders[1]['Team_ID'] == riders[2]['Team_ID']:
            if riders[1]['Rider_points'] == 1 and riders[2]['Rider_points'] == 2:
                values[1], values[2] = True, True

    evaluated_group['Is_accurate_with_bonus'] = values

    return evaluated_group


def prepare_evaluation_df(predicted_df):
    evaluation_df = predicted_df.groupby(['Competition_ID', 'Heat_number'], group_keys=False, observed=False).apply(classify_points, include_groups=False)
    evaluation_df['Competition_ID'] = predicted_df['Competition_ID']
    evaluation_df['Heat_number'] = predicted_df['Heat_number']

    evaluation_df['Is_accurate'] = np.where((evaluation_df['Rider_points'] == evaluation_df['Predicted_points_integer']), True, False)

    evaluation_df = evaluation_df.groupby(['Competition_ID', 'Heat_number'], group_keys=False, observed=False).apply(count_bonuses_as_accurate, include_groups=False)
    evaluation_df['Competition_ID'] = predicted_df['Competition_ID']
    evaluation_df['Heat_number'] = predicted_df['Heat_number']

    evaluation_df['Is_accurate_sum'] = False
    evaluation_df.loc[(evaluation_df['Is_accurate']) | (evaluation_df['Is_accurate_with_bonus']), 'Is_accurate_sum'] = True

    evaluation_df['Predicted_points_integer_with_bonus'] = evaluation_df['Predicted_points_integer']
    evaluation_df.loc[(evaluation_df['Is_accurate'] == False) & (evaluation_df['Is_accurate_sum']), 'Predicted_points_integer_with_bonus'] = evaluation_df['Rider_points']

    return evaluation_df


def prepare_pairs_evaluation_df(predicted_df):
    # Keeping just columns we need for evaluation
    df = predicted_df[PAIRS_COLUMNS_TO_KEEP]

    # Getting prediction for an opposite pair to enable normalization
    df_opposite_pred = df[['Competition_ID', 'Heat_number', 'Rider', 'Opponent', 'Pair_prediction_prob']].copy()
    df_opposite_pred.columns = ['Competition_ID', 'Heat_number', 'Opponent', 'Rider', 'Pair_prediction_prob_opposite']

    df = pd.merge(
        df,
        df_opposite_pred,
        on=['Competition_ID', 'Heat_number', 'Rider', 'Opponent'],
        how='left'
    )

    # Normalizing predictions
    df['Normalized_prob'] = df['Pair_prediction_prob'] / (df['Pair_prediction_prob'] + df['Pair_prediction_prob_opposite'])
    df['BT_prob'] = df.apply(lambda x: bt_aggregate(x['Pair_prediction_prob'], x['Pair_prediction_prob_opposite']), axis=1)

    # Summing predictions and leaving just one row per rider
    cols = ['Result', 'Pair_prediction', 'Pair_prediction_prob', 'Normalized_prob', 'BT_prob']
    df.loc[:, cols] = df.groupby(['Competition_ID', 'Heat_number', 'Rider'])[cols].transform(lambda x: x.sum())

    df = df.drop_duplicates(subset=['Competition_ID', 'Heat_number', 'Rider']).reset_index(drop=True)

    # Assigning points and bonus points
    evaluation_df = df.groupby(['Competition_ID', 'Heat_number'], group_keys=False, observed=False).apply(
        lambda x: classify_points_and_bonuses(x, EVALUATION_COLUMN))
    evaluation_df['Predicted_points_integer_with_bonus'] = evaluation_df.apply(
        lambda row: row['Predicted_points_integer'] if row['Is_accurate_with_bonus'] == 0 else row['Is_accurate_with_bonus'], axis=1
    ).astype(int)

    evaluation_df['Is_accurate_sum'] = evaluation_df.apply(lambda row: 1 if row['Rider_points'] == row['Predicted_points_integer_with_bonus'] else 0, axis=1)

    return evaluation_df.drop(columns=['Is_accurate_with_bonus'])


def classify_points_and_bonuses(predicted_group, column_to_classify):
    evaluated_group = predicted_group.sort_values(column_to_classify, ascending=False)

    if evaluated_group[column_to_classify].sum() > 6.1:
        raise Exception(f"{column_to_classify} sum > 6 in group! The sum: {evaluated_group[column_to_classify].sum()}\n{evaluated_group.to_string()}")

    ranks = [3, 2, 1, 0, 0, 0]

    evaluated_group['Predicted_points_integer'] = ranks[:len(evaluated_group)]

    evaluated_group = evaluated_group.sort_values('Predicted_points_integer', ascending=False)
    values = [0] * len(evaluated_group)
    rider_count = len(evaluated_group)

    if rider_count >= 2:
        riders = [evaluated_group.iloc[i] for i in range(min(3, rider_count))]

        # Check for 2-3 points bonus pair (positions 1-2)
        if riders[0]['Rider_team'] == riders[1]['Rider_team']:
            if riders[0]['Rider_points'] == 2 and riders[1]['Rider_points'] == 3:
                values[0], values[1] = 2, 3

        # Check for 1-2 points bonus pair (positions 2-3)
        if rider_count >= 3 and riders[1]['Rider_team'] == riders[2]['Rider_team']:
            if riders[1]['Rider_points'] == 1 and riders[2]['Rider_points'] == 2:
                values[1], values[2] = 1, 2

    evaluated_group['Is_accurate_with_bonus'] = values

    return evaluated_group


def bt_aggregate(p_ab, p_ba):
    numerator = p_ab / (1 - p_ab)
    denominator = numerator + (p_ba / (1 - p_ba))
    return numerator / denominator
