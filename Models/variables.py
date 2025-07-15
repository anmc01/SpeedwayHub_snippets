DATA_START_DATE = '2020-01-01'
DATA_END_DATE = '2024-12-31'

COLUMNS_TO_DROP = ['Competition_ID', 'Game_time', 'Year', 'Team_ID', 'Rider_ID', 'Name', 'Surname', 'Track_ID',
                   'Heat_ID', 'Heat_run_no', 'Result_ID', 'Bonus_point', 'Letter', 'Teammate_ID', 'Rival_1_ID',
                   'Rival_2_ID', 'ELO_ranking_ID', 'Substituted']

PAIRS_COLUMNS_TO_DROP = ['Competition_ID', 'Heat_number', 'Year', 'Rider', 'Opponent', 'Rider_points', 'Opponent_points',
                         'Rider_team', 'Opponent_team', 'Rider_bonus_point', 'Opponent_bonus_point', 'Track_ID']

PAIRS_COLUMNS_TO_KEEP = ['Competition_ID', 'Heat_number', 'Rider', 'Opponent', 'Rider_points', 'Rider_ELO', 'Rider_team',
                         'Result', 'Pair_prediction', 'Pair_prediction_prob']

NAN_DATASET = "dataset_full_0427_2229_2020.parquet"
MODELS_DATASET = "dataset_full_0427_2229_2020.parquet"
HYPERPARAMETERS_DATASET = "dataset_full_0409_1810_4NN.parquet"

PAIRS_DATASET = "dataset_pairs_0526_engineered.parquet"
EVALUATION_COLUMN = "BT_prob"

HYPEROPT_EVALUATIONS = 200
OPTUNA_TRIALS = 200
OPTUNA_TIMEOUT = 300
