import optuna
import optuna.visualization.matplotlib as opplt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from Models.variables import MT_DATASET
from Models.utils import drop_columns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score


def handle_categoricals(df):
    df = drop_columns(df)

    categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns)

    return df

dataset = pd.read_parquet(f"../Models/Datasets/{MT_DATASET}")

dataset = dataset[dataset['Letter'].isna()]

test_range = [2024]
years_in_df = dataset['Year'].unique().tolist()
train_range = [year for year in years_in_df if year not in test_range]

X = dataset.drop(axis=1, columns=['Rider_points'])
y = dataset[['Year', 'Rider_points']]

X_train = X[X['Year'].isin(train_range)]
X_test = X[X['Year'].isin(test_range)]
y_train = y[y['Year'].isin(train_range)]
y_test = y[y['Year'].isin(test_range)]

y_train = y_train.loc[:, y_train.columns != 'Year']
y_test = y_test.loc[:, y_test.columns != 'Year']

X_train = handle_categoricals(X_train)
X_test = handle_categoricals(X_test)

def objective_rf_clf(trial):
    param_space = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "max_depth": trial.suggest_int('max_depth', 2, 15),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 5),
        "max_features": trial.suggest_int('max_features', 5, 50)
    }
    clf = RandomForestClassifier(**param_space)
    # Ocena przy użyciu CV:
    score = cross_val_score(
        clf, X_train, y_train.values.ravel(), cv=5, scoring='accuracy'
    ).mean()

    return score  # Optuna będzie to maksymalizować

def objective_rf_reg(trial):
    param_space = {
        "n_estimators": trial.suggest_int('n_estimators', 50, 300),
        "max_depth": trial.suggest_int('max_depth', 2, 15),
        "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 5),
        "max_features": trial.suggest_int('max_features', 5, 50)
    }
    reg = RandomForestRegressor(**param_space)
    # Ocena przy użyciu CV:
    score = cross_val_score(reg, X_train, y_train.values.ravel(), cv=5, scoring='neg_mean_absolute_error').mean()
    return score  # Optuna będzie to maksymalizować

def objective_xgb_clf(trial):
    param_space = {
        "enable_categorical": True,
        "verbosity": 0,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 6.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_leaves": trial.suggest_int("max_leaves", 2, 100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "reg_alpha": trial.suggest_int("reg_alpha", 10, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0)
    }
    clf = XGBClassifier(**param_space)
    # Ocena przy użyciu CV:
    score = cross_val_score(clf, X_train, y_train.values.ravel(), cv=5, scoring='accuracy').mean()
    return score  # Optuna będzie to maksymalizować

def objective_xgb_reg(trial):
    param_space = {
        "enable_categorical": True,
        "verbosity": 0,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 6.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "max_leaves": trial.suggest_int("max_leaves", 2, 100),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "reg_alpha": trial.suggest_int("reg_alpha", 10, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 2.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0)
    }
    reg = XGBRegressor(**param_space)
    # Ocena przy użyciu CV:
    score = cross_val_score(reg, X_train, y_train.values.ravel(), cv=5, scoring='neg_mean_absolute_error').mean()
    return score  # Optuna będzie to maksymalizować

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Classification RF
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train.values.ravel())
pred = rf_model.predict(X_test)
print(f"Accuracy of RF model with default hyperparameters: {accuracy_score(y_test, pred):.3f}")

study = optuna.create_study(direction='maximize')  # np. dla accuracy
study.optimize(objective_rf_clf, n_trials=50, show_progress_bar=True, n_jobs=-1)  # liczba prób (kombinacji hiperparametrów)

print("Najlepsze hiperparametry: ", study.best_params)
best_model = RandomForestClassifier(**study.best_params)
best_model.fit(X_train, y_train.values.ravel())
pred = best_model.predict(X_test)
print(f"Accuracy of RF model with hyperparameters from Optuna: {accuracy_score(y_test, pred):.3f}")

opplt.plot_optimization_history(study)
plt.show()
opplt.plot_param_importances(study)
plt.show()

# Regression RF
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train.values.ravel())
pred = rf_model.predict(X_test)
print(f"MAE of RF model with default hyperparameters: {mean_absolute_error(y_test, pred):.3f}")

study = optuna.create_study(direction='maximize')  # np. dla mae
study.optimize(objective_rf_reg, n_trials=50, show_progress_bar=True, n_jobs=-1)  # liczba prób (kombinacji hiperparametrów)

print("Najlepsze hiperparametry: ", study.best_params)
best_model = RandomForestRegressor(**study.best_params)
best_model.fit(X_train, y_train.values.ravel())
pred = best_model.predict(X_test)
print(f"MAE of RF model with hyperparameters from Optuna: {mean_absolute_error(y_test, pred):.3f}")

opplt.plot_optimization_history(study)
plt.show()
opplt.plot_param_importances(study)
plt.show()

# Classification XGB
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train.values.ravel())
pred = xgb_model.predict(X_test)
print(f"Accuracy of XGB model with default hyperparameters: {accuracy_score(y_test, pred):.3f}")

study = optuna.create_study(direction='maximize')  # np. dla accuracy
study.optimize(objective_xgb_clf, n_trials=50, show_progress_bar=True, n_jobs=-1)  # liczba prób (kombinacji hiperparametrów)

print("Najlepsze hiperparametry: ", study.best_params)
best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train.values.ravel())
pred = best_model.predict(X_test)
print(f"Accuracy of XGB model with hyperparameters from Optuna: {accuracy_score(y_test, pred):.3f}")

opplt.plot_optimization_history(study)
plt.show()
opplt.plot_param_importances(study)
plt.show()

# Regression XGB
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train.values.ravel())
pred = xgb_model.predict(X_test)
print(f"MAE of XGB model with default hyperparameters: {mean_absolute_error(y_test, pred):.3f}")

study = optuna.create_study(direction='maximize')  # np. dla mae
study.optimize(objective_xgb_reg, n_trials=50, show_progress_bar=True, n_jobs=-1)  # liczba prób (kombinacji hiperparametrów)

print("Najlepsze hiperparametry: ", study.best_params)
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, y_train.values.ravel())
pred = best_model.predict(X_test)
print(f"MAE of XGB model with hyperparameters from Optuna: {mean_absolute_error(y_test, pred):.3f}")

opplt.plot_optimization_history(study)
plt.show()
opplt.plot_param_importances(study)
plt.show()
