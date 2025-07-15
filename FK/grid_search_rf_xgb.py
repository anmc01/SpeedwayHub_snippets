import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from Models.variables import MT_DATASET
from Models.utils import drop_columns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def handle_categoricals(df):
    df = drop_columns(df)

    #categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
    #df = pd.get_dummies(df, columns=categorical_columns)

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

# Classification RF

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train.values.ravel())
pred = rf_model.predict(X_test)
print(f"Accuracy of RF model with default hyperparameters: {accuracy_score(y_test, pred):.3f}")

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 3, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [10, 20]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train.values.ravel())

print(f"Best hyperparameters found: {grid_rf.best_params_}")
rf_model_hyper = RandomForestClassifier(**grid_rf.best_params_)
rf_model_hyper.fit(X_train, y_train.values.ravel())
pred_hyper = rf_model_hyper.predict(X_test)

print(f"Accuracy of RF model with hyperparameters from Grid Search: {accuracy_score(y_test, pred_hyper):.3f}")

# Regression RF
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train.values.ravel())
pred = rf_model.predict(X_test)
print(f"MAE of RF model with default hyperparameters: {mean_absolute_error(y_test, pred):.3f}")

grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train.values.ravel())

print(f"Best hyperparameters found: {grid_rf.best_params_}")
rf_model_hyper = RandomForestRegressor(**grid_rf.best_params_)
rf_model_hyper.fit(X_train, y_train.values.ravel())
pred_hyper = rf_model_hyper.predict(X_test)

print(f"MAE of RF model with hyperparameters from Grid Search: {mean_absolute_error(y_test, pred_hyper):.3f}")

# Classification XGB

xgb_model = XGBClassifier(colsample_bytree=0.6, learning_rate=0.01, max_depth=10, min_child_weight=10, subsample=0.8)
xgb_model.fit(X_train, y_train.values.ravel())
pred = xgb_model.predict(X_test)
print(f"Accuracy of XGB model with default hyperparameters: {accuracy_score(y_test, pred):.3f}")

param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 5, 10]
}

xgb_model = XGBClassifier()
grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
    cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_xgb.fit(X_train, y_train.values.ravel())

print(f"Best hyperparameters found: {grid_xgb.best_params_}")
xgb_model_hyper = XGBClassifier(**grid_xgb.best_params_)
xgb_model_hyper.fit(X_train, y_train.values.ravel())
pred_hyper = xgb_model_hyper.predict(X_test)

print(f"Accuracy of XGB model with hyperparameters from Grid Search: {accuracy_score(y_test, pred_hyper):.3f}")

# Regression XGB

xgb_model = XGBRegressor(enable_categorical=True)
xgb_model.fit(X_train, y_train.values.ravel())
pred = xgb_model.predict(X_test)
print(f"MAE of XGB model with default hyperparameters: {mean_absolute_error(y_test, pred):.3f}")

xgb_model = XGBRegressor(enable_categorical=True)
grid_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
    cv=5, n_jobs=-1, verbose=1
)
grid_xgb.fit(X_train, y_train.values.ravel())

print(f"Best hyperparameters found: {grid_xgb.best_params_}")
xgb_model_hyper = XGBRegressor(**grid_xgb.best_params_, enable_categorical=True)
xgb_model_hyper.fit(X_train, y_train.values.ravel())
pred_hyper = xgb_model_hyper.predict(X_test)

print(f"MAE of XGB model with hyperparameters from Grid Search: {mean_absolute_error(y_test, pred_hyper):.3f}")
