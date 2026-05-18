import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

def objective(trial, X, y):
    """Optuna Objective Function"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'early_stopping_rounds': 50
    }
    model = LGBMRegressor(**params, random_state=42, force_col_wise=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            categorical_feature=[col for col in X.columns if col in ['sale_season', 'property_type', 'Locality_x']]
        )
        y_pred = model.predict(X_val)
        scores.append(mean_absolute_error(np.expm1(y_val), np.expm1(y_pred)))
    return np.mean(scores)
