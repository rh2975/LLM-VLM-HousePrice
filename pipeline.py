import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import os
import optuna
from preprocess_data import *
from optuna_optim import objective


def run_pipeline(csv_file, image_csv):
    logger.info(f"Starting pipeline with main CSV: {os.path.abspath(csv_file)}")
    
    X, y, le, scaler = preprocess_data(csv_file, image_csv, llama_rows=7000)
    
    # Optimize LGBM hyperparameters with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=150)
    best_params = study.best_params
    logger.info(f"Best LGBM parameters: {best_params}")
    
    lgbm_model = LGBMRegressor(**best_params, random_state=42, force_col_wise=True)
    lgbm_model.fit(
        X, y,
        categorical_feature=[col for col in X.columns if col in ['sale_season', 'property_type', 'Locality_x']]
    )
    
    y_pred = lgbm_model.predict(X)
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred))) / np.mean(np.expm1(y))
    mae = mean_absolute_error(np.expm1(y), np.expm1(y_pred)) / np.mean(np.expm1(y))
    
    logger.info("\nLightGBM Metrics:")
    logger.info(f"RMSE: ${rmse:,.2f}")
    logger.info(f"MAE: ${mae:,.2f}")

    results_df = pd.DataFrame({
        'Actual': np.expm1(y),
        'Predicted': np.expm1(y_pred)
    })
    
    logger.info("\nActual vs Predicted (First 10):")
    logger.info(results_df.head(10).to_string(index=False))
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': lgbm_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    logger.info("\nTop 10 Feature Importances:")
    logger.info(importance.head(10).to_string(index=False))
    
    low_importance = importance[importance['Importance'] < importance['Importance'].quantile(0.2)]['Feature'].tolist()
    logger.info(f"Dropping {len(low_importance)} low-importance features: {low_importance}")
    X = X.drop(columns=low_importance, errors='ignore')
    
    lgbm_model.fit(
        X, y,
        categorical_feature=[col for col in X.columns if col in ['sale_season', 'property_type', 'Locality_x']]
    )
    y_pred = lgbm_model.predict(X)
    rmse = np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred))) / np.mean(np.expm1(y))
    mae = mean_absolute_error(np.expm1(y), np.expm1(y_pred)) / np.mean(np.expm1(y))
    
    logger.info("\nLightGBM Metrics (After Dropping Low-Importance Features):")
    logger.info(f"RMSE: ${rmse:,.2f}")
    logger.info(f"MAE: ${mae:,.2f}")


if __name__ == "__main__":
    try:
        csv_file = "input_file.csv"
        image_csv = "all_the_image.csv"
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Main dataset not found: {os.path.abspath(csv_file)}")
        if not os.path.exists(image_csv):
            raise FileNotFoundError(f"Image metadata not found: {os.path.abspath(image_csv)}")
        if not os.path.exists(CFG.image_path):
            raise FileNotFoundError(f"Image directory not found: {CFG.image_path}")

        run_pipeline(csv_file, image_csv)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise