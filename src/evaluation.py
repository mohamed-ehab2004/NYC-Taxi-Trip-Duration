import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error


def evaluate_model(y_true, y_pred, model_name='model'):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmsle = mean_squared_log_error(y_true, y_pred) ** 0.5

    return pd.Series(
        {
            'model': model_name,
            'R2': r2,
            'RMSE': mse,
            'RMSLE': rmsle,
        }
    )
