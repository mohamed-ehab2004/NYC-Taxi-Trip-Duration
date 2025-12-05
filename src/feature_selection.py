import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pipeline import run_pipeline
from data_load import load_data, Outliers_handling
from feature_engineering import prepare_data
from pipeline import encoding

def run_feature_selection():
    """
    Very simple feature selection using RandomForest importance.
    Returns list of top_n feature names.
    """
    # 1) Load & clean
    train, _ = load_data()  # we only need training data for feature importance
    train = Outliers_handling(train)

    # 2) Feature engineering (haversine, bearing, datetime, etc.)
    train = prepare_data(train)

    # 3) One-hot encode and keep numerical features
    train_full, _ = encoding(train, train, use_all_features=True)
    train_full = train_full.select_dtypes(include=['number']).copy()

    # 4) Split X / y (log_trip_duration is the target)
    feature_cols = [c for c in train_full.columns if c != 'log_trip_duration']
    X_train = train_full[feature_cols]
    y_train = train_full['log_trip_duration']


    model = RandomForestRegressor(
        n_estimators=80, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    fi = pd.DataFrame(
        {'Feature': X_train.columns, 'Importance': model.feature_importances_}
    ).sort_values('Importance', ascending=False)

    print(fi.head(30))
