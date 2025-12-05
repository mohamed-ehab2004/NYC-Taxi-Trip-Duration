import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from data_load import load_data, Outliers_handling
from feature_engineering import prepare_data


def encoding(train, validation, use_all_features):
    """
    One-hot encode categoricals + select final feature set including target.
    """
    categorical_features = [
        'passenger_count',
        'vendor_id',
        'store_and_fwd_flag',
        'Working_days',
        'rush_hour',
        'month',
        'hour',
        'day',
        'day_of_week',
    ]

    enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    cat_train = pd.DataFrame(
        enc.fit_transform(train[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=train.index,
    )

    cat_val = pd.DataFrame(
        enc.transform(validation[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=validation.index,
    )

    train_full = pd.concat([train, cat_train], axis=1)
    val_full = pd.concat([validation, cat_val], axis=1)

    if use_all_features:
        # keep all features for feature selection
        return train_full, val_full


    # Cleaned feature set
    features = [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'latitude_distance',
        'longitude_distance',
        'manhattan_distance',
        'haversine_distance',
        'distance_ratio',
        'midpoint_latitude',
        'midpoint_longitude',
        'bearing',
        'vendor_id_2',
        'Working_days_1',
        'rush_hour_1',
        'month_2',
        'month_3',
        'month_4',
        'month_5',
        'month_6',
        'hour_sin',
        'hour_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'day_sin',
        'day_cos',
        'bearing_sin',
        'bearing_cos',
        'log_trip_duration',
    ]

    df_train = train_full[features]
    df_val = val_full[features]

    return df_train, df_val


def preparedata(train, test, processor):
    """
    Convert to numpy, split into X/y and optionally scale.
    processor:
        0 -> no scaling (tree models)
        1 -> StandardScaler
        2 -> MinMaxScaler
    """
    train = train.to_numpy()
    test = test.to_numpy()

    x_train = train[:, :-1]
    t_train = train[:, -1]

    x_test = test[:, :-1]
    t_test = test[:, -1]

    scaler = None
    if processor == 1:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif processor == 2:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, t_train, x_test, t_test


def run_pipeline(processor, use_all_features):
    """
    Full pipeline:
      load → outliers → feature engineering → encoding → X/y split
    Returns:
      X_train, y_train, X_val, y_val, feature_names
    """
    train, validation = load_data()

    print(f'Training data before processed: {train.shape}')
    print(f'Validation data before processed: {validation.shape}')

    train = Outliers_handling(train)
    validation = Outliers_handling(validation)

    train = prepare_data(train)
    validation = prepare_data(validation)

    train_encoded, val_encoded = encoding(train, validation, use_all_features)
    feature_names = list(train_encoded.columns[:-1])

    X_train, y_train, X_val, y_val = preparedata(
        train_encoded, val_encoded, processor=processor
    )

    print(f'Training data after processed: {train_encoded.shape}')
    print(f'Validation data after processed: {val_encoded.shape}')

    return X_train, y_train, X_val, y_val, feature_names
