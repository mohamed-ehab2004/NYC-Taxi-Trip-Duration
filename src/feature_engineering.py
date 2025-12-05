import numpy as np
import pandas as pd


def haversine(p_lat, p_lon, d_lat, d_lon):
    """Great-circle distance between two coordinates in kilometers"""
    R = 6371
    p_lat = np.radians(p_lat)
    p_lon = np.radians(p_lon)
    d_lat = np.radians(d_lat)
    d_lon = np.radians(d_lon)

    avg_lat_dis = (d_lat - p_lat) / 2
    avg_lon_dis = (d_lon - p_lon) / 2

    a = (
        np.sin(avg_lat_dis) ** 2
        + np.cos(p_lat) * np.cos(d_lat) * np.sin(avg_lon_dis) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def bearing(p_lat, p_lon, d_lat, d_lon):
    """Bearing (direction) from pickup to dropoff in degrees [0, 360)"""
    p_lat = np.radians(p_lat)
    p_lon = np.radians(p_lon)
    d_lat = np.radians(d_lat)
    d_lon = np.radians(d_lon)

    x = np.sin(d_lon - p_lon) * np.cos(d_lat)
    y = np.cos(p_lat) * np.sin(d_lat) - (
        np.sin(p_lat) * np.cos(d_lat) * np.cos(d_lon - p_lon)
    )
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def prepare_data(df):
    """
    Apply feature engineering:
      - datetime features
      - distances
      - midpoint
      - bearing
      - cyclic encodings
      - log_trip_duration as target
    """
    df = df.copy()

    # Drop ID
    df.drop(columns='id', axis=1, inplace=True)

    # Datetime features
    working = [0, 1, 2, 3, 4]
    rush = [10, 11, 12, 13, 14, 15, 16, 17]

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['day_of_week'] = df['pickup_datetime'].dt.weekday
    df['hour'] = df['pickup_datetime'].dt.hour

    is_working_day = df['pickup_datetime'].dt.weekday.isin(working)
    is_rush_time = df['pickup_datetime'].dt.hour.isin(rush)

    df['Working_days'] = is_working_day.astype(int)
    df['rush_hour'] = (is_working_day & is_rush_time).astype(int)

    # Distance features
    df['latitude_distance'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['longitude_distance'] = df['dropoff_longitude'] - df['pickup_longitude']

    df['manhattan_distance'] = (
        np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
        + np.abs(df['pickup_latitude'] - df['dropoff_latitude'])
    )

    df['haversine_distance'] = haversine(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude'],
    )

    df['distance_ratio'] = df['manhattan_distance'] / (
        df['haversine_distance'] + 1e-24
    )

    # Midpoint
    df['midpoint_latitude'] = (
        df['pickup_latitude'] + df['dropoff_latitude']
    ) / 2
    df['midpoint_longitude'] = (
        df['pickup_longitude'] + df['dropoff_longitude']
    ) / 2

    # Bearing
    df['bearing'] = bearing(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude'],
    )

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['bearing_sin'] = np.sin(2 * np.pi * df['bearing'] / 360)
    df['bearing_cos'] = np.cos(2 * np.pi * df['bearing'] / 360)

    # Target
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    df.drop(columns=['pickup_datetime', 'trip_duration'], inplace=True)
    return df
