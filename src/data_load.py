import pandas as pd


def load_data():
    train = pd.read_csv('../data/train.csv')
    validation = pd.read_csv('../data/val.csv')

    return train, validation


def Outliers_handling(df):

    df = df.copy()

    # Trip duration outliers
    MAX_DURATION_SECONDS = 43200  # 12 hours
    df = df[df['trip_duration'] > 0]
    df = df[df['trip_duration'] <= MAX_DURATION_SECONDS]

    # NYC geo bounds
    NYC_BOUNDS = {'min_lon': -74.03, 'max_lon': -73.75,
                  'min_lat': 40.60, 'max_lat': 40.87}

    valid_lon = ((df['pickup_longitude'] >= NYC_BOUNDS['min_lon']) &
                 (df['pickup_longitude'] <= NYC_BOUNDS['max_lon']) &
                 (df['dropoff_longitude'] >= NYC_BOUNDS['min_lon']) &
                 (df['dropoff_longitude'] <= NYC_BOUNDS['max_lon']))

    valid_lat = ((df['pickup_latitude'] >= NYC_BOUNDS['min_lat']) &
                 (df['pickup_latitude'] <= NYC_BOUNDS['max_lat']) &
                 (df['dropoff_latitude'] >= NYC_BOUNDS['min_lat']) &
                 (df['dropoff_latitude'] <= NYC_BOUNDS['max_lat']))

    df = df[valid_lon & valid_lat].copy()

    # Passenger count outliers
    df = df[~df['passenger_count'].isin([0, 7])]

    return df
