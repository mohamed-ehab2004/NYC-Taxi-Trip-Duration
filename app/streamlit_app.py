import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time
from catboost import CatBoostRegressor

import folium
from streamlit_folium import st_folium


# ----------------- Page config ----------------- #

st.set_page_config(
    page_title="Tawqeet – NYC ETA Prediction",
    page_icon="⏱️",
    layout="wide",
)

# ----------------- Global styling (modern & colorful) ----------------- #

st.markdown(
    """
    <style>
    /* Full app background */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #22c55e 0%, #0f172a 40%, #6366f1 100%);
        color: #f9fafb;
    }

    /* Main container as a glassmorphism card */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        margin: 2rem auto;
        background: rgba(15, 23, 42, 0.92);       /* slate-900 with transparency */
        border-radius: 24px;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Title & subtitle */
    .app-header {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #facc15, #f97316, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -10px;
        letter-spacing: 1px;
    }

    .app-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #e5e7eb;
        margin-top: 0.1rem;
        margin-bottom: 1.6rem;
        opacity: 0.9;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #fbbf24;
    }

    .section-title span.icon {
        font-size: 1.2rem;
    }

    /* Inputs look cleaner & darker */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background-color: #111827 !important;   /* slate-900 */
        color: #e5e7eb !important;
        border-radius: 12px !important;
    }

    .stNumberInput > div > div > input:focus {
        outline: 2px solid #f97316 !important;
        border-color: #f97316 !important;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        font-weight: 700;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
        background: linear-gradient(90deg, #facc15, #f97316);
        color: #111827;
        border: none;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #fde047, #fb923c);
        color: #020617;
    }

    /* Success card */
    .result-card {
        padding: 0.9rem 1.1rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.45));
        border: 1px solid rgba(22,163,74,0.9);
        margin-bottom: 1.0rem;
    }
    .result-card span.label {
        font-weight: 600;
        color: #bbf7d0;
    }
    .result-card span.value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ecfeff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Constants & helpers ----------------- #

NYC_BOUNDS = {
    "min_lon": -74.03,
    "max_lon": -73.75,
    "min_lat": 40.60,
    "max_lat": 40.87,
}

NYC_CENTER = [40.73, -73.97]


def is_inside_nyc(lat: float, lon: float) -> bool:
    return (
        NYC_BOUNDS["min_lat"] <= lat <= NYC_BOUNDS["max_lat"]
        and NYC_BOUNDS["min_lon"] <= lon <= NYC_BOUNDS["max_lon"]
    )


# ----------------- Feature engineering ----------------- #

def haversine(p_lat, p_lon, d_lat, d_lon):
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
    p_lat = np.radians(p_lat)
    p_lon = np.radians(p_lon)
    d_lat = np.radians(d_lat)
    d_lon = np.radians(d_lon)

    x = np.sin(d_lon - p_lon) * np.cos(d_lat)
    y = np.cos(p_lat) * np.sin(d_lat) - (
        np.sin(p_lat) * np.cos(d_lat) * np.cos(d_lon - p_lon)
    )
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def build_catboost_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_week"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour

    working = [0, 1, 2, 3, 4]
    rush = [10, 11, 12, 13, 14, 15, 16, 17]

    is_working_day = df["day_of_week"].isin(working)
    is_rush_time = df["hour"].isin(rush)

    df["Working_days"] = is_working_day.astype(int)
    df["rush_hour"] = (is_working_day & is_rush_time).astype(int)

    df["latitude_distance"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["longitude_distance"] = df["dropoff_longitude"] - df["pickup_longitude"]

    df["manhattan_distance"] = (
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"])
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    )

    df["haversine_distance"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["distance_ratio"] = df["manhattan_distance"] / (
        df["haversine_distance"] + 1e-24
    )

    df["midpoint_latitude"] = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2
    df["midpoint_longitude"] = (df["pickup_longitude"] + df["dropoff_longitude"]) / 2

    df["bearing"] = bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df["bearing_sin"] = np.sin(2 * np.pi * df["bearing"] / 360)
    df["bearing_cos"] = np.cos(2 * np.pi * df["bearing"] / 360)

    df["vendor_id_2"] = (df["vendor_id"] == 2).astype(int)
    df["Working_days_1"] = df["Working_days"]
    df["rush_hour_1"] = df["rush_hour"]

    for m in [2, 3, 4, 5, 6]:
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    feature_order = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "latitude_distance",
        "longitude_distance",
        "manhattan_distance",
        "haversine_distance",
        "distance_ratio",
        "midpoint_latitude",
        "midpoint_longitude",
        "bearing",
        "vendor_id_2",
        "Working_days_1",
        "rush_hour_1",
        "month_2",
        "month_3",
        "month_4",
        "month_5",
        "month_6",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_sin",
        "day_cos",
        "bearing_sin",
        "bearing_cos",
    ]

    return df[feature_order]


# ----------------- Load model ----------------- #

@st.cache_resource
def load_catboost_model():
    model = CatBoostRegressor()
    model.load_model("models/catboost_nyc.cbm")
    return model


# ----------------- Main app ----------------- #

def main():
    # Gradient title
    st.markdown(
        """
        <h1 class="app-header">Tawqeet: NYC ETA Prediction</h1>
        <p class="app-subtitle">
            Predict NYC taxi trip duration from pickup time and locations using your trained ML model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    model = load_catboost_model()

    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None
        st.session_state["coords"] = None

    st.markdown("---")
    st.markdown(
        "<div class='section-title'><span>Trip Details</span></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # Left: vendor + date + custom time
    with col1:
        vendor_id = st.selectbox("Vendor ID", [1, 2], index=0)

        pickup_date = st.date_input(
            "Pickup Date",
            datetime(2016, 3, 1).date(),
        )

        st.markdown("**Pickup Time**")
        t_col1, t_col2 = st.columns(2)
        hour = t_col1.number_input(
            "Hour (0–23)", min_value=0, max_value=23, value=8, step=1
        )
        minute = t_col2.number_input(
            "Minute (0–59)", min_value=0, max_value=59, value=0, step=1
        )
        pickup_time = time(int(hour), int(minute))

    # Right: coordinates
    with col2:
        st.markdown("**Pickup Location**")
        pickup_latitude = st.number_input(
            "Pickup Latitude", value=40.751432, format="%.6f"
        )
        pickup_longitude = st.number_input(
            "Pickup Longitude", value=-73.979815, format="%.6f"
        )

        st.markdown("**Dropoff Location**")
        dropoff_latitude = st.number_input(
            "Dropoff Latitude", value=40.726421, format="%.6f"
        )
        dropoff_longitude = st.number_input(
            "Dropoff Longitude", value=-73.983640, format="%.6f"
        )

        st.caption(
            f"Valid NYC latitude: {NYC_BOUNDS['min_lat']} – {NYC_BOUNDS['max_lat']}, "
            f"longitude: {NYC_BOUNDS['min_lon']} – {NYC_BOUNDS['max_lon']}."
        )

    st.markdown("")
    if st.button("Predict Trip Duration"):
        pickup_ok = is_inside_nyc(pickup_latitude, pickup_longitude)
        dropoff_ok = is_inside_nyc(dropoff_latitude, dropoff_longitude)

        if not pickup_ok or not dropoff_ok:
            st.session_state["prediction"] = None
            st.session_state["coords"] = None
            st.error(
                "The coordinates you entered are outside the NYC area.\n\n"
                "Please enter pickup and dropoff locations inside New York City."
            )
        else:
            pickup_datetime_str = datetime.combine(
                pickup_date, pickup_time
            ).strftime("%Y-%m-%d %H:%M:%S")

            row = {
                "vendor_id": vendor_id,
                "pickup_datetime": pickup_datetime_str,
                "pickup_longitude": pickup_longitude,
                "pickup_latitude": pickup_latitude,
                "dropoff_longitude": dropoff_longitude,
                "dropoff_latitude": dropoff_latitude,
            }

            df_input = pd.DataFrame([row])
            X_input = build_catboost_features(df_input)

            log_pred = model.predict(X_input)[0]
            duration_seconds = float(np.expm1(log_pred))
            duration_minutes = duration_seconds / 60.0

            st.session_state["prediction"] = duration_minutes
            st.session_state["coords"] = (
                pickup_latitude,
                pickup_longitude,
                dropoff_latitude,
                dropoff_longitude,
            )

    if st.session_state["prediction"] is not None:
        duration_minutes = st.session_state["prediction"]
        pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude = (
            st.session_state["coords"]
        )

        st.markdown("---")
        st.markdown(
            "<div class='section-title'><span>Prediction</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card">
                <span class="label">Estimated trip duration:</span>
                <span class="value"> {duration_minutes:.1f} minutes</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Trip Route Map (NYC Only)")

        m = folium.Map(
            location=NYC_CENTER,
            zoom_start=12,
            tiles="CartoDB positron",
        )

        folium.CircleMarker(
            [pickup_latitude, pickup_longitude],
            radius=8,
            color="#22c55e",
            fill=True,
            fill_color="#22c55e",
            fill_opacity=0.95,
            popup="Pickup",
        ).add_to(m)

        folium.CircleMarker(
            [dropoff_latitude, dropoff_longitude],
            radius=8,
            color="#ef4444",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.95,
            popup="Dropoff",
        ).add_to(m)

        st_folium(m, width=900, height=500)


if __name__ == "__main__":
    main()
