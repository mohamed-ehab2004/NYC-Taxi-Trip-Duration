# NYC Taxi Trip Duration Prediction

A machine learning project for predicting New York City taxi trip duration using the Kaggle _NYC Taxi Trip Duration_ dataset.  
This project includes data cleaning, rich feature engineering, multiple regression models (CatBoost, Random Forest, Ridge), and a modern Streamlit web application for making predictions with an interactive NYC map.

---

## 🎯 Overview

This project implements a trip duration prediction system that estimates how long a taxi ride in NYC will take, based on pickup time and GPS locations.  
It combines custom feature engineering with powerful tree-based models to produce accurate, interpretable predictions.

**Key Technologies:**

- Python 3.8+
- NumPy & Pandas for data handling
- Scikit-learn for pipelines and metrics
- CatBoost for gradient boosting
- Streamlit for the web interface
- Folium / streamlit-folium for interactive maps

---

## 📁 Project Structure

```text
.
├── data/
│   ├── train.zip                     # Compressed training set
│   └── val.zip                       # Compressed validation set
│
├── models/
│   ├── catboost_nyc.cbm              # Final CatBoost model
│   └── ridge.pkl                     # Ridge regression baseline
│
├── notebook/
│   └── EDA.ipynb                     # Exploratory data analysis & experiments
│
├── data_load.py                      # Data loading & outlier handling
├── evaluation.py                     # Evaluation metrics (R2, RMSE, RMSLE)
├── feature_engineering.py            # All feature engineering utilities
├── feature_selection.py              # Simple feature importance via Random Forest
├── pipeline.py                       # End-to-end preprocessing pipeline
├── train_catboost.py                 # Train CatBoost model
├── train_random_forest.py            # Train Random Forest model
├── train_ridge.py                    # Train Ridge regression model
├── main.py                           # Example script to run training
└── streamlit_app.py                  # Streamlit web application
```

---

## ✨ Features

- **Data Preprocessing**

  - Loads train/validation splits from `data/`
  - Removes trips with:

    - Non-positive or extremely long durations (> 12 hours)
    - Coordinates outside a NYC geographic bounding box
    - Invalid passenger counts (0, 7, etc.)

- **Feature Engineering**

  - Haversine distance (great-circle distance in km)
  - Manhattan distance and latitude/longitude deltas
  - Distance ratio (Manhattan / Haversine)
  - Midpoint latitude & longitude
  - Bearing (direction from pickup to dropoff)
  - Time-based features (day, month, weekday, hour)
  - Working-day and rush-hour flags
  - Cyclical encodings for time and bearing (sin/cos)
  - Simple one-hot style features (e.g., vendor ID, month dummies)

- **Models**

  - CatBoostRegressor (main production model)
  - RandomForestRegressor (baseline)
  - Ridge Regression (linear baseline)
  - Metrics: R², RMSE, RMSLE

- **Interactive Web App**

  - Built with Streamlit
  - Validates that coordinates lie inside NYC
  - Quickly fill form using preset example routes
  - Displays predicted trip duration in minutes
  - Shows pickup & dropoff on an interactive NYC map (Folium)

---

## 💻 Usage

### 🔹 Training the Models

Train the main CatBoost model:

```bash
python train_catboost.py
```

Train the Random Forest baseline:

```bash
python train_random_forest.py
```

Train the Ridge regression baseline:

```bash
python train_ridge.py
```

Each script:

- Loads and cleans the data using `data_load.py`
- Applies feature engineering from `feature_engineering.py`
- Runs the preprocessing pipeline from `pipeline.py`
- Evaluates performance with `evaluation.py`
- Saves the trained model into the `models/` directory

---

### 🔹 Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app provides:

- A form to enter:

  - Vendor ID
  - Pickup date & time (hour + minute)
  - Pickup & dropoff latitude/longitude

- Coordinate validation to ensure all points are inside NYC bounds
- Quick preset example routes for demo
- Predicted trip duration in **minutes**
- An interactive map showing pickup and dropoff points

---

## 🔬 Model Details

**Algorithm**

- Main model: `CatBoostRegressor`
- Target variable: `log_trip_duration = log1p(trip_duration_seconds)`
- Loss metric: RMSE / RMSLE
- Features: all engineered time, distance, and geometry features listed above

**Evaluation**

`evaluation.py` computes:

- R² (coefficient of determination)
- RMSE (root mean squared error)
- RMSLE (root mean squared log error – robust to skewed durations)

---

## 📊 Dataset

The project is based on the **NYC Taxi Trip Duration** competition dataset on Kaggle:

> [https://www.kaggle.com/c/nyc-taxi-trip-duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)

The full CSVs are large, so this repository stores zipped versions (`train.zip`, `val.zip`).
You can download or regenerate the original CSVs from Kaggle as needed.

---

## 👤 Author

**Mahmoud Ashraf**
Machine Learning Engineer
