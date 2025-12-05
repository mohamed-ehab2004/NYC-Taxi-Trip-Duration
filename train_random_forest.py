from sklearn.ensemble import RandomForestRegressor
import joblib

from pipeline import run_pipeline
from evaluation import evaluate_model


def run_random_forest():
    X_train, y_train, X_val, y_val, feature_names = run_pipeline(processor=0, use_all_features=False)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        max_features='sqrt',
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, model_name='Random Forest')

    print(results)

    joblib.dump(model, "models/random_forest.pkl")
    print('Random forest model saved')


if __name__ == '__main__':
    run_random_forest()
