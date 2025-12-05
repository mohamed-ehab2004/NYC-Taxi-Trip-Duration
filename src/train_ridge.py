from pipeline import run_pipeline
from sklearn.linear_model import Ridge
from evaluation import evaluate_model
import joblib


def run_ridge():
    X_train, y_train, X_val, y_val, _ = run_pipeline(processor=1, use_all_features=False)

    model = Ridge(alpha=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    results = evaluate_model(y_val, y_pred, model_name='Ridge α=1')

    print(results)
    joblib.dump(model, 'models/ridge.pkl')
    print('Ridge model saved')


if __name__ == '__main__':
    run_ridge()
