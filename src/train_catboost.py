from catboost import CatBoostRegressor

from pipeline import run_pipeline
from evaluation import evaluate_model

def run_catboost():
    X_train, y_train, X_val, y_val, feature_names = run_pipeline(processor=0, use_all_features=False)

    model = CatBoostRegressor(
        iterations=3000,
        depth=8,
        learning_rate=0.03,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    y_pred = model.predict(X_val)

    results = evaluate_model(y_val, y_pred, model_name='CatBoost')

    print(results)

    model.save_model('models/catboost_nyc.cbm')
    print('Catboost model saved')
