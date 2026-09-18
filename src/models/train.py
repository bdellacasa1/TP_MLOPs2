from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.models.preprocessing import build_preprocessor


def build_model_pipeline(classifier) -> Pipeline:
    """
    Construye un pipeline completo:

        preprocessing -> classifier

    El preprocesamiento queda incluido dentro del pipeline para evitar
    data leakage durante la validación cruzada.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_xgboost_search() -> GridSearchCV:
    """
    Construye el GridSearchCV para XGBoost utilizando
    únicamente las variables without_scores.
    """

    classifier = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    pipeline = build_model_pipeline(classifier)

    param_grid = {
        "classifier__n_estimators": [100, 150, 250],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__subsample": [0.8],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )


def build_random_forest_search() -> GridSearchCV:
    """
    Construye el GridSearchCV para Random Forest utilizando
    únicamente las variables without_scores.
    """

    classifier = RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipeline = build_model_pipeline(classifier)

    param_grid = {
        "classifier__n_estimators": [150, 200, 250],
        "classifier__max_depth": [3, 5],
        "classifier__min_samples_split": [2, 5],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )


def train_models(X_train, y_train):
    """
    Entrena XGBoost y Random Forest mediante GridSearchCV.

    Returns
    -------
    dict
        Diccionario con los GridSearchCV entrenados.
    """

    print("\n=================================")
    print("Entrenando XGBoost")
    print("=================================")

    grid_xgb = build_xgboost_search()
    grid_xgb.fit(X_train, y_train)

    print("\nXGBoost terminado")
    print("Mejores parámetros:")
    print(grid_xgb.best_params_)
    print(
        f"Mejor ROC AUC CV: "
        f"{grid_xgb.best_score_:.4f}"
    )

    print("\n=================================")
    print("Entrenando Random Forest")
    print("=================================")

    grid_rf = build_random_forest_search()
    grid_rf.fit(X_train, y_train)

    print("\nRandom Forest terminado")
    print("Mejores parámetros:")
    print(grid_rf.best_params_)
    print(
        f"Mejor ROC AUC CV: "
        f"{grid_rf.best_score_:.4f}"
    )

    return {
        "xgboost": grid_xgb,
        "random_forest": grid_rf,
    }


if __name__ == "__main__":

    from src.data.load_data import load_and_prepare_data
    from src.models.preprocessing import prepare_features

    train_df, test_df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = prepare_features(
        train_df,
        test_df,
    )

    print("\nDataset preparado")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    trained_models = train_models(
        X_train,
        y_train,
    )

    print("\n=================================")
    print("RESUMEN")
    print("=================================")

    for model_name, grid_search in trained_models.items():

        print(f"\n{model_name}")
        print(
            "Best ROC AUC CV:",
            round(grid_search.best_score_, 4),
        )
        print(
            "Best params:",
            grid_search.best_params_,
        )