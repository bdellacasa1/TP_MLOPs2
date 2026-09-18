import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    TARGET_COL,
    CAT_COLS,
    NUM_BASE_COLS,
    FEATURES_WITHOUT_SCORES,
)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Separa las features y el target para entrenamiento y test.

    Se utilizan únicamente las variables without_scores,
    es decir, no se incluyen las puntuaciones de satisfacción
    otorgadas por los pasajeros.
    """

    X_train = train_df[FEATURES_WITHOUT_SCORES].copy()
    X_test = test_df[FEATURES_WITHOUT_SCORES].copy()

    y_train = train_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el preprocesador para las variables utilizadas
    por los modelos de clasificación.

    Variables numéricas:
        - Imputación de valores faltantes mediante la mediana.

    Variables categóricas:
        - Imputación mediante la categoría más frecuente.
        - One-Hot Encoding.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUM_BASE_COLS,
            ),
            (
                "categorical",
                categorical_transformer,
                CAT_COLS,
            ),
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from src.data.load_data import load_and_prepare_data

    train_df, test_df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = prepare_features(
        train_df,
        test_df,
    )

    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("\n--- PREPROCESSING WITHOUT SCORES ---")

    print("X_train original:", X_train.shape)
    print("X_test original:", X_test.shape)

    print(
        "X_train transformado:",
        X_train_transformed.shape,
    )

    print(
        "X_test transformado:",
        X_test_transformed.shape,
    )

    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)