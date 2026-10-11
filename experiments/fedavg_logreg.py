"""Par de la misma regresión logística: centralizada vs FedAvg.

No registra nada en MLflow y no toca al champion.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_CLIENTS = 5
N_ROUNDS = 10
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "satisfaction"
TARGET_MAP = {"neutral or dissatisfied": 0, "satisfied": 1}

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
TEST_FILE = ROOT / "data" / "raw" / "test.csv"


def load_xy(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
    y = df[TARGET].map(TARGET_MAP).to_numpy()
    return df[FEATURES], y


def preprocess(X_train, X_test):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUM_COLS),
            ("categorical", categorical, CAT_COLS),
        ]
    )
    return (
        preprocessor.fit_transform(X_train),
        preprocessor.transform(X_test),
    )


def new_classifier():
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="constant",
        eta0=0.05,
        random_state=RANDOM_STATE,
        max_iter=1,
        tol=None,
        warm_start=True,
    )


def local_update(coef, intercept, X, y):
    clf = new_classifier()
    seed_x = X[:1]
    # partial_fit necesita ver las dos clases una vez para fijar el espacio.
    clf.partial_fit(
        np.vstack([seed_x, seed_x]),
        np.array([0, 1]),
        classes=np.array([0, 1]),
    )
    clf.coef_ = coef.copy()
    clf.intercept_ = intercept.copy()
    clf.partial_fit(X, y)
    return clf.coef_.copy(), clf.intercept_.copy(), len(y)


def run_rounds(splits, n_features):
    coef = np.zeros((1, n_features))
    intercept = np.zeros(1)
    for _ in range(N_ROUNDS):
        updates = [local_update(coef, intercept, X, y) for X, y in splits]
        total = sum(count for _, _, count in updates)
        coef = sum(c * (count / total) for c, _, count in updates)
        intercept = sum(b * (count / total) for _, b, count in updates)
    return coef, intercept


def auc(coef, intercept, X, y):
    scores = np.asarray(X @ coef.ravel() + intercept.ravel()[0]).ravel()
    return float(roc_auc_score(y, scores))


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    X_train_df, y_train = load_xy(TRAIN_FILE)
    X_test_df, y_test = load_xy(TEST_FILE)
    X_train, X_test = preprocess(X_train_df, X_test_df)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    order = rng.permutation(len(y_train))
    X_train = X_train[order]
    y_train = y_train[order]
    folds = np.array_split(np.arange(len(y_train)), N_CLIENTS)
    clients = [(X_train[idx], y_train[idx]) for idx in folds]

    coef_c, intercept_c = run_rounds([(X_train, y_train)], X_train.shape[1])
    coef_f, intercept_f = run_rounds(clients, X_train.shape[1])

    auc_c = auc(coef_c, intercept_c, X_test, y_test)
    auc_f = auc(coef_f, intercept_f, X_test, y_test)
    print(f"clientes={N_CLIENTS} rondas={N_ROUNDS}")
    print(f"roc_auc_centralizada={auc_c:.4f}")
    print(f"roc_auc_federada={auc_f:.4f}")
    print(f"diferencia={auc_f - auc_c:+.4f}")


if __name__ == "__main__":
    main()
