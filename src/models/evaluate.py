import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def evaluate_model(
    grid_search,
    X_test,
    y_test,
):
    """
    Evalúa el mejor modelo encontrado por GridSearchCV
    utilizando el conjunto de test.

    Returns
    -------
    dict
        Métricas, predicciones, probabilidades y figuras
        generadas durante la evaluación.
    """

    model = grid_search.best_estimator_

    # Predicción de clase
    y_pred = model.predict(X_test)

    # Probabilidad de la clase positiva: satisfied = 1
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_proba,
        ),
    }

    # Matriz de confusión
    confusion_display = (
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
        )
    )

    confusion_fig = confusion_display.figure_
    confusion_fig.tight_layout()

    # Curva ROC
    roc_display = RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
    )

    roc_fig = roc_display.figure_
    roc_fig.tight_layout()

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix_figure": confusion_fig,
        "roc_curve_figure": roc_fig,
    }


def evaluate_models(
    trained_models,
    X_test,
    y_test,
):
    """
    Evalúa todos los modelos entrenados.

    Parameters
    ----------
    trained_models : dict
        Diccionario generado por train_models().

    Returns
    -------
    dict
        Resultados de evaluación para cada modelo.
    """

    results = {}

    for model_name, grid_search in trained_models.items():

        print("\n=================================")
        print(f"Evaluando {model_name}")
        print("=================================")

        evaluation = evaluate_model(
            grid_search,
            X_test,
            y_test,
        )

        results[model_name] = evaluation

        for metric_name, value in evaluation[
            "metrics"
        ].items():
            print(
                f"{metric_name}: "
                f"{value:.4f}"
            )

    return results


def get_best_model(results):
    """
    Selecciona el mejor modelo según ROC AUC en test.

    Returns
    -------
    tuple
        Nombre del modelo y resultados de evaluación.
    """

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["roc_auc"],
    )

    return (
        best_model_name,
        results[best_model_name],
    )


if __name__ == "__main__":

    from src.data.load_data import load_and_prepare_data
    from src.models.preprocessing import prepare_features
    from src.models.train import train_models

    train_df, test_df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = prepare_features(
        train_df,
        test_df,
    )

    trained_models = train_models(
        X_train,
        y_train,
    )

    results = evaluate_models(
        trained_models,
        X_test,
        y_test,
    )

    best_model_name, best_result = get_best_model(
        results
    )

    print("\n=================================")
    print("MEJOR MODELO")
    print("=================================")

    print(
        "Modelo:",
        best_model_name,
    )

    print(
        "ROC AUC:",
        round(
            best_result["metrics"]["roc_auc"],
            4,
        ),
    )

    # Cerramos las figuras porque desde consola
    # no necesitamos mostrarlas.
    for result in results.values():
        plt.close(
            result["confusion_matrix_figure"]
        )
        plt.close(
            result["roc_curve_figure"]
        )