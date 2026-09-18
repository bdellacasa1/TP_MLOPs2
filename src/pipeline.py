import matplotlib.pyplot as plt

from src.data.load_data import load_and_prepare_data
from src.models.preprocessing import prepare_features
from src.models.train import train_models
from src.models.evaluate import evaluate_models
from src.tracking.mlflow_utils import (
    setup_mlflow,
    register_models,
    assign_model_aliases,
)


def run_pipeline():
    """
    Ejecuta el pipeline completo de entrenamiento.

    Flujo:
        1. Configurar MLflow
        2. Cargar y preparar datos
        3. Separar features y target
        4. Entrenar modelos
        5. Evaluar modelos
        6. Registrar runs y modelos en MLflow
        7. Asignar champion y challenger
    """

    print("\n========================================")
    print("INICIO PIPELINE")
    print("Airline Passenger Satisfaction")
    print("========================================")

    # ---------------------------------------------------------
    # 1. Configurar MLflow
    # ---------------------------------------------------------

    print("\n[1/7] Configurando MLflow...")

    client, experiment_id = setup_mlflow()

    print(
        f"Experiment ID: {experiment_id}"
    )

    # ---------------------------------------------------------
    # 2. Cargar datos
    # ---------------------------------------------------------

    print("\n[2/7] Cargando dataset...")

    train_df, test_df = (
        load_and_prepare_data()
    )

    print(
        f"Train: {train_df.shape}"
    )

    print(
        f"Test: {test_df.shape}"
    )

    # ---------------------------------------------------------
    # 3. Preparar features
    # ---------------------------------------------------------

    print("\n[3/7] Preparando features...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_features(
        train_df,
        test_df,
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test: {X_test.shape}"
    )

    # ---------------------------------------------------------
    # 4. Entrenar modelos
    # ---------------------------------------------------------

    print("\n[4/7] Entrenando modelos...")

    trained_models = train_models(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 5. Evaluar modelos
    # ---------------------------------------------------------

    print("\n[5/7] Evaluando modelos...")

    evaluation_results = evaluate_models(
        trained_models,
        X_test,
        y_test,
    )

    # ---------------------------------------------------------
    # 6. Registrar en MLflow
    # ---------------------------------------------------------

    print(
        "\n[6/7] Registrando modelos "
        "y métricas en MLflow..."
    )

    registration_results = register_models(
        trained_models=trained_models,
        evaluation_results=(
            evaluation_results
        ),
        X_test=X_test,
        experiment_id=experiment_id,
    )

    # ---------------------------------------------------------
    # 7. Champion / Challenger
    # ---------------------------------------------------------

    print(
        "\n[7/7] Asignando aliases..."
    )

    ranked_models = assign_model_aliases(
        registration_results
    )

    # ---------------------------------------------------------
    # Resumen final
    # ---------------------------------------------------------

    print("\n========================================")
    print("RESUMEN FINAL")
    print("========================================")

    for position, model_info in enumerate(
        ranked_models,
        start=1,
    ):

        alias = (
            "champion"
            if position == 1
            else "challenger"
        )

        print(
            f"\n{position}. "
            f"{model_info['model_name']}"
        )

        print(
            f"   Alias: {alias}"
        )

        print(
            f"   ROC AUC test: "
            f"{model_info['roc_auc']:.4f}"
        )

        print(
            f"   ROC AUC CV: "
            f"{model_info['cv_roc_auc']:.4f}"
        )

        print(
            f"   Model version: "
            f"{model_info['version']}"
        )

        print(
            f"   Run ID: "
            f"{model_info['run_id']}"
        )

    # ---------------------------------------------------------
    # Cerrar figuras matplotlib
    # ---------------------------------------------------------

    for result in evaluation_results.values():

        plt.close(
            result[
                "confusion_matrix_figure"
            ]
        )

        plt.close(
            result[
                "roc_curve_figure"
            ]
        )

    print("\n========================================")
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("========================================")

    return {
        "trained_models": trained_models,
        "evaluation_results": (
            evaluation_results
        ),
        "registration_results": (
            registration_results
        ),
        "ranked_models": ranked_models,
    }


if __name__ == "__main__":
    run_pipeline()