import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models import infer_signature

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    REGISTRY_WITHOUT_SCORES,
)


def setup_mlflow():
    """
    Configura la conexión con MLflow y obtiene o crea
    el experimento del proyecto.

    Si el experimento existía pero estaba eliminado,
    lo restaura.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    experiments = client.search_experiments(
        view_type=ViewType.ALL,
    )

    experiment = next(
        (
            exp
            for exp in experiments
            if exp.name == MLFLOW_EXPERIMENT_NAME
        ),
        None,
    )

    if experiment is None:
        experiment_id = client.create_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        print(
            f"Experimento creado: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    else:
        experiment_id = experiment.experiment_id

        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(
                experiment_id
            )

            print(
                f"Experimento restaurado: "
                f"{MLFLOW_EXPERIMENT_NAME}"
            )
        else:
            print(
                f"Experimento encontrado: "
                f"{MLFLOW_EXPERIMENT_NAME}"
            )

    return client, experiment_id


def create_feature_importance(model):
    """
    Obtiene la importancia de variables del pipeline completo.

    Funciona para los modelos utilizados actualmente:
    XGBoost y Random Forest.
    """

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        return None, None

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importances = (
        classifier.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    top_features = importance_df.head(20)

    ax.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )

    ax.set_title(
        "Feature Importance"
    )

    ax.set_xlabel(
        "Importance"
    )

    fig.tight_layout()

    return fig, importance_df


def log_model_run(
    model_name,
    grid_search,
    evaluation,
    X_test,
    experiment_id,
):
    """
    Registra un modelo y sus resultados en MLflow.

    Registra:
    - hiperparámetros
    - métricas de test
    - mejor ROC AUC de cross-validation
    - matriz de confusión
    - curva ROC
    - importancia de variables
    - pipeline completo
    - firma del modelo
    - ejemplo de entrada
    - versión en Model Registry
    """

    model = grid_search.best_estimator_

    metrics = evaluation[
        "metrics"
    ].copy()

    metrics[
        "cv_best_roc_auc"
    ] = float(
        grid_search.best_score_
    )

    params = {
        key.replace(
            "classifier__",
            "",
        ): value
        for key, value
        in grid_search.best_params_.items()
    }

    input_example = (
        X_test.head(5).copy()
    )

    signature = infer_signature(
        X_test.head(50),
        model.predict(
            X_test.head(50)
        ),
    )

    tags = {
        "project": (
            "TP aprendizaje de máquina - Grupo 8"
        ),
        "task": "binary_classification",
        "target": "satisfaction",
        "model_family": model_name,
        "feature_set": "without_scores",
        "recommended_for_production": "true",
        "potential_data_leakage": "false",
    }

    run_name = (
        f"{model_name}_without_scores"
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:

        mlflow.log_params(
            params
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_figure(
            evaluation[
                "confusion_matrix_figure"
            ],
            "plots/confusion_matrix.png",
        )

        mlflow.log_figure(
            evaluation[
                "roc_curve_figure"
            ],
            "plots/roc_curve.png",
        )

        (
            importance_fig,
            importance_df,
        ) = create_feature_importance(
            model
        )

        if importance_fig is not None:

            mlflow.log_figure(
                importance_fig,
                "plots/feature_importance.png",
            )

            mlflow.log_table(
                data=importance_df,
                artifact_file=(
                    "tables/"
                    "feature_importance.json"
                ),
            )

            plt.close(
                importance_fig
            )

        model_info = (
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=(
                    REGISTRY_WITHOUT_SCORES
                ),
                signature=signature,
                input_example=input_example,
                serialization_format=(
                    "cloudpickle"
                ),
                metadata={
                    "target": "satisfaction",
                    "feature_set": (
                        "without_scores"
                    ),
                },
            )
        )

        version = getattr(
            model_info,
            "registered_model_version",
            None,
        )

        # Fallback por si MLflow no devuelve
        # directamente registered_model_version.
        if version is None:

            client = MlflowClient()

            versions = (
                client.search_model_versions(
                    filter_string=(
                        f"name = "
                        f"'{REGISTRY_WITHOUT_SCORES}'"
                    )
                )
            )

            matching_versions = [
                item
                for item in versions
                if item.run_id == run.info.run_id
            ]

            if not matching_versions:
                raise RuntimeError(
                    "No se pudo determinar "
                    "la versión registrada."
                )

            version = max(
                matching_versions,
                key=lambda item: int(
                    item.version
                ),
            ).version

        print(
            f"\nModelo registrado: "
            f"{model_name}"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        print(
            f"Model Version: "
            f"{version}"
        )

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "version": str(version),
            "roc_auc": metrics[
                "roc_auc"
            ],
            "cv_roc_auc": metrics[
                "cv_best_roc_auc"
            ],
        }


def register_models(
    trained_models,
    evaluation_results,
    X_test,
    experiment_id,
):
    """
    Registra en MLflow todos los modelos entrenados.
    """

    registration_results = []

    for (
        model_name,
        grid_search,
    ) in trained_models.items():

        result = log_model_run(
            model_name=model_name,
            grid_search=grid_search,
            evaluation=(
                evaluation_results[
                    model_name
                ]
            ),
            X_test=X_test,
            experiment_id=(
                experiment_id
            ),
        )

        registration_results.append(
            result
        )

    return registration_results


def assign_model_aliases(
    registration_results,
):
    """
    Asigna:

        champion   -> mayor ROC AUC test
        challenger -> segundo ROC AUC test
    """

    client = MlflowClient()

    ranked_models = sorted(
        registration_results,
        key=lambda item: item[
            "roc_auc"
        ],
        reverse=True,
    )

    champion = ranked_models[0]

    client.set_registered_model_alias(
        name=REGISTRY_WITHOUT_SCORES,
        alias="champion",
        version=champion[
            "version"
        ],
    )

    print(
        "\nChampion:"
        f" {champion['model_name']}"
        f" | ROC AUC:"
        f" {champion['roc_auc']:.4f}"
        f" | versión:"
        f" {champion['version']}"
    )

    if len(ranked_models) > 1:

        challenger = ranked_models[1]

        client.set_registered_model_alias(
            name=(
                REGISTRY_WITHOUT_SCORES
            ),
            alias="challenger",
            version=challenger[
                "version"
            ],
        )

        print(
            "Challenger:"
            f" {challenger['model_name']}"
            f" | ROC AUC:"
            f" {challenger['roc_auc']:.4f}"
            f" | versión:"
            f" {challenger['version']}"
        )

    return ranked_models


if __name__ == "__main__":

    client, experiment_id = (
        setup_mlflow()
    )

    print(
        "\nMLflow Tracking URI:",
        mlflow.get_tracking_uri(),
    )

    print(
        "Experiment ID:",
        experiment_id,
    )

    print(
        "Registry:",
        REGISTRY_WITHOUT_SCORES,
    )