import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import matplotlib.pyplot as plt
import shap
from typing import Tuple

# Usar backend Agg para generar imágenes sin GUI
import matplotlib
matplotlib.use('Agg')

# Añadir raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_dataset, clean_dataset, encode_target
from src.config import MLFLOW_TRACKING_URI, REGISTRY_WITHOUT_SCORES, MLFLOW_EXPERIMENT_NAME

def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Carga y prepara los datos reales de test."""
    _, test_df = load_dataset()
    test_df = clean_dataset(test_df)
    test_df = encode_target(test_df)
    
    X_test = test_df.drop(columns=["satisfaction"])
    y_test = test_df["satisfaction"]
    return X_test, y_test

def run_perturbation_test(model, X_test: pd.DataFrame, n_samples: int = 100):
    """Prueba de estabilidad ante ruido gaussiano en variables numéricas."""
    print("🧪 Ejecutando test de perturbación (Estabilidad)...")
    X_sample = X_test.sample(n_samples, random_state=42).copy()
    
    # Identificar columnas numéricas
    num_cols = X_sample.select_dtypes(include=[np.number]).columns
    
    # Predicción original
    orig_preds = model.predict(X_sample)
    
    # Añadir ruido (10% de la desviación estándar)
    X_perturbed = X_sample.copy()
    for col in num_cols:
        scale = X_test[col].std() * 0.1
        X_perturbed[col] += np.random.normal(0, scale, size=n_samples)
    
    pert_preds = model.predict(X_perturbed)
    
    # Calcular estabilidad (cuántas predicciones se mantienen iguales)
    stability = np.mean(orig_preds == pert_preds)
    print(f"✅ Estabilidad: {stability:.2%}")
    mlflow.log_metric("robustness_stability_score", stability)
    return stability > 0.90

def run_invariance_test(model, X_test: pd.DataFrame, n_samples: int = 100):
    """Prueba de invarianza (Fairness) sobre la variable Gender."""
    print("🧪 Ejecutando test de invarianza (Equidad de Género)...")
    X_sample = X_test.sample(n_samples, random_state=42).copy()
    
    # Forzar a todos a Female y predecir
    X_sample['Gender'] = 'Female'
    preds_female = model.predict(X_sample)
    
    # Forzar a todos a Male y predecir
    X_sample['Gender'] = 'Male'
    preds_male = model.predict(X_sample)
    
    # Calcular paridad (qué porcentaje de predicciones son idénticas)
    parity = np.mean(preds_female == preds_male)
    print(f"✅ Paridad de Género: {parity:.2%}")
    mlflow.log_metric("fairness_gender_parity", parity)
    return parity > 0.95

def run_directional_test(model, X_test: pd.DataFrame, n_samples: int = 100):
    """Prueba de lógica de negocio: mayor retraso no debería aumentar satisfacción."""
    print("🧪 Ejecutando test de expectativas direccionales (Lógica de Negocio)...")
    X_sample = X_test.sample(n_samples, random_state=42).copy()
    
    # Escenario 1: Sin retraso
    X_sample['Arrival Delay in Minutes'] = 0
    probs_no_delay = model.predict_proba(X_sample)[:, 1]
    
    # Escenario 2: Gran retraso (ej. 5 horas)
    X_sample['Arrival Delay in Minutes'] = 300
    probs_high_delay = model.predict_proba(X_sample)[:, 1]
    
    # Verificar cuántos casos cumplen que Prob(Satisfecho | Delay) <= Prob(Satisfecho | No Delay)
    logical_consistency = np.mean(probs_high_delay <= probs_no_delay + 0.05) # Margen de tolerancia del 5%
    print(f"✅ Consistencia Lógica (Retraso vs Satisfacción): {logical_consistency:.2%}")
    mlflow.log_metric("business_logic_consistency", logical_consistency)
    return logical_consistency > 0.90

def generate_shap_report(model, X_test: pd.DataFrame, n_samples: int = 20):
    """Genera e integra el gráfico SHAP en MLflow."""
    print("🧪 Generando reporte de explicabilidad SHAP...")
    X_sample = X_test.sample(n_samples, random_state=42)
    
    # Usamos KernelExplainer con una función lambda para evitar problemas de compatibilidad con Pipelines
    # Es más lento pero más robusto para este caso
    predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=X_test.columns))
    explainer = shap.KernelExplainer(predict_fn, shap.sample(X_test, 10))
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    output_path = "shap_summary.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(output_path)
    print(f"✅ Gráfico SHAP registrado como artefacto.")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"🚀 Iniciando suite de pruebas de robustez en experimento '{MLFLOW_EXPERIMENT_NAME}'...")
    X_test, _ = load_test_data()
    
    # Cargar el modelo Champion de MLflow
    model_uri = f"models:/{REGISTRY_WITHOUT_SCORES}@champion"
    print(f"Cargando modelo Champion: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    # Iniciar o unirse a una corrida de MLflow (opcionalmente se puede pasar el run_id)
    # Aquí creamos una nueva corrida para auditoría de calidad
    with mlflow.start_run(run_name="robustness_audit"):
        res1 = run_perturbation_test(model, X_test)
        res2 = run_invariance_test(model, X_test)
        res3 = run_directional_test(model, X_test)
        generate_shap_report(model, X_test)
        
        status = "PASSED" if all([res1, res2, res3]) else "FAILED"
        mlflow.set_tag("robustness_status", status)
        print(f"\n📢 Resultado final de la auditoría: {status}")

if __name__ == "__main__":
    main()
