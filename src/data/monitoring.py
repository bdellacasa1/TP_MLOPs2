import pandas as pd
import numpy as np

def validate_data_quality(df: pd.DataFrame):
    """
    Realiza validaciones críticas sobre la calidad del dataset crudo.
    Lanza una excepción si los datos no cumplen con los estándares mínimos.
    """
    print("🔍 Iniciando validación de calidad de datos...")
    
    # 1. Validación de Esquema (Columnas requeridas)
    required_cols = [
        "Gender", "Customer Type", "Age", "Type of Travel", 
        "Class", "Flight Distance", "Departure Delay in Minutes", 
        "Arrival Delay in Minutes", "satisfaction"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Error de esquema: Faltan las columnas {missing_cols}")
    print("✅ Esquema validado correctamente.")

    # 2. Validación de Nulos (Umbral máximo del 10%)
    null_threshold = 0.10
    null_percentages = df[required_cols].isnull().mean()
    critical_nulls = null_percentages[null_percentages > null_threshold]
    
    if not critical_nulls.empty:
        raise ValueError(f"❌ Error de integridad: Demasiados valores nulos en {critical_nulls.to_dict()}")
    print("✅ Porcentaje de nulos dentro de límites aceptables.")

    # 3. Validación de Lógica de Negocio (Rangos)
    # Edad no puede ser negativa ni absurdamente alta
    if not df["Age"].between(0, 120).all():
        invalid_ages = df[~df["Age"].between(0, 120)]["Age"].unique()
        raise ValueError(f"❌ Error lógico: Se encontraron edades inválidas: {invalid_ages}")
    
    # Retrasos no pueden ser negativos
    if (df["Arrival Delay in Minutes"] < 0).any():
        raise ValueError("❌ Error lógico: Se encontraron valores negativos en el retraso de llegada.")
        
    print("✅ Validaciones de rango superadas.")
    print("🚀 Calidad de datos aprobada para entrenamiento.")
    return True

if __name__ == "__main__":
    # Test simple
    data = {
        "Gender": ["Female"], "Customer Type": ["Loyal"], "Age": [30],
        "Type of Travel": ["Business"], "Class": ["Eco"], "Flight Distance": [100],
        "Departure Delay in Minutes": [0], "Arrival Delay in Minutes": [0],
        "satisfaction": ["satisfied"]
    }
    validate_data_quality(pd.DataFrame(data))
