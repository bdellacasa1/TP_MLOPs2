import streamlit as st
import requests
import os
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="centered"
)

# URL del backend (desde variables de entorno o local)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("✈️ Airline Passenger Satisfaction")
st.markdown("""
Esta aplicación predice si un pasajero estará **satisfecho** o **neutral/insatisfecho** 
basado en los detalles de su vuelo.
""")

# Barra lateral para el estado de la API
st.sidebar.header("Estado del Sistema")
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("status") == "ok":
            st.sidebar.success("🟢 API de Inferencia: Conectada")
            st.sidebar.info(f"**Modelo:** {health_data.get('model_name')}")
            st.sidebar.info(f"**Versión:** {health_data.get('model_version')}")
        else:
            st.sidebar.warning("🟡 API: Modelo no cargado")
    else:
        st.sidebar.error("🔴 API: Error de respuesta")
except Exception:
    st.sidebar.error("🔴 API: No disponible")

# Formulario principal
with st.form("prediction_form"):
    st.subheader("📋 Datos del Pasajero")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Género", ["Female", "Male"])
        customer_type = st.selectbox("Tipo de Cliente", ["Loyal Customer", "disloyal Customer"])
        type_of_travel = st.selectbox("Tipo de Viaje", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Clase", ["Business", "Eco", "Eco Plus"])
    
    with col2:
        age = st.number_input("Edad", min_value=0, max_value=120, value=30)
        flight_distance = st.number_input("Distancia del Vuelo", min_value=0, value=1000)
        dep_delay = st.number_input("Retraso Salida (minutos)", min_value=0, value=0)
        arr_delay = st.number_input("Retraso Llegada (minutos)", min_value=0, value=0)

    submit_button = st.form_submit_button(label="🔍 Predecir Satisfacción")

# Acción al presionar el botón
if submit_button:
    # Mapeo de campos para que coincidan exactamente con la API
    payload = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Age": age,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": dep_delay,
        "Arrival Delay in Minutes": arr_delay
    }
    
    with st.spinner("Procesando predicción..."):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("🎯 Resultado")
                
                label = result.get("label")
                prob = result.get("probability")
                
                if label == "satisfied":
                    st.success(f"### El pasajero estará SATISFECHO")
                else:
                    st.warning(f"### El pasajero estará NEUTRAL o INSATISFECHO")
                
                st.metric("Probabilidad de Satisfacción", f"{prob:.2%}")
                
                with st.expander("Ver metadatos del modelo"):
                    st.json(result)
            
            elif response.status_code == 503:
                st.error("❌ El modelo aún no está cargado en el servidor. Ejecuta el pipeline en Airflow primero.")
            else:
                st.error(f"❌ Error en la API ({response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error al conectar con el servidor: {str(e)}")

st.divider()
st.caption("Operaciones de Aprendizaje de Máquina 1 - TP MLOps")
