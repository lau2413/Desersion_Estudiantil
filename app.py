import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración básica
st.set_page_config(page_title="Predicción de Deserción", layout="wide")

# Título
st.title("Sistema de Predicción de Deserción Estudiantil")

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('data/pipeline_final_desercion.pkl')
        columnas = joblib.load('data/columnas_esperadas.pkl')
        return pipeline, columnas
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

pipeline, columnas = load_model()

if pipeline is None:
    st.stop()

# Formulario de entrada
with st.form("student_form"):
    st.header("Datos del Estudiante")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=15, max_value=80, value=20)
        gender = st.radio("Género", options=[0, 1], format_func=lambda x: "Masculino" if x == 0 else "Femenino")
        displaced = st.checkbox("Desplazado", value=False)
        debtor = st.checkbox("Deudor", value=False)
        
    with col2:
        admission_grade = st.number_input("Nota de admisión", min_value=0.0, max_value=20.0, value=12.0)
        scholarship = st.checkbox("Becado", value=False)
        units_enrolled = st.number_input("Unidades matriculadas", min_value=0, value=6)
    
    submitted = st.form_submit_button("Predecir")

if submitted:
    # Preparar datos de entrada
    input_data = {
        'Age at enrollment': age,
        'Gender': int(gender),
        'Displaced': int(displaced),
        'Debtor': int(debtor),
        'Admission grade': admission_grade,
        'Scholarship holder': int(scholarship),
        'Curricular units 2nd sem (enrolled)': units_enrolled,
        # Añade aquí el resto de variables necesarias
    }
    
    # Completar con ceros las columnas faltantes
    for col in columnas:
        if col not in input_data:
            input_data[col] = 0
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([input_data])[columnas]
    
    # Hacer predicción
    try:
        proba = pipeline.predict_proba(input_df)[0][1]
        st.success(f"Probabilidad de deserción: {proba*100:.2f}%")
        
        # Visualización adicional
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Riesgo de deserción", f"{proba*100:.2f}%")
        with col2:
            st.progress(proba)
            
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")