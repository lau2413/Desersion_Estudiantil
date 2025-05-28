import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Cargar el pipeline completo (ya incluye preprocesador y modelo)
        model = joblib.load('pipeline_final_desercion.pkl')
        
        # Cargar las columnas esperadas
        try:
            columnas_esperadas = joblib.load('columnas_esperadas.pkl')
        except FileNotFoundError:
            # Si no existe el archivo, usar las columnas del error que mostraste
            columnas_esperadas = [
                "Application order","Daytime/evening attendance","Previous qualification (grade)",
                "Admission grade","Displaced","Debtor","Tuition fees up to date","Gender",
                "Scholarship holder","Age at enrollment","Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (without evaluations)","Curricular units 2nd sem (credited)",
                "Curricular units 2nd sem (enrolled)","Curricular units 2nd sem (evaluations)",
                "Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)",
                "Curricular units 2nd sem (without evaluations)","Unemployment rate",
                "Inflation rate","GDP","Marital status_Divorced","Marital status_FactoUnion",
                "Marital status_Separated","Marital status_Single","Application mode_Admisión Especial",
                "Application mode_Admisión Regular","Application mode_Admisión por Ordenanza",
                "Application mode_Cambios/Transferencias","Application mode_Estudiantes Internacionales",
                "Application mode_Mayores de 23 años","Course_Agricultural & Environmental Sciences",
                "Course_Arts & Design","Course_Business & Management","Course_Communication & Media",
                "Course_Education","Course_Engineering & Technology","Course_Health Sciences",
                "Course_Social Sciences","Previous qualification_Higher Education",
                "Previous qualification_Other","Previous qualification_Secondary Education",
                "Previous qualification_Technical Education","Nacionality_Colombian",
                "Nacionality_Cuban","Nacionality_Dutch","Nacionality_English","Nacionality_German",
                "Nacionality_Italian","Nacionality_Lithuanian","Nacionality_Moldovan",
                "Nacionality_Mozambican","Nacionality_Portuguese","Nacionality_Romanian",
                "Nacionality_Santomean","Nacionality_Turkish","Mother's qualification_Basic_or_Secondary",
                "Mother's qualification_Other_or_Unknown","Mother's qualification_Postgraduate",
                "Mother's qualification_Technical_Education","Father's qualification_Basic_or_Secondary",
                "Father's qualification_Other_or_Unknown","Father's qualification_Postgraduate",
                "Mother's occupation_Administrative/Clerical","Mother's occupation_Skilled Manual Workers",
                "Mother's occupation_Special Cases","Mother's occupation_Technicians/Associate Professionals",
                "Mother's occupation_Unskilled Workers","Father's occupation_Administrative/Clerical",
                "Father's occupation_Professionals","Father's occupation_Skilled Manual Workers",
                "Father's occupation_Special Cases","Father's occupation_Technicians/Associate Professionals"
            ]
            
        return model, columnas_esperadas
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None

# Función para preparar los datos
def prepare_data(input_data, columnas_esperadas):
    """
    Prepara los datos de entrada para la predicción
    El pipeline ya incluye el preprocesador, solo necesitamos formatear correctamente
    """
    try:
        # Convertir a DataFrame si es necesario
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            st.error("Formato de datos no válido")
            return None
        
        # Asegurar que todas las columnas esperadas estén presentes
        for col in columnas_esperadas:
            if col not in df.columns:
                df[col] = 0  # Valor por defecto para columnas faltantes
        
        # Reordenar columnas para que coincidan con el orden esperado
        df = df[columnas_esperadas]
        
        # Definir tipos de datos según tu documento
        variables_int = [
            'Application order', 'Daytime/evening attendance', 'Displaced',
            'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (without evaluations)'
        ]
        
        variables_float = [
            'Previous qualification (grade)', 'Admission grade', 'Curricular units 2nd sem (grade)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
        
        # Convertir tipos de datos
        for col in variables_int:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        for col in variables_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
        
        # Las variables booleanas (dummies) deben ser 0 o 1
        variables_bool = [col for col in columnas_esperadas if col not in variables_int + variables_float]
        for col in variables_bool:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                # Asegurar que sea 0 o 1
                df[col] = df[col].clip(0, 1)
        
        return df
        
    except Exception as e:
        st.error(f"Error preparando datos: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Función principal de predicción
def make_prediction(input_data):
    """
    Realiza la predicción usando el pipeline completo
    """
    try:
        # Cargar modelo y columnas esperadas
        model, columnas_esperadas = load_model()
        if model is None:
            return None
        
        # Preparar datos
        df_prepared = prepare_data(input_data, columnas_esperadas)
        if df_prepared is None:
            return None
        
        st.write("Datos preparados exitosamente:")
        st.write(f"Shape: {df_prepared.shape}")
        st.write("Primeras columnas:", df_prepared.columns[:10].tolist())
        
        # Realizar predicción directamente con el pipeline
        # El pipeline ya incluye el preprocesador, no necesitamos aplicar scaling adicional
        prediction = model.predict(df_prepared)
        
        # Obtener probabilidades si es posible
        try:
            probabilities = model.predict_proba(df_prepared)
            return {
                'prediction': prediction[0],
                'probabilities': probabilities[0],
                'prediction_label': 'Deserción' if prediction[0] == 1 else 'No Deserción'
            }
        except Exception as prob_error:
            st.warning(f"No se pudieron obtener probabilidades: {prob_error}")
            return {
                'prediction': prediction[0],
                'probabilities': None,
                'prediction_label': 'Deserción' if prediction[0] == 1 else 'No Deserción'
            }
            
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Función para crear datos de ejemplo
def create_sample_data():
    """
    Crea un ejemplo de datos para testing
    """
    sample = {
        "Application order": 1,
        "Daytime/evening attendance": 1,
        "Previous qualification (grade)": 15.0,
        "Admission grade": 16.5,
        "Displaced": 0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Gender": 1,
        "Scholarship holder": 0,
        "Age at enrollment": 20,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 14.2,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.8,
        "Inflation rate": 1.4,
        "GDP": 1.74,
        # Marital status (solo uno debe ser 1)
        "Marital status_Divorced": 0,
        "Marital status_FactoUnion": 0,
        "Marital status_Separated": 0,
        "Marital status_Single": 1,
        # Application mode (solo uno debe ser 1)
        "Application mode_Admisión Especial": 0,
        "Application mode_Admisión Regular": 1,
        "Application mode_Admisión por Ordenanza": 0,
        "Application mode_Cambios/Transferencias": 0,
        "Application mode_Estudiantes Internacionales": 0,
        "Application mode_Mayores de 23 años": 0,
        # Course (solo uno debe ser 1)
        "Course_Agricultural & Environmental Sciences": 0,
        "Course_Arts & Design": 0,
        "Course_Business & Management": 1,
        "Course_Communication & Media": 0,
        "Course_Education": 0,
        "Course_Engineering & Technology": 0,
        "Course_Health Sciences": 0,
        "Course_Social Sciences": 0,
        # Previous qualification (solo uno debe ser 1)
        "Previous qualification_Higher Education": 0,
        "Previous qualification_Other": 0,
        "Previous qualification_Secondary Education": 1,
        "Previous qualification_Technical Education": 0,
        # Nationality (solo uno debe ser 1)
        "Nacionality_Colombian": 1,
        "Nacionality_Cuban": 0,
        "Nacionality_Dutch": 0,
        "Nacionality_English": 0,
        "Nacionality_German": 0,
        "Nacionality_Italian": 0,
        "Nacionality_Lithuanian": 0,
        "Nacionality_Moldovan": 0,
        "Nacionality_Mozambican": 0,
        "Nacionality_Portuguese": 0,
        "Nacionality_Romanian": 0,
        "Nacionality_Santomean": 0,
        "Nacionality_Turkish": 0,
        # Mother's qualification
        "Mother's qualification_Basic_or_Secondary": 1,
        "Mother's qualification_Other_or_Unknown": 0,
        "Mother's qualification_Postgraduate": 0,
        "Mother's qualification_Technical_Education": 0,
        # Father's qualification
        "Father's qualification_Basic_or_Secondary": 1,
        "Father's qualification_Other_or_Unknown": 0,
        "Father's qualification_Postgraduate": 0,
        # Mother's occupation
        "Mother's occupation_Administrative/Clerical": 0,
        "Mother's occupation_Skilled Manual Workers": 1,
        "Mother's occupation_Special Cases": 0,
        "Mother's occupation_Technicians/Associate Professionals": 0,
        "Mother's occupation_Unskilled Workers": 0,
        # Father's occupation
        "Father's occupation_Administrative/Clerical": 0,
        "Father's occupation_Professionals": 0,
        "Father's occupation_Skilled Manual Workers": 1,
        "Father's occupation_Special Cases": 0,
        "Father's occupation_Technicians/Associate Professionals": 0
    }
    
    return sample

# Ejemplo de uso en Streamlit
def main():
    st.title("🎓 Predictor de Deserción Académica")
    st.write("Sistema de predicción basado en XGBoost optimizado")
    
    # Botón para probar con datos de ejemplo
    if st.button("🧪 Probar con datos de ejemplo"):
        sample_data = create_sample_data()
        
        with st.spinner("Realizando predicción..."):
            resultado = make_prediction(sample_data)
        
        if resultado:
            st.success("✅ Predicción realizada exitosamente!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicción", resultado['prediction_label'])
                st.metric("Valor numérico", resultado['prediction'])
            
            with col2:
                if resultado['probabilities'] is not None:
                    prob_no_desercion = resultado['probabilities'][0] * 100
                    prob_desercion = resultado['probabilities'][1] * 100
                    
                    st.metric("Prob. No Deserción", f"{prob_no_desercion:.1f}%")
                    st.metric("Prob. Deserción", f"{prob_desercion:.1f}%")
            
            # Mostrar interpretación
            st.subheader("📊 Interpretación")
            if resultado['prediction'] == 1:
                st.warning("⚠️ **Alto riesgo de deserción**: Se recomienda intervención temprana")
            else:
                st.success("✅ **Bajo riesgo de deserción**: El estudiante tiene buenas perspectivas de continuidad")
        else:
            st.error("❌ Error en la predicción")
    
    # Información sobre el modelo
    st.subheader("ℹ️ Información del Modelo")
    st.info("""
    - **Modelo**: XGBoost optimizado con Optimización Bayesiana
    - **Métricas**: F1-score optimizado para balancear precisión y recall
    - **Preprocesamiento**: MinMaxScaler aplicado a variables numéricas
    - **Variables**: 69 características incluyendo datos académicos, socioeconómicos y demográficos
    """)

if __name__ == "__main__":
    main()