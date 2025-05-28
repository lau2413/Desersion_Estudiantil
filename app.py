import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Deserción Universitaria", layout="wide")

# Título de la aplicación
st.title("Sistema de Predicción de Deserción Universitaria")
st.markdown("""
Complete el formulario con la información del estudiante para predecir el riesgo de deserción.
""")

# Cargar el modelo y las columnas esperadas
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('data/pipeline_final_desercion.pkl')
        columnas = joblib.load('data/columnas_esperadas.pkl')
        return pipeline, columnas
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")
        return None, None

pipeline, columnas_esperadas = load_model()

if pipeline is None:
    st.stop()

# Crear formulario para la entrada de datos
with st.form("student_form"):
    # Dividir el formulario en pestañas para mejor organización
    tab1, tab2, tab3, tab4 = st.tabs(["Información Personal", "Datos Académicos", "Información Económica", "Historial Familiar"])
    
    with tab1:
        st.subheader("Información Personal")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=20)
            gender = st.radio("Género", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
            marital_status = st.radio("Estado civil", 
                                     options=["Single", "Divorced", "FactoUnion", "Separated"],
                                     horizontal=True)
            
        with col2:
            displaced = st.checkbox("Desplazado", value=False)
            debtor = st.checkbox("Deudor", value=False)
            tuition_up_to_date = st.checkbox("Matrícula al día", value=True)
            scholarship = st.checkbox("Becado", value=False)
    
    with tab2:
        st.subheader("Datos Académicos")
        col1, col2 = st.columns(2)
        
        with col1:
            application_order = st.slider("Orden de aplicación", 0, 9, 1)
            daytime_attendance = st.radio("Asistencia", options=[1, 0], 
                                        format_func=lambda x: "Diurna" if x == 1 else "Nocturna",
                                        horizontal=True)
            
            prev_qualification = st.selectbox("Titulación previa", options=[
                "Secondary education", "Higher education - bachelor's degree",
                "Higher education - degree", "Higher education - master's",
                "Higher education - doctorate", "Frequency of higher education",
                "12th year of schooling - not completed", "11th year of schooling - not completed",
                "Other - 11th year of schooling", "10th year of schooling",
                "10th year of schooling - not completed", "Basic education 3rd cycle",
                "Basic education 2nd cycle", "Technological specialization course",
                "Higher education - degree (1st cycle)", "Professional higher technical course",
                "Higher education - master (2nd cycle)"
            ])
            
            admission_grade = st.number_input("Nota de admisión", min_value=0.0, max_value=200.0, value=120.0)
            
        with col2:
            application_mode = st.selectbox("Modo de aplicación", options=[
                "Admisión Regular", "Admisión Especial", 
                "Admisión por Ordenanza", "Cambios/Transferencias",
                "Estudiantes Internacionales", "Mayores de 23 años"
            ])
            
            course = st.selectbox("Curso", options=[
                "Agricultural & Environmental Sciences", "Arts & Design",
                "Business & Management", "Communication & Media",
                "Education", "Engineering & Technology",
                "Health Sciences", "Social Sciences"
            ])
            
            # Primer semestre
            st.markdown("**Primer Semestre**")
            col2a, col2b = st.columns(2)
            with col2a:
                units_1sem_eval = st.number_input("Unidades evaluadas", min_value=0, max_value=45, value=5)
            with col2b:
                units_1sem_noeval = st.number_input("Unidades no evaluadas", min_value=0, max_value=12, value=0)
            
            # Segundo semestre
            st.markdown("**Segundo Semestre**")
            col2c, col2d = st.columns(2)
            with col2c:
                units_2sem_credited = st.number_input("Unidades con crédito", min_value=0, max_value=19, value=0)
                units_2sem_enrolled = st.number_input("Unidades matriculadas", min_value=0, max_value=23, value=6)
                units_2sem_eval = st.number_input("Unidades evaluadas", min_value=0, max_value=33, value=6)
            with col2d:
                units_2sem_approved = st.number_input("Unidades aprobadas", min_value=0, max_value=20, value=4)
                units_2sem_grade = st.number_input("Nota media", min_value=0.0, max_value=18.57, value=12.0)
                units_2sem_noeval = st.number_input("Unidades no evaluadas", min_value=0, max_value=12, value=0)
    
    with tab3:
        st.subheader("Información Económica")
        col1, col2 = st.columns(2)
        
        with col1:
            unemployment = st.slider("Tasa de desempleo (%)", min_value=7.6, max_value=16.2, value=10.0)
            inflation = st.slider("Tasa de inflación (%)", min_value=-0.8, max_value=3.7, value=1.5)
        
        with col2:
            gdp = st.slider("GDP", min_value=-4.06, max_value=3.51, value=0.0)
    
    with tab4:
        st.subheader("Historial Familiar")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Madre**")
            mother_qualification = st.radio("Titulación", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate", "Technical_Education"],
                                          horizontal=True)
            
            mother_occupation = st.selectbox("Ocupación", options=[
                "Administrative/Clerical", "Skilled Manual Workers",
                "Special Cases", "Technicians/Associate Professionals",
                "Unskilled Workers"
            ])
        
        with col2:
            st.markdown("**Padre**")
            father_qualification = st.radio("Titulación", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate"],
                                          horizontal=True)
            
            father_occupation = st.selectbox("Ocupación", options=[
                "Administrative/Clerical", "Professionals",
                "Skilled Manual Workers", "Special Cases",
                "Technicians/Associate Professionals"
            ])
    
    # Nacionalidad (en la última pestaña)
    with tab4:
        nationality = st.selectbox("Nacionalidad", options=[
            "Colombian", "Cuban", "Dutch", "English", "German",
            "Italian", "Lithuanian", "Moldovan", "Mozambican",
            "Portuguese", "Romanian", "Santomean", "Turkish"
        ])
    
    submitted = st.form_submit_button("Predecir Riesgo de Deserción")

# Cuando se envía el formulario
if submitted:
    # Crear un diccionario con todas las columnas esperadas inicializadas a 0
    input_data = {col: 0 for col in columnas_esperadas}
    
    # Mapeo de titulaciones previas a códigos numéricos
    prev_qualification_map = {
        "Secondary education": 1,
        "Higher education - bachelor's degree": 2,
        "Higher education - degree": 3,
        "Higher education - master's": 4,
        "Higher education - doctorate": 5,
        "Frequency of higher education": 6,
        "12th year of schooling - not completed": 9,
        "11th year of schooling - not completed": 10,
        "Other - 11th year of schooling": 12,
        "10th year of schooling": 14,
        "10th year of schooling - not completed": 15,
        "Basic education 3rd cycle": 19,
        "Basic education 2nd cycle": 38,
        "Technological specialization course": 39,
        "Higher education - degree (1st cycle)": 40,
        "Professional higher technical course": 42,
        "Higher education - master (2nd cycle)": 43
    }
    
    # Actualizar los valores ingresados
    input_data.update({
        # Información personal
        'Age at enrollment': age,
        'Gender': gender,
        'Displaced': int(displaced),
        'Debtor': int(debtor),
        'Tuition fees up to date': int(tuition_up_to_date),
        'Scholarship holder': int(scholarship),
        f'Marital status_{marital_status}': 1,
        
        # Datos académicos
        'Application order': application_order,
        'Daytime/evening attendance': daytime_attendance,
        'Previous qualification (grade)': prev_qualification_map[prev_qualification],
        'Admission grade': admission_grade,
        f'Application mode_{application_mode.replace(" ", "_")}': 1,
        f'Course_{course}': 1,
        
        # Primer semestre
        'Curricular units 1st sem (evaluations)': units_1sem_eval,
        'Curricular units 1st sem (without evaluations)': units_1sem_noeval,
        
        # Segundo semestre
        'Curricular units 2nd sem (credited)': units_2sem_credited,
        'Curricular units 2nd sem (enrolled)': units_2sem_enrolled,
        'Curricular units 2nd sem (evaluations)': units_2sem_eval,
        'Curricular units 2nd sem (approved)': units_2sem_approved,
        'Curricular units 2nd sem (grade)': units_2sem_grade,
        'Curricular units 2nd sem (without evaluations)': units_2sem_noeval,
        
        # Información económica
        'Unemployment rate': unemployment,
        'Inflation rate': inflation,
        'GDP': gdp,
        
        # Historial familiar
        f'Mother\'s qualification_{mother_qualification}': 1,
        f'Mother\'s occupation_{mother_occupation.replace(" ", "_")}': 1,
        f'Father\'s qualification_{father_qualification}': 1,
        f'Father\'s occupation_{father_occupation.replace(" ", "_")}': 1,
        
        # Nacionalidad
        f'Nacionality_{nationality}': 1
    })
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([input_data])[columnas_esperadas]
    
    # Hacer la predicción
    try:
        prediction_proba = pipeline.predict_proba(input_df)[0][1]  # Probabilidad de deserción
        prediction_percent = round(prediction_proba * 100, 2)
        
        # Mostrar resultados
        st.subheader("Resultado de la Predicción")
        
        # Barra de progreso y métrica
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad de Deserción", f"{prediction_percent}%")
        with col2:
            st.progress(prediction_proba)
        
        # Interpretación
        if prediction_proba > 0.7:
            st.error("🚨 Alto riesgo de deserción. Se recomienda intervención inmediata.")
        elif prediction_proba > 0.4:
            st.warning("⚠️ Riesgo moderado de deserción. Se sugiere monitoreo cercano.")
        else:
            st.success("✅ Bajo riesgo de deserción.")
        
        # Mostrar los datos ingresados (opcional)
        with st.expander("Ver datos técnicos enviados al modelo"):
            st.dataframe(input_df.T.rename(columns={0: "Valor"}))
            
    except Exception as e:
        st.error(f"Error al hacer la predicción: {str(e)}")

# Información adicional en el sidebar
st.sidebar.markdown("""
### Instrucciones:
1. Complete todas las pestañas del formulario.
2. Revise que los datos sean correctos.
3. Haga clic en **Predecir Riesgo de Deserción**.
4. Interprete los resultados según el nivel de riesgo.

### Notas:
- Los campos booleanos se convierten automáticamente (0=False, 1=True)
- Para variables categóricas, seleccione solo una opción
""")