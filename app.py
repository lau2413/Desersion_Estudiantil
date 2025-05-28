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
                units_1sem_eval = st.number_input("Unidades evaluadas (1er sem)", min_value=0, max_value=45, value=5)
            with col2b:
                units_1sem_noeval = st.number_input("Unidades no evaluadas (1er sem)", min_value=0, max_value=12, value=0)
            
            # Segundo semestre
            st.markdown("**Segundo Semestre**")
            col2c, col2d = st.columns(2)
            with col2c:
                units_2sem_credited = st.number_input("Unidades con crédito (2do sem)", min_value=0, max_value=19, value=0)
                units_2sem_enrolled = st.number_input("Unidades matriculadas (2do sem)", min_value=0, max_value=23, value=6)
                units_2sem_eval = st.number_input("Unidades evaluadas (2do sem)", min_value=0, max_value=33, value=6)
            with col2d:
                units_2sem_approved = st.number_input("Unidades aprobadas (2do sem)", min_value=0, max_value=20, value=4)
                units_2sem_grade = st.number_input("Nota media (2do sem)", min_value=0.0, max_value=18.57, value=12.0)
                units_2sem_noeval = st.number_input("Unidades no evaluadas (2do sem)", min_value=0, max_value=12, value=0)
    
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
            mother_qualification = st.radio("Titulación de la madre", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate", "Technical_Education"],
                                          horizontal=True)
            
            mother_occupation = st.selectbox("Ocupación de la madre", options=[
                "Administrative/Clerical", "Skilled Manual Workers",
                "Special Cases", "Technicians/Associate Professionals",
                "Unskilled Workers"
            ])
        
        with col2:
            st.markdown("**Padre**")
            father_qualification = st.radio("Titulación del padre", 
                                          options=["Basic_or_Secondary", "Other_or_Unknown", 
                                                  "Postgraduate"],
                                          horizontal=True)
            
            father_occupation = st.selectbox("Ocupación del padre", options=[
                "Administrative/Clerical", "Professionals",
                "Skilled Manual Workers", "Special Cases",
                "Technicians/Associate Professionals"
            ])
    
        # Nacionalidad (movido dentro del tab4 pero fuera de las columnas)
        nationality = st.selectbox("Nacionalidad", options=[
            "Colombian", "Cuban", "Dutch", "English", "German",
            "Italian", "Lithuanian", "Moldovan", "Mozambican",
            "Portuguese", "Romanian", "Santomean", "Turkish"
        ])
    
    submitted = st.form_submit_button("Predecir Riesgo de Deserción")

# Función para crear el input correcto
def create_input_data():
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
    
    # Actualizar los valores básicos
    input_data.update({
        # Información personal
        'Age at enrollment': age,
        'Gender': gender,
        'Displaced': int(displaced),
        'Debtor': int(debtor),
        'Tuition fees up to date': int(tuition_up_to_date),
        'Scholarship holder': int(scholarship),
        
        # Datos académicos
        'Application order': application_order,
        'Daytime/evening attendance': daytime_attendance,
        'Previous qualification (grade)': prev_qualification_map.get(prev_qualification, 1),
        'Admission grade': admission_grade,
        
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
    })
    
    # Variables categóricas - Estado civil
    marital_status_cols = [f'Marital status_{status}' for status in ["Single", "Divorced", "FactoUnion", "Separated"]]
    for col in marital_status_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f'Marital status_{marital_status}' else 0
    
    # Variables categóricas - Modo de aplicación
    application_mode_map = {
        "Admisión Regular": "Admisión Regular",
        "Admisión Especial": "Admisión Especial",
        "Admisión por Ordenanza": "Admisión por Ordenanza",
        "Cambios/Transferencias": "Cambios/Transferencias",
        "Estudiantes Internacionales": "Estudiantes Internacionales",
        "Mayores de 23 años": "Mayores de 23 años"
    }
    
    for mode in application_mode_map.values():
        col_name = f'Application mode_{mode}'
        if col_name in columnas_esperadas:
            input_data[col_name] = 1 if mode == application_mode else 0
    
    # Variables categóricas - Curso
    course_cols = [f'Course_{c}' for c in ["Agricultural & Environmental Sciences", "Arts & Design",
                                           "Business & Management", "Communication & Media",
                                           "Education", "Engineering & Technology",
                                           "Health Sciences", "Social Sciences"]]
    for col in course_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f'Course_{course}' else 0
    
    # Variables categóricas - Titulación de la madre
    mother_qual_cols = [f"Mother's qualification_{qual}" for qual in ["Basic_or_Secondary", "Other_or_Unknown", 
                                                                     "Postgraduate", "Technical_Education"]]
    for col in mother_qual_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f"Mother's qualification_{mother_qualification}" else 0
    
    # Variables categóricas - Ocupación de la madre
    mother_occ_cols = [f"Mother's occupation_{occ.replace(' ', '_')}" for occ in [
        "Administrative/Clerical", "Skilled Manual Workers", "Special Cases", 
        "Technicians/Associate Professionals", "Unskilled Workers"]]
    for col in mother_occ_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f"Mother's occupation_{mother_occupation.replace(' ', '_')}" else 0
    
    # Variables categóricas - Titulación del padre
    father_qual_cols = [f"Father's qualification_{qual}" for qual in ["Basic_or_Secondary", "Other_or_Unknown", "Postgraduate"]]
    for col in father_qual_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f"Father's qualification_{father_qualification}" else 0
    
    # Variables categóricas - Ocupación del padre
    father_occ_cols = [f"Father's occupation_{occ.replace(' ', '_')}" for occ in [
        "Administrative/Clerical", "Professionals", "Skilled Manual Workers", 
        "Special Cases", "Technicians/Associate Professionals"]]
    for col in father_occ_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f"Father's occupation_{father_occupation.replace(' ', '_')}" else 0
    
    # Variables categóricas - Nacionalidad
    nationality_cols = [f'Nacionality_{nat}' for nat in ["Colombian", "Cuban", "Dutch", "English", "German",
                                                        "Italian", "Lithuanian", "Moldovan", "Mozambican",
                                                        "Portuguese", "Romanian", "Santomean", "Turkish"]]
    for col in nationality_cols:
        if col in columnas_esperadas:
            input_data[col] = 1 if col == f'Nacionality_{nationality}' else 0
    
    return input_data

# Cuando se envía el formulario
if submitted:
    try:
        # Crear los datos de entrada
        input_data = create_input_data()
        
        # Convertir a DataFrame con el orden correcto de columnas
        input_df = pd.DataFrame([input_data])
        
        # Asegurar que todas las columnas esperadas estén presentes
        for col in columnas_esperadas:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reordenar columnas según el orden esperado
        input_df = input_df[columnas_esperadas]
        
        # Convertir tipos de datos
        input_df = input_df.astype(float)
        
        # Hacer la predicción
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
        st.error(f"Tipo de error: {type(e).__name__}")
        
        # Información adicional de debug
        st.write("**Información de debug:**")
        st.write(f"Columnas esperadas: {len(columnas_esperadas) if columnas_esperadas else 'No cargadas'}")
        if 'input_df' in locals():
            st.write(f"Columnas en input_df: {len(input_df.columns)}")

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