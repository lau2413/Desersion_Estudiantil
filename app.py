import streamlit as st
import pandas as pd
import pickle
import numpy as np
from predictor import StudentDropoutPredictor
from utils import validate_csv_columns, format_prediction_result
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎓 Sistema de Predicción de Deserción Estudiantil")
st.markdown("---")

# Inicializar el predictor
@st.cache_resource
def load_predictor():
    """Carga el modelo y las columnas esperadas"""
    try:
        predictor = StudentDropoutPredictor()
        predictor.load_model()
        return predictor
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

predictor = load_predictor()

# Sidebar con información del modelo
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.info("""
    **Modelo de Predicción de Deserción Estudiantil**
    
    Este sistema utiliza un modelo de machine learning pre-entrenado para predecir la probabilidad de deserción de estudiantes basándose en diversos factores académicos y personales.
    """)
    
    st.markdown("### 📊 Variables del Modelo")
    st.markdown("""
    - **Datos académicos**: Calificaciones, unidades curriculares, evaluaciones
    - **Datos personales**: Edad, género, becas, situación financiera
    - **Datos institucionales**: Orden de aplicación, modalidad, desplazamiento
    """)

# Pestañas principales
tab1, tab2 = st.tabs(["👤 Predicción Individual", "📖 Guía de Uso"])

with tab1:
    st.header("👤 Predicción Individual")
    st.markdown("Introduce los datos de un estudiante para obtener una predicción individual.")
    
    # Formulario para predicción individual
    with st.form("prediction_form"):
        st.subheader("📝 Datos del Estudiante")
        
        # Organizar los campos en columnas
        col1, col2, col3 = st.columns(3)
        
        # Diccionario para almacenar los valores del formulario
        form_data = {}
        
        # Datos académicos principales
        with col1:
            st.markdown("**📚 Datos Académicos**")
            form_data['Application order'] = st.number_input("Orden de aplicación", min_value=1, max_value=20, value=1)
            form_data['Admission grade'] = st.number_input("Calificación de admisión (0-200)", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
            form_data['Previous qualification (grade)'] = st.number_input("Calificación previa (0-200)", min_value=0.0, max_value=200.0, value=150.0, step=0.1)
            form_data['Curricular units 2nd sem (grade)'] = st.number_input("Calificación 2do semestre (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            form_data['Curricular units 1st sem (evaluations)'] = st.number_input("Evaluaciones 1er semestre", min_value=0, max_value=20, value=6)
            form_data['Curricular units 2nd sem (evaluations)'] = st.number_input("Evaluaciones 2do semestre", min_value=0, max_value=20, value=6)
            
        with col2:
            st.markdown("**👤 Datos Personales**")
            form_data['Age at enrollment'] = st.number_input("Edad al inscribirse", min_value=16, max_value=70, value=20)
            form_data['Gender'] = st.selectbox("Género", [0, 1], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
            form_data['Daytime/evening attendance'] = st.selectbox("Asistencia", [0, 1], format_func=lambda x: "Nocturna" if x == 0 else "Diurna")
            form_data['Scholarship holder'] = st.selectbox("Becario", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            form_data['Tuition fees up to date'] = st.selectbox("Pagos al día", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            form_data['Displaced'] = st.selectbox("Desplazado", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            form_data['Debtor'] = st.selectbox("Deudor", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        
        with col3:
            st.markdown("**👨‍👩‍👧‍👦 Datos Familiares**")
            
            # Estado civil
            marital_options = ['Single', 'Divorced', 'FactoUnion', 'Separated']
            marital_status = st.selectbox("Estado civil", marital_options)
            for status in marital_options:
                form_data[f'Marital status_{status}'] = 1 if status == marital_status else 0
            
            # Educación de la madre
            mother_qual_options = ['Basic_or_Secondary', 'Technical_Education', 'Other_or_Unknown', 'Postgraduate']
            mother_qual = st.selectbox("Educación de la madre", mother_qual_options, 
                                     format_func=lambda x: {
                                         'Basic_or_Secondary': 'Básica o Secundaria',
                                         'Technical_Education': 'Educación Técnica', 
                                         'Other_or_Unknown': 'Otra o Desconocida',
                                         'Postgraduate': 'Postgrado'
                                     }[x])
            for qual in mother_qual_options:
                form_data[f"Mother's qualification_{qual}"] = 1 if qual == mother_qual else 0
            
            # Educación del padre (incluye todas las categorías)
            father_qual_options = ['Basic_or_Secondary', 'Other_or_Unknown', 'Postgraduate']
            father_qual = st.selectbox("Educación del padre", father_qual_options,
                                     format_func=lambda x: {
                                         'Basic_or_Secondary': 'Básica o Secundaria',
                                         'Other_or_Unknown': 'Otra o Desconocida', 
                                         'Postgraduate': 'Postgrado'
                                     }[x])
            for qual in father_qual_options:
                form_data[f"Father's qualification_{qual}"] = 1 if qual == father_qual else 0

        # Segunda fila de columnas para más variables principales
        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("**💼 Ocupación Familiar**")
            
            # Ocupación de la madre
            mother_occ_options = ['Administrative/Clerical', 'Skilled Manual Workers', 'Special Cases', 
                                'Technicians/Associate Professionals', 'Unskilled Workers']
            mother_occ = st.selectbox("Ocupación de la madre", mother_occ_options,
                                    format_func=lambda x: {
                                        'Administrative/Clerical': 'Administrativa/Oficina',
                                        'Skilled Manual Workers': 'Trabajadora Manual Calificada',
                                        'Special Cases': 'Casos Especiales',
                                        'Technicians/Associate Professionals': 'Técnica/Profesional Asociada',
                                        'Unskilled Workers': 'Trabajadora No Calificada'
                                    }[x])
            for occ in mother_occ_options:
                form_data[f"Mother's occupation_{occ}"] = 1 if occ == mother_occ else 0
            
            # Ocupación del padre
            father_occ_options = ['Administrative/Clerical', 'Professionals', 'Skilled Manual Workers', 
                                'Special Cases', 'Technicians/Associate Professionals']
            father_occ = st.selectbox("Ocupación del padre", father_occ_options,
                                    format_func=lambda x: {
                                        'Administrative/Clerical': 'Administrativa/Oficina',
                                        'Professionals': 'Profesional',
                                        'Skilled Manual Workers': 'Trabajador Manual Calificado',
                                        'Special Cases': 'Casos Especiales',
                                        'Technicians/Associate Professionals': 'Técnico/Profesional Asociado'
                                    }[x])
            for occ in father_occ_options:
                form_data[f"Father's occupation_{occ}"] = 1 if occ == father_occ else 0
        
        with col5:
            st.markdown("**🌍 Datos Institucionales**")
            
            # Nacionalidad
            nationality_options = ['Portuguese', 'Colombian', 'German', 'Italian', 'English', 'Other']
            nationality = st.selectbox("Nacionalidad", nationality_options,
                                     format_func=lambda x: {
                                         'Portuguese': 'Portuguesa',
                                         'Colombian': 'Colombiana', 
                                         'German': 'Alemana',
                                         'Italian': 'Italiana',
                                         'English': 'Inglesa',
                                         'Other': 'Otra'
                                     }[x])
            main_nationalities = ['Colombian', 'German', 'Italian', 'English', 'Portuguese']
            for nat in main_nationalities:
                form_data[f'Nacionality_{nat}'] = 1 if nat == nationality else 0
            
            # Modalidad de Admisión
            admission_options = ['Admisión Regular', 'Admisión Especial', 'Cambios/Transferencias', 'Mayores de 23 años']
            admission_mode = st.selectbox("Modalidad de admisión", admission_options)
            for mode in admission_options:
                form_data[f'Application mode_{mode}'] = 1 if mode == admission_mode else 0
                
            # Calificación previa (tipo)
            prev_qual_options = ['Secondary Education', 'Higher Education', 'Technical Education', 'Other']
            prev_qual_type = st.selectbox("Tipo de calificación previa", prev_qual_options,
                                        format_func=lambda x: {
                                            'Secondary Education': 'Educación Secundaria',
                                            'Higher Education': 'Educación Superior',
                                            'Technical Education': 'Educación Técnica',
                                            'Other': 'Otra'
                                        }[x])
            prev_qual_types = ['Higher Education', 'Other', 'Secondary Education', 'Technical Education']
            for ptype in prev_qual_types:
                form_data[f'Previous qualification_{ptype}'] = 1 if ptype == prev_qual_type else 0
                
        with col6:
            st.markdown("**💼 Área de Estudio**")
            
            course_options = ['Engineering & Technology', 'Business & Management', 'Health Sciences', 
                            'Social Sciences', 'Education', 'Arts & Design', 'Agricultural & Environmental Sciences',
                            'Communication & Media']
            course = st.selectbox("Área de estudio", course_options,
                                format_func=lambda x: {
                                    'Engineering & Technology': 'Ingeniería y Tecnología',
                                    'Business & Management': 'Negocios y Administración',
                                    'Health Sciences': 'Ciencias de la Salud',
                                    'Social Sciences': 'Ciencias Sociales',
                                    'Education': 'Educación',
                                    'Arts & Design': 'Artes y Diseño',
                                    'Agricultural & Environmental Sciences': 'Ciencias Agrícolas y Ambientales',
                                    'Communication & Media': 'Comunicación y Medios'
                                }[x])
            
            course_list = ['Agricultural & Environmental Sciences', 'Arts & Design', 'Business & Management',
                          'Communication & Media', 'Education', 'Engineering & Technology', 
                          'Health Sciences', 'Social Sciences']
            for course_name in course_list:
                form_data[f'Course_{course_name}'] = 1 if course_name == course else 0
        
        # Sección adicional solo para indicadores económicos
        with st.expander("📈 Indicadores Económicos (Opcional)"):
            col7, col8, col9 = st.columns(3)
            
            with col7:
                form_data['Unemployment rate'] = st.number_input("Tasa de desempleo (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
            with col8:
                form_data['Inflation rate'] = st.number_input("Tasa de inflación (%)", min_value=-5.0, max_value=20.0, value=2.0, step=0.1)
            with col9:
                form_data['GDP'] = st.number_input("GDP", min_value=-5.0, max_value=10.0, value=2.0, step=0.1)
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary")
        
        if submitted:
            try:
                # Crear un DataFrame con valores por defecto para las columnas faltantes
                student_data = predictor.create_default_student_data()
                
                # Actualizar con los valores del formulario
                for key, value in form_data.items():
                    if key in student_data.columns:
                        student_data[key] = value
                
                # Realizar predicción
                prediction = predictor.predict_single(student_data)
                
                # Mostrar resultados
                st.success("✅ Predicción completada")
                
                # Mostrar resultado principal
                result_text, color = format_prediction_result(
                    prediction['probability'], 
                    prediction['prediction'], 
                    prediction['risk_level']
                )
                
                st.markdown(f"### {result_text}")
                
                # Métricas detalladas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Probabilidad de Deserción",
                        f"{prediction['probability']:.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Clasificación",
                        "Deserción" if prediction['prediction'] == 1 else "No Deserción",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Nivel de Riesgo",
                        prediction['risk_level'],
                        delta=None
                    )
                
                # Recomendaciones basadas en el riesgo
                if prediction['risk_level'] == 'Alto':
                    st.warning("""
                    **⚠️ Recomendaciones para Riesgo Alto:**
                    - Implementar seguimiento académico personalizado
                    - Considerar apoyo psicopedagógico
                    - Revisar situación financiera del estudiante
                    - Establecer tutorías académicas
                    """)
                elif prediction['risk_level'] == 'Medio':
                    st.info("""
                    **💡 Recomendaciones para Riesgo Medio:**
                    - Monitoreo regular del rendimiento académico
                    - Facilitar acceso a recursos de apoyo
                    - Fomentar participación en actividades estudiantiles
                    """)
                else:
                    st.success("""
                    **✅ Estudiante con Bajo Riesgo:**
                    - Continuar con el seguimiento regular
                    - Mantener canales de comunicación abiertos
                    - Reconocer el buen desempeño académico
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")

with tab2:
    st.header("📖 Guía de Uso")
    
    st.markdown("""
    ## 🎯 Propósito del Sistema
    
    Este sistema de predicción de deserción estudiantil utiliza técnicas de machine learning para identificar estudiantes en riesgo de abandonar sus estudios, permitiendo intervenciones tempranas y personalizadas.
    
    ## 👤 Cómo usar la Predicción Individual
    
    1. **Completa el formulario** con los datos del estudiante
    2. **Haz clic en "Realizar Predicción"**
    3. **Analiza los resultados** y las recomendaciones
    4. **Implementa las acciones** sugeridas según el nivel de riesgo
    
    ## 📋 Variables Principales del Modelo
    
    ### Datos Académicos
    - **Calificación de admisión**: Nota obtenida en el proceso de admisión
    - **Calificación previa**: Calificación de estudios anteriores
    - **Calificaciones semestrales**: Rendimiento en cada semestre
    - **Unidades curriculares**: Número de materias evaluadas, aprobadas, etc.
    
    ### Datos Personales
    - **Edad al inscribirse**: Edad del estudiante al momento de la inscripción
    - **Género**: Identificación de género del estudiante
    - **Situación de beca**: Si el estudiante recibe apoyo financiero
    - **Estado de pagos**: Si las cuotas están al día
    
    ### Datos Institucionales
    - **Modalidad de asistencia**: Diurna o nocturna
    - **Orden de aplicación**: Preferencia en la postulación
    - **Desplazamiento**: Si el estudiante se desplaza para estudiar
    
    ## 🎯 Interpretación de Resultados
    
    ### Niveles de Riesgo
    - **🔴 Alto (≥70%)**: Requiere intervención inmediata
    - **🟡 Medio (30-70%)**: Necesita monitoreo cercano
    - **🟢 Bajo (<30%)**: Seguimiento regular
    
    ### Acciones Recomendadas
    
    **Para Riesgo Alto:**
    - Contacto inmediato con el estudiante
    - Evaluación de la situación académica y personal
    - Implementación de plan de apoyo personalizado
    - Seguimiento semanal
    
    **Para Riesgo Medio:**
    - Seguimiento quincenal
    - Acceso a recursos de apoyo académico
    - Monitoreo de indicadores clave
    
    **Para Riesgo Bajo:**
    - Seguimiento mensual regular
    - Mantenimiento de canales de comunicación
    - Reconocimiento del buen desempeño
    
    ## ⚠️ Consideraciones Importantes
    
    - Los resultados son **predictivos** y deben usarse como herramienta de apoyo
    - Siempre combinar con **evaluación humana profesional**
    - Mantener **confidencialidad** de los datos estudiantiles
    - Usar los resultados de manera **constructiva** y de apoyo
    
    ## 🔧 Soporte Técnico
    
    Si experimentas problemas técnicos:
    1. Revisa que los valores estén en los rangos esperados
    2. Contacta al administrador del sistema si persisten los problemas
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 14px;'>
    Sistema de Predicción de Deserción Estudiantil - Desarrollado con ❤️ usando Streamlit
</div>
""", unsafe_allow_html=True)
