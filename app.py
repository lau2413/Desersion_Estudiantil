# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo entrenado
@st.cache_resource
def cargar_modelo():
    return joblib.load("mejor_modelo_desercion.pkl")

modelo = cargar_modelo()

st.title("🎓 Predicción de Deserción Estudiantil")
st.markdown("Completa los datos del estudiante para predecir si existe riesgo de deserción.")

# DEBUG: Información del modelo
st.sidebar.subheader("🔍 Debug del Modelo")
st.sidebar.write(f"**Tipo de modelo:** {type(modelo)}")

if hasattr(modelo, 'steps'):
    st.sidebar.write("**Pipeline steps:**")
    for i, (name, step) in enumerate(modelo.steps):
        st.sidebar.write(f"{i+1}. {name}: {type(step).__name__}")
        
        # Si es ColumnTransformer, mostrar más detalles
        if hasattr(step, 'transformers_'):
            st.sidebar.write("   Transformers:")
            for j, (trans_name, transformer, cols) in enumerate(step.transformers_):
                st.sidebar.write(f"   - {trans_name}: {type(transformer).__name__}")
                st.sidebar.write(f"     Columnas: {cols}")

if hasattr(modelo, 'feature_names_in_'):
    st.sidebar.write(f"**Total features esperadas:** {len(modelo.feature_names_in_)}")
    with st.sidebar.expander("Ver nombres de features"):
        for i, feature in enumerate(modelo.feature_names_in_):
            st.write(f"{i+1}. {feature}")

# Formulario de entrada
with st.form("formulario"):
    st.subheader("🧑‍🎓 Información general")
    col1, col2 = st.columns(2)

    with col1:
        application_order = st.slider("Application order (0=1ra opción)", 0, 9, 1)
        attendance = st.radio("Horario", ["Diurno", "Vespertino"])
        prev_grade = st.number_input("Nota previa", 0.0, 200.0, 120.0)
        admission_grade = st.number_input("Nota de admisión", 0.0, 200.0, 130.0)
        age = st.slider("Edad al ingresar", 17, 60, 22)
        gender = st.radio("Género", ["Mujer", "Hombre"])

    with col2:
        displaced = st.radio("¿Desplazado?", ["No", "Sí"])
        debtor = st.radio("¿Moroso?", ["No", "Sí"])
        tuition_paid = st.radio("¿Pago al día?", ["No", "Sí"])
        scholarship = st.radio("¿Becado?", ["No", "Sí"])
        unemployment = st.slider("Tasa de desempleo (%)", 0.0, 25.0, 7.5)
        inflation = st.slider("Inflación (%)", 0.0, 15.0, 3.0)
        gdp = st.slider("PIB (GDP)", 0.0, 20.0, 5.0)

    st.subheader("📚 Rendimiento académico")
    col3, col4 = st.columns(2)

    with col3:
        eval1 = st.number_input("Evaluaciones 1er semestre", 0, 20, 5)
        noeval1 = st.number_input("Sin evaluación 1er semestre", 0, 10, 0)
        eval2 = st.number_input("Evaluaciones 2do semestre", 0, 20, 5)
        noeval2 = st.number_input("Sin evaluación 2do semestre", 0, 10, 0)

    with col4:
        credited2 = st.number_input("Créditos 2do semestre", 0, 20, 6)
        enrolled2 = st.number_input("Inscritas 2do semestre", 0, 20, 6)
        approved2 = st.number_input("Aprobadas 2do semestre", 0, 20, 4)
        grade2 = st.number_input("Nota 2do semestre", 0.0, 20.0, 13.0)

    st.subheader("📌 Selección de categoría")

    marital = st.selectbox("Estado civil", ["Divorced", "FactoUnion", "Separated", "Single"])
    app_mode = st.selectbox("Modalidad de ingreso", [
        "Admisión Especial", "Admisión Regular", "Admisión por Ordenanza",
        "Cambios/Transferencias", "Estudiantes Internacionales", "Mayores de 23 años"
    ])
    course = st.selectbox("Curso", [
        "Agricultural & Environmental Sciences", "Arts & Design", "Business & Management",
        "Communication & Media", "Education", "Engineering & Technology",
        "Health Sciences", "Social Sciences"
    ])
    prev_qual = st.selectbox("Tipo de formación previa", [
        "Higher Education", "Other", "Secondary Education", "Technical Education"
    ])
    nacionality = st.selectbox("Nacionalidad", [
        "Colombian", "Cuban", "Dutch", "English", "German", "Italian", "Lithuanian",
        "Moldovan", "Mozambican", "Portuguese", "Romanian", "Santomean", "Turkish"
    ])
    mq = st.selectbox("Nivel educativo de la madre", [
        "Basic_or_Secondary", "Other_or_Unknown", "Postgraduate", "Technical_Education"
    ])
    fq = st.selectbox("Nivel educativo del padre", [
        "Basic_or_Secondary", "Other_or_Unknown", "Postgraduate"
    ])
    mo = st.selectbox("Ocupación de la madre", [
        "Administrative/Clerical", "Skilled Manual Workers", "Special Cases",
        "Technicians/Associate Professionals", "Unskilled Workers"
    ])
    fo = st.selectbox("Ocupación del padre", [
        "Administrative/Clerical", "Professionals", "Skilled Manual Workers",
        "Special Cases", "Technicians/Associate Professionals"
    ])

    submit = st.form_submit_button("Predecir")

if submit:
    try:
        st.subheader("🔍 Proceso de Debug")
        
        # MÉTODO 1: Intentar recrear exactamente como en entrenamiento
        st.write("**Paso 1:** Recreando datos como en entrenamiento...")
        
        # Definir solo las variables numéricas (sin crear dummies)
        datos_numericos = {
            "Application order": application_order,
            "Daytime/evening attendance": 1 if attendance == "Diurno" else 0,
            "Previous qualification (grade)": prev_grade,
            "Admission grade": admission_grade,
            "Displaced": 1 if displaced == "Sí" else 0,
            "Debtor": 1 if debtor == "Sí" else 0,
            "Tuition fees up to date": 1 if tuition_paid == "Sí" else 0,
            "Gender": 1 if gender == "Hombre" else 0,
            "Scholarship holder": 1 if scholarship == "Sí" else 0,
            "Age at enrollment": age,
            "Curricular units 1st sem (evaluations)": eval1,
            "Curricular units 1st sem (without evaluations)": noeval1,
            "Curricular units 2nd sem (credited)": credited2,
            "Curricular units 2nd sem (enrolled)": enrolled2,
            "Curricular units 2nd sem (evaluations)": eval2,
            "Curricular units 2nd sem (approved)": approved2,
            "Curricular units 2nd sem (grade)": grade2,
            "Curricular units 2nd sem (without evaluations)": noeval2,
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp
        }
        
        # Variables categóricas
        datos_categoricos = {
            "Marital status": marital,
            "Application mode": app_mode,
            "Course": course,
            "Previous qualification": prev_qual,
            "Nacionality": nacionality,
            "Mother's qualification": mq,
            "Father's qualification": fq,
            "Mother's occupation": mo,
            "Father's occupation": fo
        }
        
        # Combinar todos los datos
        todos_los_datos = {**datos_numericos, **datos_categoricos}
        
        # Crear DataFrame
        X_new = pd.DataFrame([todos_los_datos])
        
        st.write(f"**DataFrame creado:** {X_new.shape}")
        st.write("**Tipos de datos:**")
        st.dataframe(X_new.dtypes.to_frame().T)
        
        st.write("**Primeras filas:**")
        st.dataframe(X_new.head())
        
        # Verificar que tenemos todas las columnas esperadas
        if hasattr(modelo, 'feature_names_in_'):
            missing_cols = set(modelo.feature_names_in_) - set(X_new.columns)
            extra_cols = set(X_new.columns) - set(modelo.feature_names_in_)
            
            if missing_cols:
                st.error(f"**Columnas faltantes:** {missing_cols}")
                
                # Agregar columnas faltantes con 0s
                for col in missing_cols:
                    X_new[col] = 0
                st.info("Columnas faltantes agregadas con valor 0")
            
            if extra_cols:
                st.warning(f"**Columnas extra:** {extra_cols}")
                # Remover columnas extra
                X_new = X_new[modelo.feature_names_in_]
                st.info("Columnas extra removidas")
            
            # Reordenar columnas según el modelo
            X_new = X_new[modelo.feature_names_in_]
            st.success("Columnas reordenadas según el modelo")
        
        st.write("**DataFrame final:**")
        st.write(f"Forma: {X_new.shape}")
        st.dataframe(X_new.head())
        
        # Intentar predicción
        st.write("**Paso 2:** Intentando predicción...")
        
        pred = modelo.predict(X_new)[0]
        proba = modelo.predict_proba(X_new)[0][1]

        # Mostrar resultado
        st.subheader("📈 Resultado de la predicción:")
        if pred == 1:
            st.error(f"🚨 El estudiante tiene riesgo de **deserción**.\n\nProbabilidad: {proba:.2%}")
        else:
            st.success(f"✅ El estudiante **no tiene riesgo de deserción**.\n\nProbabilidad: {proba:.2%}")
            
    except Exception as e:
        st.error(f"**Error en la predicción:** {str(e)}")
        st.error(f"**Tipo de error:** {type(e).__name__}")
        
        # Debug más profundo
        st.subheader("🚨 Debug del Error")
        
        import traceback
        st.code(traceback.format_exc())
        
        # Intentar diagnóstico
        st.write("**Diagnóstico:**")
        
        if hasattr(modelo, 'steps'):
            preprocessor = modelo.steps[0][1]  # Asumir que preprocessor es el primer step
            
            st.write(f"Preprocessor type: {type(preprocessor)}")
            
            if hasattr(preprocessor, 'transformers_'):
                st.write("Transformers en el preprocessor:")
                for name, transformer, cols in preprocessor.transformers_:
                    st.write(f"- {name}: {type(transformer).__name__} para columnas {cols}")
            
            # Intentar solo el preprocessor
            try:
                st.write("**Intentando solo preprocessor...**")
                X_transformed = preprocessor.transform(X_new)
                st.success(f"Preprocessor OK. Shape después: {X_transformed.shape}")
            except Exception as prep_error:
                st.error(f"Error en preprocessor: {prep_error}")
        
        # Información adicional del DataFrame
        st.write("**Info del DataFrame:**")
        st.write(f"Forma: {X_new.shape}")
        st.write(f"Columnas: {list(X_new.columns)}")
        st.write(f"Tipos: {X_new.dtypes.to_dict()}")
        
        # Verificar valores únicos en categóricas
        categorical_cols = ["Marital status", "Application mode", "Course", "Previous qualification", 
                          "Nacionality", "Mother's qualification", "Father's qualification", 
                          "Mother's occupation", "Father's occupation"]
        
        st.write("**Valores en columnas categóricas:**")
        for col in categorical_cols:
            if col in X_new.columns:
                st.write(f"{col}: {X_new[col].iloc[0]} (tipo: {type(X_new[col].iloc[0])})")
