import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

def validate_csv_columns(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
    """
    Valida que un DataFrame contenga las columnas esperadas
    
    Args:
        df: DataFrame a validar
        expected_columns: Lista de columnas esperadas
    
    Returns:
        Diccionario con información de validación
    """
    df_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing_columns = list(expected_set - df_columns)
    extra_columns = list(df_columns - expected_set)
    
    is_valid = len(missing_columns) == 0
    
    return {
        "is_valid": is_valid,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "total_columns_df": len(df.columns),
        "total_expected_columns": len(expected_columns)
    }

def format_prediction_result(probability: float, prediction: int, risk_level: str) -> Tuple[str, str]:
    """
    Formatea el resultado de la predicción para mostrar al usuario
    
    Args:
        probability: Probabilidad de deserción (0-1)
        prediction: Predicción binaria (0 o 1)
        risk_level: Nivel de riesgo (Alto, Medio, Bajo)
    
    Returns:
        Tupla con (texto_resultado, color)
    """
    percentage = probability * 100
    
    if risk_level == "Alto":
        color = "#ff4b4b"  # Rojo
        icon = "🔴"
        text = f"{icon} **RIESGO ALTO DE DESERCIÓN** ({percentage:.1f}%)"
    elif risk_level == "Medio":
        color = "#ffa500"  # Naranja
        icon = "🟡"
        text = f"{icon} **RIESGO MEDIO DE DESERCIÓN** ({percentage:.1f}%)"
    else:
        color = "#00cc00"  # Verde
        icon = "🟢"
        text = f"{icon} **RIESGO BAJO DE DESERCIÓN** ({percentage:.1f}%)"
    
    return text, color

def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y convierte datos numéricos en un DataFrame
    
    Args:
        df: DataFrame a limpiar
    
    Returns:
        DataFrame limpio
    """
    cleaned_df = df.copy()
    
    # Convertir columnas numéricas
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Intentar convertir a numérico
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Rellenar valores nulos con 0
    cleaned_df = cleaned_df.fillna(0)
    
    return cleaned_df

def get_summary_statistics(df: pd.DataFrame, predictions: Dict[str, List]) -> Dict[str, Any]:
    """
    Calcula estadísticas resumidas de las predicciones
    
    Args:
        df: DataFrame original
        predictions: Diccionario con predicciones
    
    Returns:
        Diccionario con estadísticas
    """
    total_students = len(df)
    predicted_dropouts = sum(predictions['predictions'])
    avg_probability = np.mean(predictions['probabilities'])
    
    risk_counts = {}
    for level in ['Alto', 'Medio', 'Bajo']:
        risk_counts[level] = predictions['risk_levels'].count(level)
    
    return {
        'total_students': total_students,
        'predicted_dropouts': predicted_dropouts,
        'predicted_continuers': total_students - predicted_dropouts,
        'avg_dropout_probability': avg_probability,
        'risk_distribution': risk_counts,
        'dropout_rate': predicted_dropouts / total_students if total_students > 0 else 0
    }

def create_risk_recommendations(risk_level: str) -> str:
    """
    Genera recomendaciones basadas en el nivel de riesgo
    
    Args:
        risk_level: Nivel de riesgo (Alto, Medio, Bajo)
    
    Returns:
        String con recomendaciones
    """
    recommendations = {
        "Alto": """
        **🚨 ACCIONES INMEDIATAS REQUERIDAS:**
        
        • **Contacto urgente**: Reunión con el estudiante en máximo 48 horas
        • **Evaluación integral**: Revisar situación académica, financiera y personal
        • **Plan de intervención**: Crear estrategia personalizada de apoyo
        • **Seguimiento intensivo**: Reuniones semanales hasta estabilizar la situación
        • **Recursos especializados**: Derivar a servicios de apoyo psicopedagógico
        • **Flexibilización académica**: Considerar opciones de cronograma modificado
        """,
        
        "Medio": """
        **⚠️ MONITOREO Y APOYO PREVENTIVO:**
        
        • **Seguimiento regular**: Reuniones quincenales con tutor académico
        • **Identificación de factores**: Detectar causas específicas de riesgo
        • **Apoyo académico**: Facilitar acceso a tutorías y recursos de estudio
        • **Motivación**: Fomentar participación en actividades estudiantiles
        • **Comunicación familiar**: Involucrar red de apoyo cuando sea apropiado
        • **Alerta temprana**: Monitorear indicadores clave de rendimiento
        """,
        
        "Bajo": """
        **✅ MANTENIMIENTO Y PREVENCIÓN:**
        
        • **Seguimiento estándar**: Reuniones mensuales de rutina
        • **Reconocimiento**: Destacar y celebrar el buen desempeño
        • **Desarrollo integral**: Ofrecer oportunidades de crecimiento académico
        • **Comunicación abierta**: Mantener canales disponibles para consultas
        • **Prevención**: Estar atento a cambios en el comportamiento académico
        • **Mentorías**: Considerar al estudiante como mentor para otros
        """
    }
    
    return recommendations.get(risk_level, "No hay recomendaciones disponibles para este nivel de riesgo.")

def export_predictions_to_csv(df: pd.DataFrame, predictions: Dict[str, List]) -> str:
    """
    Prepara los datos para exportar a CSV
    
    Args:
        df: DataFrame original
        predictions: Diccionario con predicciones
    
    Returns:
        String CSV para descarga
    """
    export_df = df.copy()
    
    # Agregar columnas de predicción
    export_df['Probabilidad_Desercion'] = predictions['probabilities']
    export_df['Prediccion_Desercion'] = predictions['predictions']
    export_df['Clasificacion'] = ['Deserción' if pred == 1 else 'No Deserción' 
                                  for pred in predictions['predictions']]
    export_df['Nivel_Riesgo'] = predictions['risk_levels']
    
    # Agregar timestamp
    from datetime import datetime
    export_df['Fecha_Prediccion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return export_df.to_csv(index=False)

def validate_data_ranges(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Valida que los datos estén en rangos razonables
    
    Args:
        df: DataFrame a validar
    
    Returns:
        Diccionario con advertencias de validación
    """
    warnings = []
    
    # Validaciones específicas por tipo de campo
    validations = {
        'age': {'min': 16, 'max': 70, 'columns': ['Age at enrollment']},
        'grade': {'min': 0, 'max': 20, 'columns': [col for col in df.columns if 'grade' in col.lower()]},
        'qualification': {'min': 0, 'max': 200, 'columns': [col for col in df.columns if 'qualification' in col.lower()]},
        'binary': {'min': 0, 'max': 1, 'columns': ['Gender', 'Scholarship holder', 'Debtor']}
    }
    
    for validation_type, rules in validations.items():
        for col in rules['columns']:
            if col in df.columns:
                out_of_range = df[(df[col] < rules['min']) | (df[col] > rules['max'])]
                if len(out_of_range) > 0:
                    warnings.append(f"Columna '{col}': {len(out_of_range)} valores fuera del rango esperado ({rules['min']}-{rules['max']})")
    
    return {'warnings': warnings}
