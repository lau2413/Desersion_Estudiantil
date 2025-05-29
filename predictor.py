import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Tuple
import streamlit as st

class StudentDropoutPredictor:
    """Clase para manejar las predicciones de deserción estudiantil"""
    
    def __init__(self):
        self.model = None
        self.expected_columns = None
        self.is_loaded = False
    
    def load_model(self):
        """Carga el modelo y las columnas esperadas desde los archivos pickle"""
        try:
            # Cargar el modelo con diferentes protocolos de pickle
            model_path = "attached_assets/pipeline_final_desercion.pkl"
            if os.path.exists(model_path):
                try:
                    # Intentar cargar con protocolo actual
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except (pickle.UnpicklingError, TypeError, AttributeError) as e:
                    # Si falla, intentar con encoding latin-1 para compatibilidad con versiones anteriores
                    try:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f, encoding='latin-1')
                    except Exception as e2:
                        # Como último recurso, usar joblib si está disponible
                        try:
                            import joblib
                            self.model = joblib.load(model_path)
                        except Exception as e3:
                            raise Exception(f"No se pudo cargar el modelo con ningún método: pickle error: {e}, latin-1 error: {e2}, joblib error: {e3}")
                
                st.success("✅ Modelo cargado exitosamente")
            else:
                raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
            
            # Cargar las columnas esperadas
            columns_path = "attached_assets/columnas_esperadas.pkl"
            if os.path.exists(columns_path):
                with open(columns_path, 'rb') as f:
                    self.expected_columns = pickle.load(f)
                st.success("✅ Columnas esperadas cargadas exitosamente")
            else:
                raise FileNotFoundError(f"No se encontró el archivo de columnas: {columns_path}")
            
            self.is_loaded = True
            
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            raise e
    
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Valida que los datos de entrada tengan las columnas correctas"""
        if self.expected_columns is None:
            return False, ["Columnas esperadas no cargadas"]
        
        missing_columns = []
        for col in self.expected_columns:
            if col not in data.columns:
                missing_columns.append(col)
        
        return len(missing_columns) == 0, missing_columns
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa los datos para que coincidan con el formato esperado por el modelo"""
        processed_data = data.copy()
        
        # Asegurar que todas las columnas esperadas estén presentes
        for col in self.expected_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0  # Valor por defecto
        
        # Reordenar columnas según el orden esperado
        processed_data = processed_data[self.expected_columns]
        
        # Convertir a tipos numéricos
        for col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Rellenar valores NaN con 0
        processed_data = processed_data.fillna(0)
        
        return processed_data
    
    def predict_single(self, student_data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza una predicción para un solo estudiante"""
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido cargado")
        
        # Validar datos
        is_valid, missing_cols = self.validate_input_data(student_data)
        if not is_valid:
            st.warning(f"Algunas columnas están ausentes: {missing_cols}. Usando valores por defecto.")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(student_data)
        
        # Realizar predicción
        try:
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0][1]  # Probabilidad de la clase positiva (deserción)
            
            # Determinar nivel de riesgo
            risk_level = self._get_risk_level(probability)
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level
            }
        except Exception as e:
            raise ValueError(f"Error en la predicción: {str(e)}")
    
    def predict_batch(self, data: pd.DataFrame) -> Dict[str, List]:
        """Realiza predicciones para múltiples estudiantes"""
        if not self.is_loaded:
            raise ValueError("El modelo no ha sido cargado")
        
        # Validar datos
        is_valid, missing_cols = self.validate_input_data(data)
        if not is_valid:
            st.warning(f"Algunas columnas están ausentes: {missing_cols}. Usando valores por defecto.")
        
        # Preprocesar datos
        processed_data = self.preprocess_data(data)
        
        # Realizar predicciones
        try:
            predictions = self.model.predict(processed_data)
            probabilities = self.model.predict_proba(processed_data)[:, 1]  # Probabilidad de deserción
            
            # Determinar niveles de riesgo
            risk_levels = [self._get_risk_level(prob) for prob in probabilities]
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'risk_levels': risk_levels
            }
        except Exception as e:
            raise ValueError(f"Error en las predicciones: {str(e)}")
    
    def _get_risk_level(self, probability: float) -> str:
        """Determina el nivel de riesgo basado en la probabilidad"""
        if probability >= 0.7:
            return "Alto"
        elif probability >= 0.3:
            return "Medio"
        else:
            return "Bajo"
    
    def create_default_student_data(self) -> pd.DataFrame:
        """Crea un DataFrame con valores por defecto para todos los campos requeridos"""
        if self.expected_columns is None:
            raise ValueError("Columnas esperadas no cargadas")
        
        # Crear DataFrame con valores por defecto
        default_values = {}
        for col in self.expected_columns:
            # Asignar valores por defecto basados en el nombre de la columna
            if 'grade' in col.lower() or 'qualification' in col.lower():
                default_values[col] = 0.0
            elif 'age' in col.lower():
                default_values[col] = 20
            elif 'order' in col.lower():
                default_values[col] = 1
            elif any(keyword in col.lower() for keyword in ['units', 'sem', 'curricular']):
                default_values[col] = 0
            else:
                default_values[col] = 0
        
        return pd.DataFrame([default_values])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información sobre el modelo cargado"""
        if not self.is_loaded:
            return {"loaded": False}
        
        try:
            model_type = type(self.model).__name__
            num_features = len(self.expected_columns) if self.expected_columns else 0
            
            return {
                "loaded": True,
                "model_type": model_type,
                "num_features": num_features,
                "expected_columns": self.expected_columns
            }
        except Exception as e:
            return {"loaded": True, "error": str(e)}
