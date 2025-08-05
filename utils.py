"""
Utility functions for clinical data analysis application.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataValidator:
    """Validates and processes medical data with privacy considerations."""
    
    SENSITIVE_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # Credit card pattern
    ]
    
    @staticmethod
    def detect_sensitive_data(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect potentially sensitive data in the DataFrame."""
        sensitive_columns = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                sensitive_patterns_found = []
                sample_values = df[column].dropna().astype(str).head(100)
                
                for pattern in MedicalDataValidator.SENSITIVE_PATTERNS:
                    if any(re.search(pattern, str(val)) for val in sample_values):
                        sensitive_patterns_found.append(pattern)
                
                if sensitive_patterns_found:
                    sensitive_columns[column] = sensitive_patterns_found
        
        return sensitive_columns
    
    @staticmethod
    def anonymize_data(df: pd.DataFrame, columns_to_anonymize: List[str]) -> pd.DataFrame:
        """Anonymize sensitive columns while preserving data structure."""
        df_anon = df.copy()
        
        for col in columns_to_anonymize:
            if col in df_anon.columns:
                # Create consistent anonymized IDs
                unique_values = df_anon[col].unique()
                anonymization_map = {
                    val: f"ANON_{col.upper()}_{i:04d}" 
                    for i, val in enumerate(unique_values) if pd.notna(val)
                }
                df_anon[col] = df_anon[col].map(anonymization_map)
        
        return df_anon

class DataTypeInferencer:
    """Infers appropriate data types for medical data."""
    
    @staticmethod
    def infer_medical_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Infer data types with medical context awareness."""
        type_mapping = {}
        
        for column in df.columns:
            col_lower = column.lower()
            sample_data = df[column].dropna()
            
            if len(sample_data) == 0:
                type_mapping[column] = "empty"
                continue
            
            # Medical-specific type inference
            if any(keyword in col_lower for keyword in ['date', 'time', 'birth', 'admission', 'discharge']):
                type_mapping[column] = "datetime"
            elif any(keyword in col_lower for keyword in ['id', 'patient', 'mrn', 'record']):
                type_mapping[column] = "identifier"
            elif any(keyword in col_lower for keyword in ['age', 'weight', 'height', 'bmi', 'pressure', 'rate']):
                type_mapping[column] = "numeric_measurement"
            elif any(keyword in col_lower for keyword in ['gender', 'sex', 'race', 'ethnicity', 'status']):
                type_mapping[column] = "categorical"
            elif any(keyword in col_lower for keyword in ['diagnosis', 'procedure', 'medication', 'icd', 'cpt']):
                type_mapping[column] = "medical_code"
            elif any(keyword in col_lower for keyword in ['note', 'comment', 'description', 'text']):
                type_mapping[column] = "text"
            else:
                # General type inference
                if pd.api.types.is_numeric_dtype(sample_data):
                    if sample_data.dtype in ['int64', 'int32']:
                        type_mapping[column] = "integer"
                    else:
                        type_mapping[column] = "float"
                elif pd.api.types.is_datetime64_any_dtype(sample_data):
                    type_mapping[column] = "datetime"
                elif sample_data.dtype == 'bool':
                    type_mapping[column] = "boolean"
                else:
                    # Check if it's categorical based on unique values
                    unique_ratio = len(sample_data.unique()) / len(sample_data)
                    if unique_ratio < 0.1 and len(sample_data.unique()) < 50:
                        type_mapping[column] = "categorical"
                    else:
                        type_mapping[column] = "text"
        
        return type_mapping

class StatisticalSummarizer:
    """Generates statistical summaries for medical data."""
    
    @staticmethod
    def generate_column_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate comprehensive summary for a column."""
        col_data = df[column].dropna()
        
        summary = {
            "name": column,
            "total_count": len(df[column]),
            "non_null_count": len(col_data),
            "null_count": df[column].isnull().sum(),
            "null_percentage": (df[column].isnull().sum() / len(df[column])) * 100,
            "data_type": str(df[column].dtype),
            "unique_count": df[column].nunique(),
            "memory_usage": df[column].memory_usage(deep=True)
        }
        
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            summary.update({
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "outliers_count": len(col_data[(col_data < (col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))) | 
                                             (col_data > (col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))))])
            })
        
        elif pd.api.types.is_categorical_dtype(col_data) or df[column].dtype == 'object' or pd.api.types.is_bool_dtype(col_data):
            value_counts = col_data.value_counts()
            summary.update({
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "least_frequent": value_counts.index[-1] if len(value_counts) > 0 else None,
                "least_frequent_count": value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head().to_dict()
            })
        
        return summary

def safe_execute_code(code: str, data: pd.DataFrame, max_rows: int = 1000) -> Tuple[bool, Any, str]:
    """Safely execute generated code on data with error handling."""
    
    # Create a safe execution environment
    safe_globals = {
        'pd': pd,
        'np': np,
        'plt': None,  # Will import matplotlib.pyplot if needed
        'sns': None,  # Will import seaborn if needed
        'df': data.head(max_rows) if len(data) > max_rows else data.copy(),
        'data': data.head(max_rows) if len(data) > max_rows else data.copy(),
    }
    
    # Import plotting libraries if needed
    if 'plt.' in code or 'matplotlib' in code:
        import matplotlib.pyplot as plt
        safe_globals['plt'] = plt
    
    if 'sns.' in code or 'seaborn' in code:
        import seaborn as sns
        safe_globals['sns'] = sns
    
    try:
        # Compile the code first to check for syntax errors
        compiled_code = compile(code, '<string>', 'exec')
        
        # Execute in the safe environment
        local_vars = {}
        exec(compiled_code, safe_globals, local_vars)
        
        # Try to extract meaningful results
        result = None
        if 'result' in local_vars:
            result = local_vars['result']
        elif 'output' in local_vars:
            result = local_vars['output']
        else:
            # Look for any variable that might be the result
            for var_name, var_value in local_vars.items():
                if not var_name.startswith('_'):
                    result = var_value
                    break
        
        return True, result, "Code executed successfully"
    
    except SyntaxError as e:
        return False, None, f"Syntax Error: {str(e)}"
    except Exception as e:
        return False, None, f"Runtime Error: {str(e)}"

def format_number(num: float, decimal_places: int = 2) -> str:
    """Format numbers for display in medical context."""
    if pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1000000:
        return f"{num/1000000:.{decimal_places}f}M"
    elif abs(num) >= 1000:
        return f"{num/1000:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"

def detect_medical_concepts(text_or_columns) -> List[str]:
    """Detect medical concepts in text or list of column names for better analysis planning."""
    medical_keywords = {
        'demographics': ['age', 'gender', 'race', 'ethnicity', 'sex'],
        'vitals': ['blood pressure', 'heart rate', 'temperature', 'respiratory rate', 'weight', 'height', 'bmi'],
        'lab_results': ['glucose', 'cholesterol', 'hemoglobin', 'hematocrit', 'creatinine', 'bun'],
        'diagnoses': ['icd', 'diagnosis', 'condition', 'disease', 'disorder'],
        'medications': ['drug', 'medication', 'prescription', 'dosage'],
        'procedures': ['cpt', 'procedure', 'surgery', 'treatment'],
        'outcomes': ['mortality', 'readmission', 'length of stay', 'discharge', 'outcome']
    }
    
    # Handle both string and list inputs
    if isinstance(text_or_columns, list):
        # Join column names with spaces for keyword matching
        text_lower = ' '.join(text_or_columns).lower()
    else:
        text_lower = str(text_or_columns).lower()
    
    detected_concepts = []
    
    for concept, keywords in medical_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_concepts.append(concept)
    
    return detected_concepts

def create_data_preview(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Create a formatted preview of the data for AI analysis."""
    preview = f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
    preview += "Column Information:\n"
    
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        preview += f"- {col}: {dtype}, {null_count} nulls, {unique_count} unique values\n"
    
    preview += f"\nFirst {max_rows} rows:\n"
    preview += df.head(max_rows).to_string()
    
    return preview
