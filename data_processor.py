"""
Data processing and metadata extraction for clinical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from utils import (
    DataTypeInferencer, 
    StatisticalSummarizer,
    create_data_preview
)

logger = logging.getLogger(__name__)

class ClinicalDataProcessor:
    """Main class for processing clinical data files."""
    
    def __init__(self):
        self.inferencer = DataTypeInferencer()
        self.summarizer = StatisticalSummarizer()
        
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and perform initial processing of the uploaded file."""
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Initial data validation
            if df.empty:
                raise ValueError("The uploaded file is empty")
            
            # Basic cleaning
            df = self._basic_cleaning(df)
            
            # Generate metadata
            metadata = self._extract_metadata(df)
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # Handle unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        return df
    
    def _extract_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive metadata from the dataset."""
        
        # Basic information
        metadata = {
            "basic_info": {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "total_cells": df.shape[0] * df.shape[1],
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            }
        }
        
        # Data types inference
        metadata["data_types"] = self.inferencer.infer_medical_data_types(df)
        
        # Column-wise summaries
        metadata["column_summaries"] = {}
        for column in df.columns:
            metadata["column_summaries"][column] = self.summarizer.generate_column_summary(df, column)
        
        # Data quality assessment
        metadata["data_quality"] = self._assess_data_quality(df)
        
        # Statistical overview
        metadata["statistical_overview"] = self._generate_statistical_overview(df)
        
        return metadata
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_metrics = {
            "completeness": {},
            "consistency": {},
            "validity": {}
        }
        
        # Completeness assessment
        for col in df.columns:
            completeness = (df[col].count() / len(df)) * 100
            quality_metrics["completeness"][col] = {
                "percentage": completeness,
                "status": "good" if completeness >= 90 else "fair" if completeness >= 70 else "poor"
            }
        
        # Consistency assessment (looking for data type inconsistencies)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed data types in object columns
                sample_data = df[col].dropna().head(100)
                type_consistency = len(set(type(val).__name__ for val in sample_data)) == 1
                quality_metrics["consistency"][col] = {
                    "consistent": type_consistency,
                    "issues": [] if type_consistency else ["Mixed data types detected"]
                }
        
        # Validity assessment (basic checks)
        for col in df.columns:
            issues = []
            
            # Check for negative values in typically positive medical measurements
            if any(keyword in col.lower() for keyword in ['age', 'weight', 'height', 'pressure']):
                if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        try:
                            negative_count = (col_data < 0).sum()
                            if negative_count > 0:
                                issues.append(f"{negative_count} negative values in {col}")
                        except Exception:
                            # Skip negative value check if it fails
                            pass
            
            # Check for extreme outliers in medical measurements
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    try:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:  # Avoid division by zero
                            lower_bound = Q1 - 3 * IQR
                            upper_bound = Q3 + 3 * IQR
                            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                            if len(outliers) > 0:
                                issues.append(f"{len(outliers)} extreme outliers detected")
                    except Exception:
                        # Skip outlier detection if it fails
                        pass
            
            quality_metrics["validity"][col] = {
                "issues": issues,
                "status": "good" if len(issues) == 0 else "needs_review"
            }
        
        return quality_metrics
    
    def _generate_statistical_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical overview of the dataset."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        overview = {
            "numeric_columns": {
                "count": len(numeric_cols),
                "columns": list(numeric_cols),
                "summary": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
            },
            "categorical_columns": {
                "count": len(categorical_cols),
                "columns": list(categorical_cols),
                "summary": {}
            }
        }
        
        # Categorical summary
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            overview["categorical_columns"]["summary"][col] = {
                "unique_values": df[col].nunique(),
                "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_frequent_count": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "distribution": value_counts.head(10).to_dict()
            }
        
        return overview
    
    def prepare_data_for_analysis(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data package for analysis including privacy considerations."""
        
        df_processed = df.copy()
        
        # Create data preview for AI analysis
        data_preview = create_data_preview(df_processed, max_rows=20)
        
        # Prepare analysis package
        analysis_package = {
            "data": df_processed,
            "metadata": metadata,
            "data_preview": data_preview,
            "column_info": {
                col: {
                    "type": metadata["data_types"].get(col, "unknown"),
                    "summary": metadata["column_summaries"].get(col, {}),
                    "sample_values": df_processed[col].dropna().head(5).tolist() if col in df_processed.columns else []
                }
                for col in df_processed.columns
            }
        }
        
        return analysis_package
