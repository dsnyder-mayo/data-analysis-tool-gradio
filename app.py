"""
Gradio Web Application for Clinical Data Analysis
A professional interface for medical researchers and doctors to analyze clinical data.
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import logging
import html
import ast
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import base64
import mimetypes
from io import BytesIO
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import plotly.io as pio
from dotenv import load_dotenv
load_dotenv()


# Import your backend modules
from clinical_analyzer import ClinicalDataAnalyzer
from data_processor import ClinicalDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalAnalysisApp:
    """Main Gradio application class for clinical data analysis."""
    
    def __init__(self):
        self.analyzer = None
        self.current_data = None
        self.current_metadata = None
        self.analysis_results = None
        self.generated_questions = []
        
        # Plan confirmation state
        self.current_plan = None
        self.current_question = None
        self.data_package = None
        self.plan_confirmed = False
        
        self.sample_questions = [
            "What are the key demographic characteristics of this patient population?",
            "Are there any significant correlations between clinical variables?",
            "What patterns can you identify in the diagnostic codes?",
            "Analyze the distribution of vital signs and identify any outliers",
            "What insights can you derive from the temporal patterns in this data?",
            "Identify potential risk factors based on the available clinical variables",
            "Compare outcomes across different patient subgroups",
            "What quality issues or missing data patterns should we be aware of?"
        ]
    
    def process_uploaded_file(self, file):
        """Process uploaded file and return data summary and generated questions."""
        if file is None:
            return ("No file uploaded", 
                    "", 
                    "", 
                    gr.HTML(visible=False), 
                    gr.HTML(visible=False),
                    gr.Dataframe(visible=False))
        
        try:
            # Initialize data processor
            processor = ClinicalDataProcessor()
            if self.analyzer is None:
                try:
                    self.analyzer = ClinicalDataAnalyzer()
                except Exception as e:
                    return (f"❌ Failed to initialize analyzer: {str(e)}", 
                            "", 
                            "", 
                            gr.HTML(visible=False), 
                            gr.HTML(visible=False),
                            gr.Dataframe(visible=False))

            # Load and process data
            df, metadata = processor.load_data(file.name)
            self.current_data = df
            self.current_metadata = metadata
            
            # Prepare data for analysis (needed for question generation)
            data_package = processor.prepare_data_for_analysis(df, metadata)
            
            # Store in analyzer for question generation
            if not hasattr(self.analyzer, 'current_analysis'):
                self.analyzer.current_analysis = {}
            self.analyzer.current_analysis["data_package"] = data_package
            
            # Generate data summary
            summary = self._generate_data_summary(df, metadata)
            
            # Generate suggested questions using DSPy if analyzer is available
            try:
                suggested_questions = self.analyzer.question_generator(data_package)
                # Ensure we have a list of strings
                if isinstance(suggested_questions, dict) and 'questions' in suggested_questions:
                    suggested_questions = suggested_questions['questions']
                elif not isinstance(suggested_questions, list):
                    suggested_questions = self.sample_questions
                
                # Limit to 8 questions for display
                suggested_questions = suggested_questions[:8] if len(suggested_questions) > 8 else suggested_questions
                
                # Store generated questions for later use
                self.generated_questions = suggested_questions
                
            except Exception as e:
                logger.error(f"Error generating questions: {str(e)}")
                suggested_questions = self.sample_questions
                self.generated_questions = self.sample_questions

            logger.info(f"Successfully processed file: {file.name}")
            
            # Create the title and question buttons
            title_html = "<h4 style='color: #495057; margin: 20px 0 10px 0;'>💡 Suggested Questions</h4>"
            questions_html = self._create_question_buttons_html(suggested_questions)
            
            return (summary, 
                    title_html, 
                    questions_html, 
                    gr.HTML(visible=True), 
                    gr.HTML(visible=True),
                    gr.Dataframe(value=df, visible=True))
            
        except Exception as e:
            error_msg = f"❌ Error processing file: {str(e)}"
            logger.error(error_msg)
            return (error_msg, 
                    "", 
                    "", 
                    gr.HTML(visible=False), 
                    gr.HTML(visible=False),
                    gr.Dataframe(visible=False))
    
    def _generate_data_summary(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Generate a comprehensive data summary."""
        basic_info = metadata.get("basic_info", {})
        
        # Count column types directly from the dataframe
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        # Calculate missing data percentage correctly
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        summary_parts = [
            "# 📊 Dataset Summary",
            f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n",
            f"**Missing Data:** {missing_percentage:.2f}% overall ({missing_cells:,} out of {total_cells:,} cells)\n",
            f"**Numeric Columns:** {numeric_cols}\n",
            f"**Categorical Columns:** {categorical_cols}\n",
            f"**DateTime Columns:** {datetime_cols}",
        ]
        
        # Add data quality insights
        # data_quality = metadata.get("data_quality", {})
        # if data_quality:
        #     completeness_data = data_quality.get("completeness", {})
        #     good_quality_cols = sum(1 for col_data in completeness_data.values() 
        #                           if col_data.get("status") == "good")
            # summary_parts.extend([
            #     "",
            #     "## ⚕️ Data Quality Insights",
            #     f"**High Quality Columns:** {good_quality_cols} out of {df.shape[1]} columns",
            #     f"**Average Completeness:** {100 - missing_percentage:.1f}%"
            # ])
        return "\n".join(summary_parts)
    
            
    def analyze_data_with_question_streaming(self, question: str, progress=gr.Progress()):
        """Generator version that yields intermediate results for streaming updates."""
        if not question.strip():
            yield "❌ Please enter a question", "", None, gr.Plot(visible=False)
            return
        
        if self.current_data is None:
            yield "❌ Please upload data first", "", None, gr.Plot(visible=False)
            return
        
        # Initialize analyzer with default settings if not already initialized
        if self.analyzer is None:
            try:
                self.analyzer = ClinicalDataAnalyzer()
            except Exception as e:
                yield f"❌ Failed to initialize analyzer: {str(e)}", "", None, gr.Plot(visible=False)
                return
        
        try:
            progress(0, desc="Starting analysis...")
            
            # Clear previous analysis results to prevent persistence
            self.analysis_results = None
            
            # Save current data to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                self.current_data.to_csv(tmp_file.name, index=False)
                temp_filepath = tmp_file.name
            
            # Step 1: Generate and show plan immediately
            progress(0.1, desc="Generating analysis plan...")
            
            try:
                # Load and process data
                df, metadata = self.analyzer.data_processor.load_data(temp_filepath)
                data_package = self.analyzer.data_processor.prepare_data_for_analysis(df, metadata)
                
                # Generate analysis plan
                plan_result = self.analyzer.planner(question, data_package)
                
                # Format plan for immediate display
                plan_html = self._format_analysis_plan_from_result(plan_result)
                
                # Show plan while continuing with execution
                yield plan_html, "<div style='text-align: center; padding: 20px; color: #6c757d;'><p>⏳ Executing analysis steps...</p></div>", None, gr.Plot(visible=False)
                
                progress(0.3, desc="Executing analysis steps...")
                
                # Continue with full analysis
                results = self.analyzer.analyze_data(temp_filepath, question)
                self.analysis_results = results
                
                progress(0.8, desc="Formatting final results...")
                
                if results.get("success"):
                    # Update plan with execution status and show final outputs
                    final_plan_html = self._format_analysis_plan(results)
                    outputs_result = self._format_analysis_outputs(results)
                    
                    # Handle different return types from _format_analysis_outputs
                    if isinstance(outputs_result, tuple):
                        # We have both HTML and Plotly figure
                        outputs_html, plotly_fig = outputs_result
                        progress(1.0, desc="Analysis complete!")
                        yield final_plan_html, outputs_html, plotly_fig, gr.Plot(visible=True)
                    else:
                        # Only HTML content
                        progress(1.0, desc="Analysis complete!")
                        yield final_plan_html, outputs_result, None, gr.Plot(visible=False)
                else:
                    error_msg = f"❌ Analysis failed: {results.get('error', 'Unknown error')}"
                    yield plan_html, error_msg, None, gr.Plot(visible=False)
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filepath)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            yield f"❌ Analysis error: {str(e)}", "", None, gr.Plot(visible=False)
    
    def _format_analysis_plan(self, results: Dict[str, Any]) -> str:
        """Format the analysis plan and execution status in HTML."""
        plan_info = results.get("analysis_plan", {})
        if not plan_info:
            return "<p>No analysis plan available.</p>"
        
        html_parts = [
            "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>",
            "<h3 style='color: #495057; margin-bottom: 15px; font-size: 1.4em;'>📋 Analysis Plan & Execution</h3>",
            
            # Execution status
            "<div style='display: flex; gap: 15px; margin-bottom: 20px;'>",
            f"<div style='background: #d4edda; padding: 10px 15px; border-radius: 5px; text-align: center; flex: 1;'>",
            f"<div style='font-size: 1.5em; font-weight: bold; color: #155724;'>{plan_info.get('successful_steps', 0)}</div>",
            f"<div style='color: #155724; font-size: 0.9em;'>Successful Steps</div>",
            "</div>",
            f"<div style='background: #f8d7da; padding: 10px 15px; border-radius: 5px; text-align: center; flex: 1;'>",
            f"<div style='font-size: 1.5em; font-weight: bold; color: #721c24;'>{plan_info.get('failed_steps', 0)}</div>",
            f"<div style='color: #721c24; font-size: 0.9em;'>Failed Steps</div>",
            "</div>",
            f"<div style='background: #cce5ff; padding: 10px 15px; border-radius: 5px; text-align: center, flex: 1;'>",
            f"<div style='font-size: 1.5em; font-weight: bold; color: #004085;'>{plan_info.get('total_steps', 0)}</div>",
            f"<div style='color: #004085; font-size: 0.9em;'>Total Steps</div>",
            "</div>",
            "</div>",
            
            # Planned steps
            "<h4 style='color: #495057; margin-bottom: 10px;'>Planned Analysis Steps:</h4>"
        ]
        
        # Display planned steps
        plan_details = plan_info.get("plan_details", [])
        if plan_details:
            for i, step in enumerate(plan_details, 1):
                step_title = step.get('title', 'Analysis Step')
                step_description = step.get('description', 'No description available')
                execution_trajectory = step.get('execution_trajectory', {})
                
                # Check if this step produces visualizations
                produce_result = step.get('display_to_user', False)
                success = execution_trajectory.get('success', False)
                
                # Choose border color and icon based on visualization output and execution status
                if produce_result:
                    if success:
                        border_color = "#28a745"  # Green for successful visualization steps
                        step_icon = "✅📊"
                    else:
                        border_color = "#dc3545"  # Red for failed visualization steps
                        step_icon = "❌📊"
                    viz_indicator = f"<span style='background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 8px;'>{step.get('output_format', '')}</span>"
                else:
                    if success:
                        border_color = "#6c757d"  # Gray for successful analysis steps
                        step_icon = "✅🔍"
                    else:
                        border_color = "#dc3545"  # Red for failed analysis steps
                        step_icon = "❌🔍"
                    viz_indicator = ""
                
                # Create unique ID for this step's toggle
                toggle_id = f"trajectory_toggle_{i}"
                
                html_parts.extend([
                    f"<div style='background: white; padding: 12px; border-radius: 5px; margin: 8px 0; border-left: 3px solid {border_color};'>",
                    f"<h5 style='color: #495057; margin: 0 0 5px 0;'>{step_icon} Step {i}: {step_title}{viz_indicator}</h5>",
                    f"<p style='margin: 0 0 10px 0; color: #6c757d; font-size: 0.9em;'>{step_description}</p>",
                ])
                
                # Add trajectory section if execution information is available
                if execution_trajectory:
                    html_parts.extend([
                        f"<details style='margin-top: 10px;'>",
                        f"<summary style='cursor: pointer; color: #007bff; font-weight: 500; font-size: 0.9em; padding: 5px 0;'>🔍 View Execution Details & Code</summary>",
                        f"<div style='margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #007bff;'>",
                    ])
                    
                    # Display overall execution summary first
                    reasoning = execution_trajectory.get('reasoning', '')
                    if reasoning:
                        html_parts.extend([
                            "<div style='margin-bottom: 15px; padding: 10px; background: #e8f4f8; border-radius: 4px; border-left: 3px solid #17a2b8;'>",
                            "<h6 style='color: #17a2b8; margin: 0 0 8px 0; font-weight: bold;'>📝 Execution Summary</h6>",
                            f"<div style='color: #0c5460; font-size: 0.9em; line-height: 1.4;'>{self._escape_html(reasoning)}</div>",
                            "</div>"
                        ])
                    
                    # Display intermediate output if available
                    intermediate_output = execution_trajectory.get('intermediate_output', None)
                    if intermediate_output is not None:
                        output_html = self._format_intermediate_output(intermediate_output)
                        html_parts.extend([
                            "<details style='margin-bottom: 15px;'>",
                            "<summary style='cursor: pointer; color: #28a745; font-weight: 500; font-size: 0.9em; padding: 5px 0;'>📊 View Step Output</summary>",
                            "<div style='margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #28a745;'>",
                            output_html,
                            "</div>",
                            "</details>"
                        ])
                    
                    # Format and display detailed trajectory iterations
                    trajectory_html = self._format_trajectory(execution_trajectory.get('trajectory', ''))
                    html_parts.append(trajectory_html)
                    
                    html_parts.extend([
                        "</div>",
                        "</details>"
                    ])
                
                html_parts.append("</div>")
        else:
            html_parts.append("<p style='color: #6c757d;'>No detailed plan available.</p>")
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _format_analysis_plan_from_result(self, plan_result: Dict[str, Any]) -> str:
        """Format the analysis plan directly from planner result."""
        if not plan_result or "plan" not in plan_result:
            return "<p>No analysis plan available.</p>"
        
        plan_steps = plan_result["plan"]
        
        html_parts = [
            "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>",
            "<h3 style='color: #495057; margin-bottom: 15px; font-size: 1.4em;'>📋 Analysis Plan</h3>",
            
            # Plan overview
            "<div style='background: #e3f2fd; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px;'>",
            f"<div style='color: #1565c0; font-weight: bold;'>📊 {len(plan_steps)} Analysis Steps Planned</div>",
            "<div style='color: #1976d2; font-size: 0.9em; margin-top: 5px;'>Plan generated - execution starting...</div>",
            "</div>",
            
            # Planned steps
            "<h4 style='color: #495057; margin-bottom: 10px;'>Planned Analysis Steps:</h4>"
        ]
        
        # Display planned steps
        if plan_steps:
            for i, step in enumerate(plan_steps, 1):
                step_title = step.get('title', 'Analysis Step')
                step_description = step.get('description', 'No description available')
                
                # Check if this step produces visualizations
                produces_viz = any(keyword in step_title.lower() or keyword in step_description.lower() 
                                 for keyword in ['plot', 'chart', 'graph', 'visualization', 'visualize', 'histogram', 'scatter', 'distribution', 'correlation matrix'])
                
                # Choose border color and icon based on visualization output
                if produces_viz:
                    border_color = "#28a745"  # Green for visualization steps
                    step_icon = "📊"
                    viz_indicator = "<span style='background: #d4edda; color: #155724; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 8px;'>📈 Visualization</span>"
                else:
                    border_color = "#6c757d"  # Gray for analysis steps
                    step_icon = "🔍"
                    viz_indicator = ""
                
                html_parts.extend([
                    f"<div style='background: white; padding: 12px; border-radius: 5px; margin: 8px 0; border-left: 3px solid {border_color}; position: relative;'>",
                    f"<div style='position: absolute; top: 8px; right: 8px; background: #f8f9fa; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; color: #6c757d;'>Pending</div>",
                    f"<h5 style='color: #495057; margin: 0 0 5px 0; padding-right: 80px;'>{step_icon} Step {i}: {step_title}{viz_indicator}</h5>",
                    f"<p style='margin: 0; color: #6c757d; font-size: 0.9em;'>{step_description}</p>",
                    "</div>"
                ])
        else:
            html_parts.append("<p style='color: #6c757d;'>No detailed plan available.</p>")
        
        html_parts.append("</div>")
        return "".join(html_parts)
    
    def _handle_single_output(self, output: Any) -> str:
        """Handle a single output item and return formatted HTML."""
        # Skip module objects (like pandas, matplotlib modules)
        if hasattr(output, '__module__') and hasattr(output, '__file__'):
            return ""  # Don't display module objects
        
        # Handle matplotlib figures
        if hasattr(output, 'savefig'):
            return self._format_matplotlib_figure(output)
        
        # Handle matplotlib axes objects
        elif hasattr(output, 'figure') and hasattr(output.figure, 'savefig'):
            return self._format_matplotlib_figure(output.figure)
        
        # Handle Plotly Express objects (plotly.graph_objects.Figure)
        elif hasattr(output, 'show') and hasattr(output, 'data') and hasattr(output, 'layout'):
            # Return the figure object directly for Gradio to handle
            # Note: This returns the actual figure object, not a string
            return self._format_plotly_figure(output)
        
        # Handle base64 image strings
        elif isinstance(output, str):
            # Check if it's a data URL with base64 image
            if self._is_data_url_image(output):
                return f"""
                <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                    <img src='{output}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                </div>
                """
            # Check if it's a base64 image string (without data URL prefix)
            elif self._is_base64_image(output):
                return f"""
                <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                    <img src='data:image/png;base64,{output}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                </div>
                """
            # Check if it's a filename ending in image extension
            elif output.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                return self._format_image_file(output)
            # Check if it's a potential data URL that wasn't caught above
            elif output.lower().startswith('data:image/') and 'base64,' in output.lower():
                return f"""
                <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                    <img src='{output}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                </div>
                """
            # Check if it's a long string that might be base64 but doesn't match our pattern
            elif len(output) > 1000 and not any(char in output for char in ['\n', ' ', '\t']) and output.replace('/', '').replace('+', '').replace('=', '').isalnum():
                return f"<div style='margin: 8px 0; padding: 8px; background: #fff3cd; border-radius: 4px; color: #856404; border-left: 3px solid #ffc107;'>🖼️ Possible encoded image data (truncated display)</div>"
            # Regular text output
            else:
                escaped_output = output.replace('<', '&lt;').replace('>', '&gt;')
                formatted_output = escaped_output.replace('\n', '<br>')
                return f"<div style='margin: 8px 0; padding: 12px; background: #f8f9fa; border-radius: 4px; font-family: monospace; white-space: pre-wrap; border-left: 3px solid #007bff;'>{formatted_output}</div>"
        
        # Handle pandas DataFrames
        elif hasattr(output, 'to_html'):
            try:
                html_table = output.to_html(classes='table table-striped table-hover', table_id='data-table', escape=False, max_rows=100)
                return f"<div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; overflow-x: auto;'>{html_table}</div>"
            except:
                return f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 DataFrame ({output.shape[0]} rows × {output.shape[1]} columns)</div>"
        
        # Handle dictionaries (often contain plot data or results)
        elif isinstance(output, dict):
            # Check if it's a plotly figure dictionary
            if 'data' in output and 'layout' in output:
                return "<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 Plotly Figure (interactive display not supported in this view)</div>"
            # Check if it contains image data
            elif any(key in output for key in ['image', 'figure', 'plot', 'base64']):
                # Try to extract image data
                for key in ['image', 'figure', 'plot', 'base64']:
                    if key in output:
                        img_data = output[key]
                        if isinstance(img_data, str):
                            # Check for data URL format first
                            if self._is_data_url_image(img_data):
                                return f"""
                                <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                                    <img src='{img_data}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                                </div>
                                """
                            # Check for base64 image without data URL prefix
                            elif self._is_base64_image(img_data):
                                return f"""
                                <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                                    <img src='data:image/png;base64,{img_data}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                                </div>
                                """
                        elif hasattr(img_data, 'savefig'):
                            return self._format_matplotlib_figure(img_data)
                # If no image data found, display as formatted dict
                return self._format_dict_output(output)
            else:
                return self._format_dict_output(output)
        
        # Handle tuples (often contain multiple outputs)
        elif isinstance(output, tuple):
            # Handle mixed content - collect all results
            results = []
            plotly_figures = []
            for item in output:
                result = self._handle_single_output(item)
                if isinstance(result, str) and result:  # Only add non-empty string results
                    results.append(result)
                elif hasattr(result, 'show') and hasattr(result, 'data') and hasattr(result, 'layout'):
                    # Collect Plotly figures separately
                    plotly_figures.append(result)
            
            # If we have Plotly figures, return the first one (Gradio can only display one at a time)
            # and convert others to HTML
            if plotly_figures:
                html_content = "".join(results)
                for i, fig in enumerate(plotly_figures):
                    if i == 0:
                        # Return the first figure directly if no HTML content
                        if not html_content:
                            return fig
                        else:
                            # Convert to HTML if we have mixed content
                            html_content += self._format_plotly_figure_as_html(fig)
                    else:
                        html_content += self._format_plotly_figure_as_html(fig)
                return html_content
            else:
                return "".join(results)
        
        # Handle lists (often contain multiple outputs)
        elif isinstance(output, list):
            # Handle mixed content - collect all results
            results = []
            plotly_figures = []
            for item in output:
                result = self._handle_single_output(item)
                if isinstance(result, str) and result:  # Only add non-empty string results
                    results.append(result)
                elif hasattr(result, 'show') and hasattr(result, 'data') and hasattr(result, 'layout'):
                    # Collect Plotly figures separately
                    plotly_figures.append(result)
            
            # If we have Plotly figures, return the first one (Gradio can only display one at a time)
            # and convert others to HTML
            if plotly_figures:
                html_content = "".join(results)
                for i, fig in enumerate(plotly_figures):
                    if i == 0:
                        # Return the first figure directly if no HTML content
                        if not html_content:
                            return fig
                        else:
                            # Convert to HTML if we have mixed content
                            html_content += self._format_plotly_figure_as_html(fig)
                    else:
                        html_content += self._format_plotly_figure_as_html(fig)
                return html_content
            else:
                return "".join(results)
        
        # Handle numeric values
        elif isinstance(output, (int, float)):
            return f"<div style='margin: 8px 0; padding: 12px; background: #e8f5e8; border-radius: 4px; font-weight: bold; border-left: 3px solid #28a745;'>📊 Result: {output}</div>"
        
        # Skip None values
        elif output is None:
            return ""
        
        # Handle other object types (but skip modules)
        else:
            # Check if it's a module (don't display)
            if hasattr(output, '__module__') and hasattr(output, '__name__'):
                return ""
            
            # Try to convert to string representation
            try:
                str_repr = str(output)
                if len(str_repr) > 200:
                    str_repr = str_repr[:200] + "..."
                return f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 {type(output).__name__}: {str_repr}</div>"
            except:
                return f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 {type(output).__name__} object</div>"
    
    def _is_base64_image(self, s: str) -> bool:
        """Check if a string is a base64 encoded image."""
        if not isinstance(s, str) or len(s) < 50:  # Base64 images are typically much longer
            return False
        
        # Remove any whitespace
        s = s.strip()
        
        # If it's a data URL, extract just the base64 part
        if s.lower().startswith('data:image/') and 'base64,' in s.lower():
            try:
                s = s.split('base64,', 1)[1]
            except:
                return False
        
        # Check for common base64 image prefixes (PNG, JPEG, GIF, etc.)
        if s.startswith(('iVBORw0KGgo', '/9j/', 'R0lGOD', 'UklGR', 'Qk0', 'SUkq')):
            return True
        
        # Try to decode as base64 and check if it starts with image headers
        try:
            # Only decode a small portion to check headers
            decoded = base64.b64decode(s[:100])  # Just check the beginning
            
            # PNG signature
            if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return True
            # JPEG signature
            elif decoded.startswith(b'\xff\xd8\xff'):
                return True
            # GIF signature
            elif decoded.startswith((b'GIF87a', b'GIF89a')):
                return True
            # BMP signature
            elif decoded.startswith(b'BM'):
                return True
            # TIFF signatures
            elif decoded.startswith((b'II*\x00', b'MM\x00*')):
                return True
            # WebP signature
            elif b'WEBP' in decoded[:20]:
                return True
        except Exception:
            pass
        
        # Additional heuristic: check if it looks like base64 data
        # (only contains base64 characters and has reasonable length)
        import re
        if re.match(r'^[A-Za-z0-9+/]*={0,2}$', s) and len(s) > 1000 and len(s) % 4 == 0:
            return True
        
        return False
    
    def _is_data_url_image(self, s: str) -> bool:
        """Check if a string is a data URL with base64 encoded image."""
        if not isinstance(s, str) or len(s) < 50:
            return False
        
        # Check for data URL pattern with image MIME types
        import re
        data_url_pattern = r'^data:image/(png|jpe?g|gif|bmp|webp|svg\+xml|tiff?);base64,[A-Za-z0-9+/]*={0,2}$'
        
        if re.match(data_url_pattern, s, re.IGNORECASE):
            return True
        
        # More lenient check for partial data URLs or malformed ones
        if s.lower().startswith('data:image/') and 'base64,' in s.lower():
            # Extract the base64 part after 'base64,'
            try:
                base64_part = s.split('base64,', 1)[1]
                return self._is_base64_image(base64_part)
            except:
                pass
        
        return False
    
    def _format_image_file(self, filename: str) -> str:
        """Format a local image file as HTML image by reading and converting to base64."""
        import os
        import mimetypes
        
        try:
            # Try different possible paths for the image file
            possible_paths = [
                filename,  # Absolute path or relative from current directory
                os.path.join(os.getcwd(), filename),  # Relative to current working directory
                os.path.join(os.path.dirname(__file__), filename),  # Relative to script directory
            ]
            
            file_path = None
            for path in possible_paths:
                if os.path.isfile(path):
                    file_path = path
                    break
            
            if not file_path:
                return f"""
                <div style='margin: 8px 0; padding: 8px; background: #fff3cd; border-radius: 4px; color: #856404; border-left: 3px solid #ffc107;'>
                    ⚠️ Image file not found: {filename}
                </div>
                """
            
            # Get the MIME type for the image
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type or not mime_type.startswith('image/'):
                # Default to png if we can't determine the type
                mime_type = 'image/png'
            
            # Read the file and encode as base64
            with open(file_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                <img src='data:{mime_type};base64,{img_base64}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
                <div style='margin-top: 8px; font-size: 0.9em; color: #6c757d;'>📁 {os.path.basename(filename)}</div>
            </div>
            """
            
        except FileNotFoundError:
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px; color: #721c24; border-left: 3px solid #dc3545;'>
                ❌ Image file not found: {filename}
            </div>
            """
        except PermissionError:
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px; color: #721c24; border-left: 3px solid #dc3545;'>
                ❌ Permission denied reading file: {filename}
            </div>
            """
        except Exception as e:
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px; color: #721c24; border-left: 3px solid #dc3545;'>
                ❌ Error reading image file {filename}: {str(e)}
            </div>
            """

    def _format_matplotlib_figure(self, fig) -> str:
        """Format a matplotlib figure as HTML image."""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Try to close the figure
            try:
                if hasattr(fig, 'close'):
                    fig.close()
            except:
                pass
            
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; text-align: center;'>
                <img src='data:image/png;base64,{img_base64}' style='max-width: 800px; max-height: 600px; width: auto; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' />
            </div>
            """
        except Exception as e:
            return f"<div style='margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px; color: #721c24; border-left: 3px solid #dc3545;'>❌ Error displaying figure: {str(e)}</div>"
    
    def _format_plotly_figure(self, fig):
        """Format a Plotly figure by returning the figure object directly."""
        # Return the figure object - Gradio will handle the display
        # This method should return the figure object, not a string
        return fig
    
    def _format_plotly_figure_as_html(self, fig) -> str:
        """Convert a Plotly figure to HTML string for embedding in HTML context."""
        try:
            # Try to convert to HTML
            fig_html = fig.to_html(include_plotlyjs='cdn', div_id=f"plotly-div-{hash(str(fig)) % 10000}")
            return f"""
            <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px;'>
                <div style='margin-bottom: 8px; font-size: 0.9em; color: #6c757d; text-align: center;'>📊 Interactive Plotly Figure</div>
                {fig_html}
            </div>
            """
        except Exception as e:
            return f"<div style='margin: 8px 0; padding: 8px; background: #f8d7da; border-radius: 4px; color: #721c24; border-left: 3px solid #dc3545;'>❌ Error displaying Plotly figure: {str(e)}</div>"
    
    def _format_dict_output(self, d: dict) -> str:
        """Format a dictionary output as HTML."""
        try:
            # Pretty print the dictionary
            import json
            formatted_dict = json.dumps(d, indent=2, default=str)
            escaped_dict = formatted_dict.replace('<', '&lt;').replace('>', '&gt;')
            return f"<div style='margin: 8px 0; padding: 12px; background: #f8f9fa; border-radius: 4px; font-family: monospace; white-space: pre-wrap; border-left: 3px solid #17a2b8;'>{escaped_dict}</div>"
        except:
            return f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 Dictionary output ({len(d)} items)</div>"
    
    def _format_analysis_outputs(self, results: Dict[str, Any]):
        """Format the analysis outputs, returning mixed content (HTML strings and Plotly figures)."""
        displayed_outputs = results.get("displayed_outputs", {})
        
        if not displayed_outputs:
            return "<div style='text-align: center; padding: 20px; color: #6c757d;'><p>No analysis outputs to display.</p></div>"
        
        # Check if we have any Plotly figures in the outputs
        plotly_figures = []
        html_content = []
        
        html_content.extend([
            "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>",
            "<h3 style='color: #495057; margin-bottom: 15px; font-size: 1.4em;'>📈 Analysis Results</h3>"
        ])
        
        # Process each step's outputs
        for step_name, outputs in displayed_outputs.items():
            html_content.extend([
                f"<div style='background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;'>",
                f"<h4 style='color: #495057; margin-bottom: 10px;'>{step_name}</h4>"
            ])
            
            # Handle different types of outputs
            if isinstance(outputs, list):
                for output in outputs:
                    # Check if output is directly a Plotly figure
                    if hasattr(output, 'show') and hasattr(output, 'data') and hasattr(output, 'layout'):
                        plotly_figures.append(output)
                        html_content.append("<div style='margin: 8px 0; padding: 8px; background: #e3f2fd; border-radius: 4px; color: #1565c0; border-left: 3px solid #2196f3;'>📊 Interactive Plotly Chart (displayed below)</div>")
                    else:
                        result = self._handle_single_output(output)
                        if isinstance(result, str):
                            html_content.append(result)
                        elif hasattr(result, 'show') and hasattr(result, 'data') and hasattr(result, 'layout'):
                            plotly_figures.append(result)
                            html_content.append("<div style='margin: 8px 0; padding: 8px; background: #e3f2fd; border-radius: 4px; color: #1565c0; border-left: 3px solid #2196f3;'>📊 Interactive Plotly Chart (displayed below)</div>")
                        else:
                            html_content.append(f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 {type(result).__name__} object</div>")
            else:
                # Single output
                if hasattr(outputs, 'show') and hasattr(outputs, 'data') and hasattr(outputs, 'layout'):
                    plotly_figures.append(outputs)
                    html_content.append("<div style='margin: 8px 0; padding: 8px; background: #e3f2fd; border-radius: 4px; color: #1565c0; border-left: 3px solid #2196f3;'>📊 Interactive Plotly Chart (displayed below)</div>")
                else:
                    result = self._handle_single_output(outputs)
                    if isinstance(result, str):
                        html_content.append(result)
                    elif hasattr(result, 'show') and hasattr(result, 'data') and hasattr(result, 'layout'):
                        plotly_figures.append(result)
                        html_content.append("<div style='margin: 8px 0; padding: 8px; background: #e3f2fd; border-radius: 4px; color: #1565c0; border-left: 3px solid #2196f3;'>📊 Interactive Plotly Chart (displayed below)</div>")
                    else:
                        html_content.append(f"<div style='margin: 8px 0; padding: 8px; background: #e7f3ff; border-radius: 4px; color: #004085; border-left: 3px solid #007bff;'>📊 {type(result).__name__} object</div>")
            
            html_content.append("</div>")
        
        html_content.append("</div>")
        
        # If we have Plotly figures, return them along with HTML
        if plotly_figures:
            # Return the first figure and HTML content as a tuple
            # Gradio will display the figure in a Plot component and HTML in HTML component
            return ("".join(html_content), plotly_figures[0])
        else:
            # Only HTML content
            return "".join(html_content)
    
    def _create_question_buttons_html(self, questions: List[str]) -> str:
        """Create HTML for clickable question buttons."""
        if not questions:
            return "<p style='color: #6c757d; text-align: center; padding: 20px;'>No suggested questions available.</p>"
        
        # Create a unique ID for this set of buttons to avoid conflicts
        button_container_id = f"question_buttons_{hash(str(questions)) % 1000000}"
        
        html_parts = [
            f"<div id='{button_container_id}' style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; margin: 15px 0; width: 100%;'>"
        ]
        
        for i, question in enumerate(questions):
            # Truncate long questions for button display
            display_text = question if len(question) <= 100 else question[:97] + "..."
            
            # Escape the question text for safe JavaScript
            escaped_question = question.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
            
            button_html = f"""
            <button onclick="
                const input = document.querySelector('textarea[data-testid=\\'textbox\\']') || document.getElementById('question_input');
                if (input) {{
                    input.value = '{escaped_question}';
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    input.focus();
                }}
            "
                    style="background: #f8f9fa; border: 1px solid #dee2e6; color: #495057; 
                           padding: 12px 16px; border-radius: 8px; cursor: pointer; font-size: 0.9em;
                           transition: all 0.2s ease; text-align: left; width: 100%; min-height: 60px;
                           display: flex; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);"
                    onmouseover="this.style.background='#007bff'; this.style.color='white'; this.style.borderColor='#007bff'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 2px 6px rgba(0,123,255,0.3)';"
                    onmouseout="this.style.background='#f8f9fa'; this.style.color='#495057'; this.style.borderColor='#dee2e6'; this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)';"
                    title="{question.replace('"', '&quot;')}">
                <span style="margin-right: 8px;">💡</span> {display_text}
            </button>
            """
            html_parts.append(button_html)
        
        html_parts.append("</div>")
        
        # Add some JavaScript to ensure the buttons work properly
        html_parts.append("""
        <script>
        // Ensure the question input functionality works
        setTimeout(() => {
            const buttons = document.querySelectorAll('#""" + button_container_id + """ button');
            buttons.forEach(button => {
                button.addEventListener('click', () => {
                    // Try multiple selectors to find the question input
                    const selectors = [
                        'textarea[data-testid="textbox"]',
                        '#question_input',
                        'textarea[placeholder*="analyze"]',
                        'textarea'
                    ];
                    
                    let input = null;
                    for (const selector of selectors) {
                        input = document.querySelector(selector);
                        if (input && input.placeholder && input.placeholder.includes('analyze')) {
                            break;
                        }
                    }
                    
                    if (input) {
                        input.focus();
                    }
                });
            });
        }, 100);
        </script>
        """)
        
        return "".join(html_parts)

    def create_exports_directory(self):
        """Create exports directory if it doesn't exist."""
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        return exports_dir

    def generate_filename(self, base_name: str, extension: str) -> str:
        """Generate timestamped filename for exports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{base_name}.{extension}"

    def export_figure_to_png(self, figure, step_name: str) -> Optional[str]:
        """Export a matplotlib or plotly figure to PNG file."""
        try:
            exports_dir = self.create_exports_directory()
            filename = self.generate_filename(f"{step_name}_figure", "png")
            filepath = exports_dir / filename
            
            # Handle matplotlib figures
            if hasattr(figure, 'savefig'):
                try:
                    figure.savefig(filepath, dpi=300, bbox_inches='tight', 
                                 facecolor='white', edgecolor='none', format='png')
                    logger.info(f"Exported matplotlib figure to: {filepath}")
                    return str(filepath)
                except Exception as e:
                    logger.error(f"Error saving matplotlib figure: {str(e)}")
                    return None
            
            # Handle plotly figures
            elif hasattr(figure, 'write_image'):
                try:
                    figure.write_image(str(filepath), width=1200, height=800, scale=2, format='png')
                    logger.info(f"Exported plotly figure via write_image to: {filepath}")
                    return str(filepath)
                except Exception as e:
                    logger.warning(f"write_image failed: {str(e)}, trying alternative method")
                    # Fall through to try pio.write_image
            
            # Handle plotly figures with different API or as fallback
            if hasattr(figure, 'show') and hasattr(figure, 'data'):
                try:
                    pio.write_image(figure, str(filepath), width=1200, height=800, scale=2, format='png')
                    logger.info(f"Exported plotly figure via pio.write_image to: {filepath}")
                    return str(filepath)
                except Exception as e:
                    logger.error(f"pio.write_image failed: {str(e)}")
                    # Try one more method for plotly figures
                    try:
                        import kaleido
                        pio.write_image(figure, str(filepath), width=1200, height=800, scale=2, 
                                      format='png', engine='kaleido')
                        logger.info(f"Exported plotly figure via kaleido to: {filepath}")
                        return str(filepath)
                    except ImportError:
                        logger.warning("Kaleido not available for plotly image export")
                    except Exception as e:
                        logger.error(f"Kaleido export failed: {str(e)}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error exporting figure: {str(e)}")
            return None

    def export_dataframe_to_csv(self, df: pd.DataFrame, step_name: str) -> Optional[str]:
        """Export a pandas DataFrame to CSV file."""
        try:
            exports_dir = self.create_exports_directory()
            filename = self.generate_filename(f"{step_name}_data", "csv")
            filepath = exports_dir / filename
            
            df.to_csv(filepath, index=False)
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting DataFrame: {str(e)}")
            return None

    def export_all_outputs(self) -> Tuple[Optional[str], str]:
        """Export all analysis outputs to individual files and return a zip file."""
        if not self.analysis_results or "displayed_outputs" not in self.analysis_results:
            return None, "No analysis results to export."
        
        try:
            exports_dir = self.create_exports_directory()
            exported_files = []
            export_summary = []
            
            displayed_outputs = self.analysis_results["displayed_outputs"]
            
            for step_name, outputs in displayed_outputs.items():
                step_exports = []
                
                # Handle different types of outputs
                if isinstance(outputs, list):
                    for i, output in enumerate(outputs):
                        exported_file = self._export_single_output(output, f"{step_name}_output_{i}")
                        if exported_file:
                            exported_files.append(exported_file)
                            step_exports.append(Path(exported_file).name)
                else:
                    exported_file = self._export_single_output(outputs, step_name)
                    if exported_file:
                        exported_files.append(exported_file)
                        step_exports.append(Path(exported_file).name)
                
                if step_exports:
                    export_summary.append(f"**{step_name}:** {', '.join(step_exports)}")
            
            if not exported_files:
                return None, "No exportable outputs found."
            
            # Create zip file
            zip_filename = self.generate_filename("analysis_results", "zip")
            zip_filepath = exports_dir / zip_filename
            
            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                for file_path in exported_files:
                    zipf.write(file_path, Path(file_path).name)
            
            summary_text = f"Successfully exported {len(exported_files)} files:\n\n" + "\n".join(export_summary)
            
            return str(zip_filepath), summary_text
            
        except Exception as e:
            logger.error(f"Error exporting outputs: {str(e)}")
            return None, f"Error during export: {str(e)}"

    def _export_single_output(self, output: Any, base_name: str) -> Optional[str]:
        """Export a single output to appropriate file format."""
        try:
            # Handle pandas DataFrames
            if hasattr(output, 'to_csv'):
                return self.export_dataframe_to_csv(output, base_name)
            
            # Handle matplotlib figures
            elif hasattr(output, 'savefig'):
                return self.export_figure_to_png(output, base_name)
            
            # Handle plotly figures
            elif (hasattr(output, 'write_image') or 
                  (hasattr(output, 'show') and hasattr(output, 'data'))):
                return self.export_figure_to_png(output, base_name)
            
            # Handle dictionaries (save as JSON)
            elif isinstance(output, dict):
                exports_dir = self.create_exports_directory()
                filename = self.generate_filename(f"{base_name}_data", "json")
                filepath = exports_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(output, f, indent=2, default=str)
                return str(filepath)
            
            # Handle text outputs
            elif isinstance(output, str) and len(output) > 100:  # Only save substantial text
                exports_dir = self.create_exports_directory()
                filename = self.generate_filename(f"{base_name}_text", "txt")
                filepath = exports_dir / filename
                
                with open(filepath, 'w') as f:
                    f.write(output)
                return str(filepath)
            
            return None
            
        except Exception as e:
            logger.error(f"Error exporting single output: {str(e)}")
            return None

    def generate_pdf_report(self) -> Tuple[Optional[str], str]:
        """Generate a comprehensive PDF report of the analysis."""
        if not self.analysis_results:
            return None, "No analysis results to generate report."
        
        try:
            exports_dir = self.create_exports_directory()
            filename = self.generate_filename("analysis_report", "pdf")
            filepath = exports_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(filepath), pagesize=A4, 
                                  topMargin=1*inch, bottomMargin=1*inch,
                                  leftMargin=1*inch, rightMargin=1*inch)
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#2c5aa0'),
                alignment=TA_CENTER,
                spaceAfter=20
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2c5aa0'),
                spaceBefore=20,
                spaceAfter=10
            )
            
            # Build content
            content = []
            
            # Title
            content.append(Paragraph("Clinical Data Analysis Report", title_style))
            content.append(Spacer(1, 20))
            
            # Report metadata
            report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            content.append(Paragraph(f"<b>Generated:</b> {report_date}", styles['Normal']))
            
            if self.current_question:
                content.append(Paragraph(f"<b>Analysis Question:</b> {self.current_question}", styles['Normal']))
            
            content.append(Spacer(1, 20))
            
            # Data Summary
            if self.current_data is not None:
                content.append(Paragraph("Data Summary", heading_style))
                data_info = [
                    f"Dataset Shape: {self.current_data.shape[0]:,} rows × {self.current_data.shape[1]} columns",
                    f"Missing Data: {(self.current_data.isnull().sum().sum() / (self.current_data.shape[0] * self.current_data.shape[1]) * 100):.2f}%"
                ]
                for info in data_info:
                    content.append(Paragraph(info, styles['Normal']))
                content.append(Spacer(1, 15))
            
            # Analysis Plan
            plan_info = self.analysis_results.get("analysis_plan", {})
            if plan_info:
                content.append(Paragraph("Analysis Plan", heading_style))
                
                plan_details = plan_info.get("plan_details", [])
                if plan_details:
                    for i, step in enumerate(plan_details, 1):
                        step_title = step.get("title", f"Step {i}")
                        step_desc = step.get("description", "No description available")
                        content.append(Paragraph(f"<b>Step {i}: {step_title}</b>", styles['Normal']))
                        content.append(Paragraph(step_desc, styles['Normal']))
                        content.append(Spacer(1, 8))
                
                # Execution summary
                content.append(Paragraph(f"<b>Execution Summary:</b> "
                                       f"{plan_info.get('successful_steps', 0)} successful, "
                                       f"{plan_info.get('failed_steps', 0)} failed out of "
                                       f"{plan_info.get('total_steps', 0)} total steps", 
                                       styles['Normal']))
                content.append(Spacer(1, 20))
            
            # Analysis Results
            displayed_outputs = self.analysis_results.get("displayed_outputs", {})
            if displayed_outputs:
                content.append(Paragraph("Analysis Results", heading_style))
                
                for step_name, outputs in displayed_outputs.items():
                    content.append(Paragraph(f"<b>{step_name}</b>", styles['Heading3']))
                    
                    # Handle outputs - could be a single output or list of outputs
                    outputs_to_process = []
                    if isinstance(outputs, list):
                        outputs_to_process = outputs
                    else:
                        outputs_to_process = [outputs]
                    
                    for idx, output in enumerate(outputs_to_process):
                        if len(outputs_to_process) > 1:
                            content.append(Paragraph(f"<i>Output {idx + 1}:</i>", styles['Normal']))
                        
                        # Handle different output types
                        if hasattr(output, 'to_html'):  # DataFrame
                            self._add_dataframe_to_pdf(content, output, styles)
                        
                        elif isinstance(output, str):
                            # Text output
                            text_content = output[:1000] + ("..." if len(output) > 1000 else "")
                            content.append(Paragraph(text_content, styles['Normal']))
                        
                        elif isinstance(output, dict):
                            # Dictionary output
                            for key, value in output.items():
                                content.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles['Normal']))
                        
                        elif hasattr(output, 'savefig') or (hasattr(output, 'show') and hasattr(output, 'data')):
                            # This is a figure - export it and add to PDF
                            self._add_figure_to_pdf(content, output, step_name, idx, styles)
                        
                        elif isinstance(output, (int, float)):
                            # Numeric output
                            content.append(Paragraph(f"Result: {output}", styles['Normal']))
                        
                        else:
                            # Try to convert other types to string
                            try:
                                str_output = str(output)
                                if len(str_output) > 500:
                                    str_output = str_output[:500] + "..."
                                content.append(Paragraph(f"Output: {str_output}", styles['Normal']))
                            except Exception as e:
                                content.append(Paragraph(f"<i>Unable to display output: {str(e)}</i>", styles['Normal']))
                    
                    content.append(Spacer(1, 15))
            
            # Footer
            content.append(PageBreak())
            content.append(Paragraph("Report generated by Clinical Data Analysis Platform", 
                                   styles['Normal']))
            content.append(Paragraph(f"Timestamp: {datetime.now().isoformat()}", 
                                   styles['Normal']))
            
            # Build PDF
            doc.build(content)
            
            return str(filepath), f"PDF report generated successfully: {filename}"
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None, f"Error generating PDF report: {str(e)}"
    
    
    def _add_dataframe_to_pdf(self, content: List, df: pd.DataFrame, styles):
        """Add a DataFrame to PDF content as a table."""
        try:
            df_sample = df.head(10) if len(df) > 10 else df
            
            # Convert DataFrame to table data
            table_data = []
            
            # Add headers
            table_data.append(df_sample.columns.tolist())
            
            # Add data rows, converting all values to strings
            for _, row in df_sample.iterrows():
                table_data.append([str(cell) for cell in row.values])
            
            # Create table with styling
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8f9fa')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            content.append(table)
            
            if len(df) > 10:
                content.append(Paragraph(f"<i>Showing first 10 rows of {len(df):,} total rows</i>", 
                                       styles['Normal']))
        except Exception as e:
            content.append(Paragraph(f"<i>DataFrame display error: {str(e)}</i>", styles['Normal']))
    
    def _add_figure_to_pdf(self, content: List, figure, step_name: str, idx: int, styles):
        """Add a figure to PDF content by first exporting it as PNG."""
        try:
            # Check if we have the required dependencies
            if not self._check_and_warn_image_export():
                content.append(Paragraph("<i>Chart/Figure (image export dependencies not available)</i>", styles['Normal']))
                return
            
            # Generate unique name for this figure
            figure_name = f"{step_name}_figure_{idx}" if idx > 0 else f"{step_name}_figure"
            
            # Export the figure to PNG
            figure_path = None
            
            # Handle matplotlib figures
            if hasattr(figure, 'savefig'):
                try:
                    exports_dir = self.create_exports_directory()
                    filename = self.generate_filename(figure_name, "png")
                    figure_path = exports_dir / filename
                    
                    figure.savefig(figure_path, dpi=300, bbox_inches='tight', 
                                 facecolor='white', edgecolor='none', format='png')
                    logger.info(f"Exported matplotlib figure to: {figure_path}")
                except Exception as e:
                    logger.error(f"Error exporting matplotlib figure: {str(e)}")
                    
            # Handle plotly figures
            elif hasattr(figure, 'write_image') or (hasattr(figure, 'show') and hasattr(figure, 'data')):
                try:
                    exports_dir = self.create_exports_directory()
                    filename = self.generate_filename(figure_name, "png")
                    figure_path = exports_dir / filename
                    
                    # Try different plotly export methods
                    if hasattr(figure, 'write_image'):
                        figure.write_image(str(figure_path), width=1200, height=800, scale=2, format='png')
                    else:
                        # Use plotly.io for figures that don't have write_image method
                        pio.write_image(figure, str(figure_path), width=1200, height=800, scale=2, format='png')
                    
                    logger.info(f"Exported plotly figure to: {figure_path}")
                except Exception as e:
                    logger.error(f"Error exporting plotly figure: {str(e)}")
                    # Try alternative method for plotly
                    try:
                        import plotly.offline as pyo
                        html_content = pyo.plot(figure, output_type='div', include_plotlyjs=False)
                        content.append(Paragraph("<i>Interactive plot exported (view in analysis results)</i>", styles['Normal']))
                        return
                    except:
                        pass
            
            # If we successfully exported the figure, add it to PDF
            if figure_path and os.path.exists(figure_path):
                try:
                    # Calculate image dimensions to fit on page
                    max_width = 6 * inch
                    max_height = 4 * inch
                    
                    # Add the image to PDF
                    img = Image(str(figure_path), width=max_width, height=max_height)
                    content.append(Spacer(1, 10))
                    content.append(img)
                    content.append(Spacer(1, 10))
                    
                    logger.info(f"Successfully added figure to PDF: {figure_path}")
                    
                except Exception as e:
                    logger.error(f"Error adding image to PDF: {str(e)}")
                    content.append(Paragraph(f"<i>Chart/Figure could not be displayed in PDF: {str(e)}</i>", styles['Normal']))
            else:
                content.append(Paragraph("<i>Chart/Figure (unable to export for PDF display)</i>", styles['Normal']))
                
        except Exception as e:
            logger.error(f"Error processing figure for PDF: {str(e)}")
            content.append(Paragraph(f"<i>Chart/Figure processing error: {str(e)}</i>", styles['Normal']))

    def create_interface(self):
        """Create the Gradio interface."""
        
        # Clean, professional CSS
        css = """
        .main-container {
            max-width: 1000px;
            margin: 0 auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .compact-header {
            background: linear-gradient(135deg, #deecf9 0%, #c7e0f4 100%);
            color: white;
            padding: 30px 20px;
            border-radius: 12px;
            margin-bottom: 25px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        .compact-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            pointer-events: none;
        }
        .header-title {
            font-size: 2.2em;
            font-weight: 700;
            margin: 0 0 8px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -0.5px;
        }
        .header-subtitle {
            font-size: 1.1em;
            margin: 0;
            opacity: 0.95;
            font-weight: 300;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .header-icon {
            font-size: 1.3em;
            margin-right: 10px;
            vertical-align: middle;
        }
        .section-title {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 12px 20px;
            border-radius: 8px;
            margin: 20px 0 15px 0;
            border-left: 4px solid #667eea;
            font-weight: 600;
            color: #495057;
            font-size: 1.1em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .plan-confirmation {
            background: #f8f9fa;
            border: 2px solid #007bff;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,123,255,0.15);
        }
        .plan-step {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 12px 0;
            border-left: 4px solid #6c757d;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        .plan-step:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .plan-step-visualization {
            border-left-color: #28a745;
        }
        .plan-step-table {
            border-left-color: #17a2b8;
        }
        .plan-controls {
            background: #fff3cd;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            border-left: 4px solid #ffc107;
        }
        .config-row {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .status-box {
            padding: 8px 12px;
            border-radius: 4px;
            margin: 5px 0;
            font-size: 0.9em;
        }
        .suggestion-btn {
            margin: 2px !important;
            font-size: 0.85em !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
            background: #e9ecef !important;
            border: 1px solid #dee2e6 !important;
            color: #495057 !important;
            transition: all 0.2s ease !important;
        }
        .suggestion-btn:hover {
            background: #007bff !important;
            color: white !important;
            border-color: #007bff !important;
        }
        /* Custom table styling for intermediate outputs */
        #intermediate-output-table {
            width: 100% !important;
            margin: 0 !important;
            font-size: 0.85em !important;
        }
        #intermediate-output-table th {
            background: #f8f9fa !important;
            color: #495057 !important;
            font-weight: 600 !important;
            padding: 8px 12px !important;
            border-bottom: 2px solid #dee2e6 !important;
            text-align: left !important;
        }
        #intermediate-output-table td {
            padding: 6px 12px !important;
            border-bottom: 1px solid #dee2e6 !important;
            vertical-align: top !important;
        }
        #intermediate-output-table tr:nth-child(even) {
            background: #f8f9fa !important;
        }
        #intermediate-output-table tr:hover {
            background: #e3f2fd !important;
        }
        """
        
        with gr.Blocks(css=css, title="Clinical Data Analysis", theme=gr.themes.Soft()) as app:
            
            # Compact header
            gr.HTML("""
            <div class="compact-header">
                <h1 class="header-title">
                    <span class="header-icon">🏥</span>Clinical Data Analysis Platform
                </h1>
            </div>
            """)
            
            # Data upload section
            gr.HTML("<div class='section-title'>📁 Data Upload</div>")
            
            with gr.Row():
                with gr.Column(scale=4):
                    file_upload = gr.File(
                        label="Upload Clinical Data",
                        file_types=[".csv", ".xlsx", ".xls"],
                        file_count="single"
                    )
                    data_preview = gr.Dataframe(
                        label="Data Preview",
                        value=None,
                        interactive=False,
                        visible=False,
                        wrap=True,
                        max_height=300
                    )
                with gr.Column(scale=1):
                    data_summary = gr.Markdown(value="Upload a file to see data summary")
            
            # Analysis section
            gr.HTML("<div class='section-title'>🔬 Analysis</div>")
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to analyze? Upload data first to see suggested questions below.",
                    lines=3,
                    scale=3,
                    elem_id="question_input"
                )
                with gr.Column(scale=1):
                    generate_plan_btn = gr.Button("📋 Generate Plan", variant="primary", size="lg")
            
            # Plan confirmation section (hidden initially)
            with gr.Row(visible=False) as plan_confirmation_row:
                with gr.Column():
                    plan_display = gr.HTML(value="")
            
            # Plan editing section (hidden initially)
            with gr.Row(visible=False) as plan_editing_row:
                with gr.Column(scale=3):
                    plan_feedback = gr.Textbox(
                        label="Plan Modifications",
                        placeholder="Enter feedback to modify the plan (e.g., 'Add a correlation matrix', 'Remove step 2', 'Change to scatter plot')",
                        lines=2
                    )
                with gr.Column(scale=1):
                    update_plan_btn = gr.Button("� Update Plan", variant="secondary", size="lg")
            
            # Plan execution section (hidden initially)
            with gr.Row(visible=False) as plan_execution_row:
                with gr.Column():
                    confirm_plan_btn = gr.Button("✅ Execute Plan", variant="primary", size="lg")
            
            # Suggested questions section - positioned below the input and button
            # Will be hidden initially and shown after data upload
            with gr.Row():
                with gr.Column():
                    suggested_questions_title = gr.HTML(
                        value="",
                        visible=False
                    )
                    suggested_questions_html = gr.HTML(
                        value="",
                        elem_id="suggested_questions",
                        visible=False
                    )
            
            # Results section - Compact tabs
            gr.HTML("<div class='section-title'>📊 Results</div>")
            
            with gr.Tabs():
                with gr.TabItem("📋 Analysis Plan"):
                    analysis_plan = gr.HTML()
                
                with gr.TabItem("📊 Results"):
                    with gr.Column():
                        analysis_outputs = gr.HTML()
                        analysis_plots = gr.Plot(visible=False)
                        
                        # Export section
                        with gr.Row(visible=False) as export_row:
                            with gr.Column():
                                gr.HTML("<div style='margin-top: 20px; margin-bottom: 10px;'><h4 style='color: #495057; margin: 0;'>📁 Export Options</h4></div>")
                                
                                with gr.Row():
                                    export_all_btn = gr.Button("📦 Export All Files (ZIP)", variant="secondary", size="sm")
                                    generate_pdf_btn = gr.Button("📄 Generate PDF Report", variant="primary", size="sm")
                                
                                # Export status and download
                                export_status = gr.HTML(value="", visible=False)
                                export_download = gr.File(label="Download Exported Files", visible=False)
            
            # Event handlers
            file_upload.upload(
                fn=self.process_uploaded_file,
                inputs=[file_upload],
                outputs=[data_summary, suggested_questions_title, suggested_questions_html, suggested_questions_title, suggested_questions_html, data_preview]
            )
            
            # Plan generation
            generate_plan_btn.click(
                fn=self.generate_analysis_plan,
                inputs=[question_input],
                outputs=[plan_display, plan_confirmation_row, plan_editing_row, plan_execution_row]
            )
            
            # Plan update
            update_plan_btn.click(
                fn=self.update_plan_with_feedback,
                inputs=[plan_feedback],
                outputs=[plan_display, plan_confirmation_row, plan_editing_row, plan_execution_row, plan_feedback]
            )
            
            # Plan execution
            confirm_plan_btn.click(
                fn=self.confirm_and_execute_plan,
                inputs=[],
                outputs=[analysis_plan, analysis_outputs, analysis_plots, analysis_plots, plan_confirmation_row, plan_editing_row, export_row]
            )
            
            # Export handlers
            export_all_btn.click(
                fn=self.export_all_outputs,
                inputs=[],
                outputs=[export_download, export_status]
            ).then(
                fn=lambda file, status: (gr.update(visible=True) if file else gr.update(visible=False), 
                                       gr.update(value=status, visible=True)),
                inputs=[export_download, export_status],
                outputs=[export_download, export_status]
            )
            
            generate_pdf_btn.click(
                fn=self.generate_pdf_report,
                inputs=[],
                outputs=[export_download, export_status]
            ).then(
                fn=lambda file, status: (gr.update(visible=True) if file else gr.update(visible=False), 
                                       gr.update(value=status, visible=True)),
                inputs=[export_download, export_status],
                outputs=[export_download, export_status]
            )
            
        
        return app

    def _format_trajectory(self, trajectory_str: str) -> str:
        """Format the execution trajectory with code highlighting and structure."""
        if not trajectory_str:
            return "<p style='color: #6c757d; font-style: italic;'>No trajectory information available.</p>"
        
        # Parse trajectory string - it's usually a dict representation
        try:
            # Try to evaluate the trajectory string as a Python dict
            
            # Clean up the trajectory string if it has extra quotes or formatting
            cleaned_trajectory = trajectory_str.strip()
            if cleaned_trajectory.startswith("{'") or cleaned_trajectory.startswith('{"'):
                # Try to parse as dict
                try:
                    trajectory_dict = ast.literal_eval(cleaned_trajectory)
                except:
                    # If literal_eval fails, try a more robust parsing approach
                    trajectory_dict = self._parse_trajectory_string(cleaned_trajectory)
            else:
                trajectory_dict = self._parse_trajectory_string(cleaned_trajectory)
            
            return self._format_trajectory_dict(trajectory_dict)
            
        except Exception as e:
            # Fallback to simple text formatting
            return self._format_trajectory_as_text(trajectory_str)
    
    def _parse_trajectory_string(self, trajectory_str: str) -> dict:
        """Parse trajectory string using regex patterns."""
        
        trajectory_dict = {}
        
        # Pattern for thought_X: "content"
        thought_pattern = r"'thought_(\d+)':\s*[\"'](.*?)[\"'](?=,\s*'|\s*})"
        thoughts = re.findall(thought_pattern, trajectory_str, re.DOTALL)
        
        for i, thought in thoughts:
            trajectory_dict[f'thought_{i}'] = thought.strip()
        
        # Pattern for tool_name_X: "content"
        tool_name_pattern = r"'tool_name_(\d+)':\s*[\"'](.*?)[\"']"
        tool_names = re.findall(tool_name_pattern, trajectory_str)
        
        for i, tool_name in tool_names:
            trajectory_dict[f'tool_name_{i}'] = tool_name.strip()
        
        # Pattern for tool_args_X: {...}
        tool_args_pattern = r"'tool_args_(\d+)':\s*({.*?})(?=,\s*'|\s*})"
        tool_args = re.findall(tool_args_pattern, trajectory_str, re.DOTALL)
        
        for i, args_str in tool_args:
            try:
                args_dict = ast.literal_eval(args_str)
                trajectory_dict[f'tool_args_{i}'] = args_dict
            except:
                trajectory_dict[f'tool_args_{i}'] = args_str
        
        # Pattern for observation_X: "content"
        observation_pattern = r"'observation_(\d+)':\s*[\"'](.*?)[\"'](?=,\s*'|\s*})"
        observations = re.findall(observation_pattern, trajectory_str, re.DOTALL)
        
        for i, observation in observations:
            trajectory_dict[f'observation_{i}'] = observation.strip()
        
        return trajectory_dict
    
    def _format_trajectory_dict(self, trajectory_dict: dict) -> str:
        """Format a parsed trajectory dictionary into HTML."""
        if not trajectory_dict:
            return "<p style='color: #6c757d; font-style: italic;'>No trajectory information available.</p>"
        
        html_parts = []
        
        # Group by iteration number
        iterations = {}
        for key, value in trajectory_dict.items():
            if '_' in key:
                base_key, iteration = key.rsplit('_', 1)
                if iteration.isdigit():
                    iteration_num = int(iteration)
                    if iteration_num not in iterations:
                        iterations[iteration_num] = {}
                    iterations[iteration_num][base_key] = value
        
        # Add section header for detailed iterations
        if iterations:
            html_parts.extend([
                "<div style='margin-top: 15px; margin-bottom: 10px;'>",
                "<h6 style='color: #495057; margin: 0; font-weight: bold; font-size: 0.95em;'>🔄 Detailed Execution Steps</h6>",
                "</div>"
            ])
        
        # Format each iteration
        for iteration_num in sorted(iterations.keys()):
            iteration_data = iterations[iteration_num]
            
            html_parts.extend([
                f"<div style='margin-bottom: 15px; padding: 15px; background: white; border-radius: 6px; border-left: 3px solid #17a2b8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>",
                f"<h6 style='color: #17a2b8; margin: 0 0 12px 0; font-weight: bold; font-size: 1.0em;'>Iteration {iteration_num + 1}</h6>"
            ])
            
            # Display thought
            if 'thought' in iteration_data:
                html_parts.extend([
                    "<div style='margin-bottom: 12px;'>",
                    "<strong style='color: #495057;'>💭 Reasoning:</strong>",
                    # f"<div style='margin: 8px 0; padding: 10px; background: #e3f2fd; border-radius: 4px; font-style: italic; color: #1565c0; line-height: 1.4;'>{iteration_data['thought']}</div>",
                    f"<div style='margin: 8px 0; padding: 10px; line-height: 1.4;'>{iteration_data['thought']}</div>",
                    "</div>"
                ])
            
            # Display tool usage with side-by-side code and result
            if 'tool_name' in iteration_data:
                tool_name = iteration_data['tool_name']
                tool_args = iteration_data.get('tool_args', {})
                observation = iteration_data.get('observation', '')
                
                html_parts.extend([
                    "<div style='margin-bottom: 12px;'>",
                    f"<strong style='color: #495057; display: block; margin-bottom: 8px;'>🛠️ Tool Used: <span style='color: #28a745; font-weight: bold;'>{tool_name}</span></strong>"
                ])
                
                # Extract code content
                code_content = self._extract_code_content(tool_name, tool_args)
                
                if code_content:
                    # Side-by-side layout for code and result
                    html_parts.extend([
                        "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px;'>",
                        
                        # Left column: Code
                        "<div style='background: #f8f9fa; border-radius: 4px; border-left: 3px solid #6c757d;'>",
                        "<div style='padding: 8px 12px; background: #e9ecef; border-radius: 4px 4px 0 0; font-weight: 600; color: #495057; font-size: 0.9em;'>📝 Code Executed</div>",
                        f"<pre style='background: #f8f9fa; padding: 12px; margin: 0; overflow-x: auto; font-size: 0.8em; color: #495057; white-space: pre-wrap; line-height: 1.3;'><code>{self._escape_html(code_content)}</code></pre>",
                        "</div>",
                        
                        # Right column: Result
                        "<div style='background: #f8f9fa; border-radius: 4px; border-left: 3px solid #28a745;'>",
                        "<div style='padding: 8px 12px; background: #d4edda; border-radius: 4px 4px 0 0; font-weight: 600; color: #155724; font-size: 0.9em;'>📊 Observation</div>",
                        f"<div style='padding: 12px; margin: 0; font-size: 0.9em; color: #155724; line-height: 1.4;'>{observation if observation else 'No result captured'}</div>",
                        "</div>",
                        
                        "</div>"  # End grid
                    ])
                else:
                    # If no code, just show the result
                    if observation:
                        html_parts.extend([
                            "<div style='margin-top: 8px;'>",
                            "<strong style='color: #495057;'>📊 Result:</strong>",
                            f"<div style='margin: 5px 0; padding: 10px; background: #d4edda; border-radius: 4px; color: #155724; line-height: 1.4;'>{observation}</div>",
                            "</div>"
                        ])
                
                html_parts.append("</div>")
            
            # Display standalone observation/result if no tool was used
            elif 'observation' in iteration_data:
                observation = iteration_data['observation']
                html_parts.extend([
                    "<div style='margin-bottom: 8px;'>",
                    "<strong style='color: #495057;'>📊 Result:</strong>",
                    f"<div style='margin: 5px 0; padding: 10px; background: #d4edda; border-radius: 4px; color: #155724; line-height: 1.4;'>{observation}</div>",
                    "</div>"
                ])
            
            html_parts.append("</div>")
        
        return "".join(html_parts)
    
    def _extract_code_content(self, tool_name: str, tool_args: dict) -> str:
        """Extract code content from tool arguments."""
        if not isinstance(tool_args, dict):
            return ""
        
        # Different tools store code in different parameters
        if tool_name == "DataAnalysisTool":
            return tool_args.get('operation', '')
        elif tool_name == "ClinicalVisualizationTool":
            return tool_args.get('plot_code', '')
        elif 'code' in tool_args:
            return tool_args.get('code', '')
        elif 'operation' in tool_args:
            return tool_args.get('operation', '')
        elif 'plot_code' in tool_args:
            return tool_args.get('plot_code', '')
        
        return ""
    
    def _format_trajectory_as_text(self, trajectory_str: str) -> str:
        """Fallback method to format trajectory as simple text."""
        # Clean up and truncate if too long
        cleaned_text = trajectory_str.strip()
        if len(cleaned_text) > 1000:
            cleaned_text = cleaned_text[:1000] + "..."
        
        return f"""
        <div style='margin: 8px 0; padding: 8px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #6c757d;'>
            <strong style='color: #495057;'>🔍 Execution Trajectory:</strong>
            <pre style='margin: 5px 0; white-space: pre-wrap; font-size: 0.8em; color: #6c757d;'>{self._escape_html(cleaned_text)}</pre>
        </div>
        """
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text."""
        return html.escape(text)
    
    def _format_intermediate_output(self, output: Any) -> str:
        """Format intermediate output with special handling for DataFrames."""
        if output is None:
            return "<p style='color: #6c757d; font-style: italic;'>No output generated.</p>"
        
        # Handle pandas DataFrames with special formatting (show top 5 rows)
        if hasattr(output, 'to_html') and hasattr(output, 'shape'):
            try:
                # Get top 5 rows for display
                display_df = output.head(5) if len(output) > 5 else output
                
                # Create a nice summary header
                summary_info = f"DataFrame: {output.shape[0]} rows × {output.shape[1]} columns"
                if len(output) > 5:
                    summary_info += " (showing first 5 rows)"
                
                # Generate HTML table with better styling
                html_table = display_df.to_html(
                    classes='table table-striped table-hover table-sm',
                    table_id='intermediate-output-table',
                    escape=False,
                    border=0
                )
                
                # Style the table
                styled_table = f"""
                <div style='margin-bottom: 10px;'>
                    <div style='background: #e8f5e8; padding: 8px 12px; border-radius: 4px 4px 0 0; font-weight: bold; color: #155724; border-left: 3px solid #28a745;'>
                        📊 {summary_info}
                    </div>
                    <div style='background: white; padding: 0; border-radius: 0 0 4px 4px; overflow-x: auto; border:  1px solid #dee2e6; border-top: none;'>
                        {html_table}
                    </div>
                </div>
                """
                
                return styled_table
                
            except Exception as e:
                return f"<div style='color: #dc3545; background: #f8d7da; padding: 8px; border-radius: 4px;'>❌ Error displaying DataFrame: {str(e)}</div>"
        
        # Handle matplotlib figures
        elif hasattr(output, 'savefig'):
            return self._format_matplotlib_figure(output)
        
        # Handle Plotly figures
        elif hasattr(output, 'show') and hasattr(output, 'data') and hasattr(output, 'layout'):
            return self._format_plotly_figure_as_html(output)
        
        # Handle lists of matplotlib figures (plot results)
        elif isinstance(output, list) and len(output) > 0 and hasattr(output[0], 'savefig'):
            html_parts = []
            for i, fig in enumerate(output):
                if hasattr(fig, 'savefig'):
                    html_parts.append(f"<h6 style='color: #495057; margin: 10px 0 5px 0;'>Plot {i+1}:</h6>")
                    html_parts.append(self._format_matplotlib_figure(fig))
            return "".join(html_parts)
        
        # Handle dictionaries
        elif isinstance(output, dict):
            try:
                formatted_dict = json.dumps(output, indent=2, default=str)
                escaped_dict = self._escape_html(formatted_dict)
                return f"""
                <div style='background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 3px solid #17a2b8;'>
                    <div style='background: #e7f3ff; padding: 8px 12px; margin: -12px -12px 12px -12px; border-radius: 4px 4px 0 0; font-weight: bold; color: #004085;'>
                        📋 Dictionary Output ({len(output)} items)
                    </div>
                    <pre style='margin: 0; white-space: pre-wrap; font-size: 0.85em; color: #495057; font-family: "Courier New", monospace;'>{escaped_dict}</pre>
                </div>
                """
            except Exception as e:
                return f"<div style='color: #dc3545; background: #f8d7da; padding: 8px; border-radius: 4px;'>❌ Error displaying dictionary: {str(e)}</div>"
        
        # Handle strings
        elif isinstance(output, str):
            # Check if it's a very long string and truncate
            display_text = output if len(output) <= 500 else output[:500] + "... (truncated)"
            escaped_text = self._escape_html(display_text)
            return f"""
            <div style='background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 3px solid #6c757d;'>
                <div style='background: #e9ecef; padding: 8px 12px; margin: -12px -12px 12px -12px; border-radius: 4px 4px 0 0; font-weight: 600; color: #495057;'>
                    📝 Text Output
                </div>
                <div style='white-space: pre-wrap; font-size: 0.9em; color: #495057; line-height: 1.4;'>{escaped_text}</div>
            </div>
            """
        
        # Handle numeric values
        elif isinstance(output, (int, float)):
            return f"""
            <div style='background: #e8f5e8; padding: 12px; border-radius: 4px; border-left: 3px solid #28a745; text-align: center;'>
                <div style='font-weight: bold; color: #155724; font-size: 1.1em;'>📊 Numeric Result</div>
                <div style='font-size: 1.3em; color: #155724; margin-top: 5px; font-weight: bold;'>{output}</div>
            </div>
            """
        
        # Handle lists and tuples
        elif isinstance(output, (list, tuple)):
            if len(output) == 0:
                return "<div style='color: #6c757d; font-style: italic;'>Empty list/tuple</div>"
            
            # If it's a small list, display all items
            if len(output) <= 10:
                items_html = []
                for i, item in enumerate(output):
                    item_html = self._format_intermediate_output(item)
                    items_html.append(f"<div style='margin-bottom: 8px;'><strong>Item {i+1}:</strong> {item_html}</div>")
                
                return f"""
                <div style='background: #fff3cd; padding: 12px; border-radius: 4px; border-left: 3px solid #ffc107;'>
                    <div style='background: #ffeaa7; padding: 8px 12px; margin: -12px -12px 12px -12px; border-radius: 4px 4px 0 0; font-weight: bold; color: #856404;'>
                        📋 {type(output).__name__.title()} ({len(output)} items)
                    </div>
                    {"".join(items_html)}
                </div>
                """
            else:
                # For large lists, just show summary
                return f"""
                <div style='background: #fff3cd; padding: 12px; border-radius: 4px; border-left: 3px solid #ffc107; text-align: center;'>
                    <div style='font-weight: bold; color: #856404;'>📋 {type(output).__name__.title()}</div>
                    <div style='color: #856404; margin-top: 5px;'>{len(output)} items (too many to display)</div>
                </div>
                """
        
        # Handle other types
        else:
            try:
                str_repr = str(output)
                if len(str_repr) > 200:
                    str_repr = str_repr[:200] + "..."
                escaped_repr = self._escape_html(str_repr)
                
                return f"""
                <div style='background: #e7f3ff; padding: 12px; border-radius: 4px; border-left: 3px solid #007bff;'>
                    <div style='background: #cce5ff; padding: 8px 12px; margin: -12px -12px 12px -12px; border-radius: 4px 4px 0 0; font-weight: bold; color: #004085;'>
                        🔧 {type(output).__name__} Object
                    </div>
                    <div style='font-size: 0.9em; color: #495057; font-family: monospace;'>{escaped_repr}</div>
                </div>
                """
            except Exception as e:
                return f"<div style='color: #dc3545; background: #f8d7da; padding: 8px; border-radius: 4px;'>❌ Error displaying output: {str(e)}</div>"

    def generate_analysis_plan(self, question: str, progress=gr.Progress()):
        """Generate analysis plan without executing it."""
        if not question.strip():
            return "❌ Please enter a question", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False)
        
        if self.current_data is None:
            return "❌ Please upload data first", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False)
        
        # Initialize analyzer if not already done
        if self.analyzer is None:
            try:
                self.analyzer = ClinicalDataAnalyzer()
            except Exception as e:
                return f"❌ Failed to initialize analyzer: {str(e)}", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False)
        
        # try:
        progress(0.1, desc="Generating analysis plan...")
        
        # Save current data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            self.current_data.to_csv(tmp_file.name, index=False)
            temp_filepath = tmp_file.name
        
        try:
            # Load and process data
            df, metadata = self.analyzer.data_processor.load_data(temp_filepath)
            self.data_package = self.analyzer.data_processor.prepare_data_for_analysis(df, metadata)
            
            # Generate analysis plan
            plan_result = self.analyzer.planner(question, self.data_package)
            
            # Store plan and question for later use
            self.current_plan = plan_result
            self.current_question = question
            self.plan_confirmed = False
            
            progress(1.0, desc="Plan generated!")
            
            # Format plan for display
            plan_html = self._format_editable_plan(plan_result)
            
            return plan_html, gr.Row(visible=True), gr.Row(visible=True), gr.Button(visible=True)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filepath)
            except:
                pass
                    
        # except Exception as e:
        #     logger.error(f"Plan generation error: {str(e)}")
        #     return f"❌ Plan generation error: {str(e)}", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False)
    
    def update_plan_with_feedback(self, feedback: str, progress=gr.Progress()):
        """Update the analysis plan based on user feedback."""
        if not self.current_plan or not self.current_question:
            return "❌ No plan to update", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False), ""
        
        if not feedback.strip():
            return "❌ Please provide feedback for plan modification", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False), ""
        
        try:
            progress(0.1, desc="Updating plan with feedback...")
            
            # Create enhanced query with feedback
            enhanced_query = f"""
            Original Query: {self.current_question}
            
            User Feedback for Plan Modification: {feedback}
            
            Please modify the analysis plan based on the feedback provided. The user wants to:
            - Add, remove, or modify analysis steps
            - Change the order of operations
            - Adjust output formats or visualizations
            - Update step dependencies
            
            Generate an updated analysis plan that incorporates this feedback.
            """
            
            # Generate updated plan
            updated_plan_result = self.analyzer.planner(enhanced_query, self.data_package)
            
            # Update stored plan
            self.current_plan = updated_plan_result
            self.plan_confirmed = False
            
            progress(1.0, desc="Plan updated!")
            
            # Format updated plan for display
            plan_html = self._format_editable_plan(updated_plan_result)
            
            # Clear the feedback text box by returning empty string
            return plan_html, gr.Row(visible=True), gr.Row(visible=True), gr.Button(visible=True), ""
            
        except Exception as e:
            logger.error(f"Plan update error: {str(e)}")
            return f"❌ Plan update error: {str(e)}", gr.Row(visible=False), gr.Row(visible=False), gr.Button(visible=False), ""
    
    def confirm_and_execute_plan(self, progress=gr.Progress()):
        """Confirm the plan and execute the analysis."""
        if not self.current_plan or not self.current_question:
            yield "❌ No plan to execute", "", None, gr.Plot(visible=False), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=False)
            return
        
        try:
            # Show loading state in the plan display first
            loading_plan = f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                        border: 2px solid #1976d2; 
                        border-radius: 12px; 
                        padding: 20px; 
                        margin: 15px 0;">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="color: #1976d2; font-size: 24px; margin-right: 15px;">⏳</div>
                    <h3 style="color: #1976d2; margin: 0; font-size: 18px;">Executing Analysis Plan...</h3>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="color: #555; font-size: 14px;">
                        <strong>Question:</strong> {self.current_question}
                    </div>
                </div>
                <div style="background: rgba(25, 118, 210, 0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <div style="color: #1976d2; font-size: 16px; font-weight: 500;">
                        🔄 Processing your analysis request...
                    </div>
                    <div style="color: #666; font-size: 14px; margin-top: 8px;">
                        This may take a few moments while we execute your plan.
                    </div>
                </div>
            </div>
            """
            
            # Yield the loading state first
            yield loading_plan, "", None, gr.Plot(visible=False), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=False)
            
            progress(0.1, desc="Executing confirmed plan...")
            
            # Mark plan as confirmed
            self.plan_confirmed = True
            
            # Save current data to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                self.current_data.to_csv(tmp_file.name, index=False)
                temp_filepath = tmp_file.name
            
            try:
                # Execute the confirmed plan
                # We'll modify the analyzer to accept a pre-generated plan
                results = self._execute_confirmed_plan(temp_filepath, self.current_question, self.current_plan)
                self.analysis_results = results
                
                progress(0.8, desc="Formatting results...")
                
                if results.get("success"):
                    # Format final results
                    final_plan_html = self._format_analysis_plan(results)
                    outputs_result = self._format_analysis_outputs(results)
                    
                    # Handle different return types from _format_analysis_outputs
                    if isinstance(outputs_result, tuple):
                        # We have both HTML and Plotly figure
                        outputs_html, plotly_fig = outputs_result
                        progress(1.0, desc="Analysis complete!")
                        # Hide the plan confirmation and editing sections after execution, show export options
                        yield final_plan_html, outputs_html, plotly_fig, gr.Plot(visible=True), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=True)
                    else:
                        # Only HTML content
                        progress(1.0, desc="Analysis complete!")
                        # Hide the plan confirmation and editing sections after execution, show export options
                        yield final_plan_html, outputs_result, None, gr.Plot(visible=False), gr.Row(visible=False), gr.Row(visible=False), gr.Row(visible=True)
                else:
                    error_msg = f"❌ Analysis failed: {results.get('error', 'Unknown error')}"
                    yield self._format_editable_plan(self.current_plan), error_msg, None, gr.Plot(visible=False), gr.Row(visible=True), gr.Row(visible=True), gr.Row(visible=False)
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filepath)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Plan execution error: {str(e)}")
            yield f"❌ Plan execution error: {str(e)}", "", None, gr.Plot(visible=False), gr.Row(visible=True), gr.Row(visible=True), gr.Row(visible=False)
    
    def _execute_confirmed_plan(self, file_path: str, user_query: str, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pre-confirmed analysis plan."""
        try:
            # Initialize analysis session
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Clear previous analysis state
            self.analyzer.current_analysis = {
                "data_package": None,
                "plan": None,
                "step_results": {},
                "dependencies": {},
                "outputs_for_display": [],
                "analysis_id": analysis_id
            }
            
            # Clear previous analysis outputs
            self.analyzer.react_agent.clear_history()
            
            logger.info(f"Executing confirmed plan for analysis {analysis_id}")
            
            # Step 1: Load and process data
            df, metadata = self.analyzer.data_processor.load_data(file_path)
            data_package = self.analyzer.data_processor.prepare_data_for_analysis(df, metadata)
            self.analyzer.current_analysis["data_package"] = data_package
            
            # Step 2: Use the pre-confirmed plan
            self.analyzer.current_analysis["plan"] = plan_result
            
            # Step 3: Execute analysis steps
            execution_results = self.analyzer._execute_analysis_plan(plan_result["plan"], data_package)
            
            # Step 4: Compile final results
            final_results = self.analyzer._compile_results(
                user_query, 
                data_package, 
                plan_result, 
                execution_results
            )
            
            logger.info(f"Confirmed plan execution completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Confirmed plan execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis_id": analysis_id
            }
    
    def _format_editable_plan(self, plan_result: Dict[str, Any]) -> str:
        """Format the analysis plan for editing with confirmation interface."""
        if not plan_result or "plan" not in plan_result:
            return "<p>No analysis plan available.</p>"
        
        plan_steps = plan_result["plan"]
        
        html_parts = [
            "<div style='background: #f8f9fa; padding: 20px; border-radius: 12px; margin: 15px 0; border: 2px solid #007bff;'>",
            "<h3 style='color: #007bff; margin-bottom: 20px; font-size: 1.5em; text-align: center;'>📋 Analysis Plan - Please Review</h3>",
            
            # Plan overview
            "<div style='background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196f3;'>",
            f"<div style='color: #1565c0; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;'>📊 Planned Analysis Steps: {len(plan_steps)}</div>",
            "<div style='color: #1976d2; font-size: 0.9em; margin-top: 5px;'>Review the planned analysis steps below. You can:</div>",
            "<ul style='margin: 10px 0; padding-left: 20px;'>",
            "<li>✅ <strong>Confirm to Proceed</strong> - Execute the plan as is</li>",
            "<li>✏️ <strong>Request Modifications</strong> - Add, remove, or change steps</li>",
            "<li>🔄 <strong>Provide Feedback</strong> - Give instructions for plan refinement</li>",
            "</ul>",
            "</div>",
            
            # Planned steps
            "<h4 style='color: #495057; margin-bottom: 15px; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;'>📝 Planned Steps:</h4>"
        ]
        
        # Display planned steps with more detailed formatting
        if plan_steps:
            for i, step in enumerate(plan_steps, 1):
                step_title = step.get('title', 'Analysis Step')
                step_description = step.get('description', 'No description available')
                output_format = step.get('output_format', 'text')
                display_to_user = step.get('display_to_user', False)
                depends_on = step.get('depends_on', [])
                
                # Choose colors and icons based on output type
                if output_format in ['plot', 'chart', 'visualization']:
                    border_color = "#28a745"
                    step_icon = "📊"
                    format_badge = f"<span style='background: #d4edda; color: #155724; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 500;'>📈 {output_format.title()}</span>"
                elif output_format in ['table', 'dataframe']:
                    border_color = "#17a2b8"
                    step_icon = "📋"
                    format_badge = f"<span style='background: #d1ecf1; color: #0c5460; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 500;'>📊 {output_format.title()}</span>"
                else:
                    border_color = "#6c757d"
                    step_icon = "🔍"
                    format_badge = f"<span style='background: #e2e3e5; color: #383d41; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 500;'>📝 {output_format.title()}</span>"
                
                # Display indicator
                display_badge = ""
                if display_to_user:
                    display_badge = "<span style='background: #fff3cd; color: #856404; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 500; margin-left: 8px;'>👁️ Display</span>"
                
                # Dependencies
                depends_text = ""
                if depends_on and len(depends_on) > 0:
                    depends_text = f"<div style='margin-top: 8px; font-size: 0.85em; color: #6c757d;'><strong>Depends on:</strong> {', '.join(depends_on)}</div>"
                
                html_parts.extend([
                    f"<div style='background: white; padding: 15px; border-radius: 8px; margin: 12px 0; border-left: 4px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>",
                    f"<div style='display: flex; align-items: center; margin-bottom: 8px;'>",
                    f"<h5 style='color: #495057; margin: 0; flex-grow: 1;'>{step_icon} Step {i}: {step_title}{format_badge}{display_badge}</h5>",
                    f"</div>",
                    f"<p style='margin: 8px 0; color: #6c757d; font-size: 0.95em; line-height: 1.4;'>{step_description}</p>",
                    depends_text,
                    "</div>"
                ])
        else:
            html_parts.append("<p style='color: #6c757d; font-style: italic;'>No detailed plan available.</p>")
        
        html_parts.extend([
            "<div style='background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #ffc107;'>",
            "<h4 style='color: #856404; margin: 0 0 10px 0; font-size: 1.1em;'>💡 Next Steps</h4>",
            "<div style='color: #856404; font-size: 0.9em; line-height: 1.4;'>",
            "• <strong>Confirm to Proceed:</strong> Click the 'Execute Plan' button to run the analysis as planned<br>",
            "• <strong>Request Changes:</strong> Use the feedback box to request modifications (e.g., 'Add a correlation matrix', 'Remove step 2', 'Change visualization type')<br>",
            "• <strong>Start Over:</strong> Enter a new question to generate a different plan",
            "</div>",
            "</div>",
            "</div>"
        ])
        
        return "".join(html_parts)
    
    def _check_image_export_dependencies(self) -> Tuple[bool, str]:
        """Check if required dependencies for image export are available."""
        missing_deps = []
        
        try:
            import kaleido
        except ImportError:
            missing_deps.append("kaleido (for plotly image export)")
        
        try:
            from reportlab.platypus import Image
        except ImportError:
            missing_deps.append("reportlab (for PDF generation)")
        
        if missing_deps:
            return False, f"Missing dependencies: {', '.join(missing_deps)}. Install with: pip install {' '.join(missing_deps)}"
        
        return True, "All image export dependencies are available"
    
    def _check_and_warn_image_export(self):
        """Check image export dependencies and log warnings if missing."""
        available, message = self._check_image_export_dependencies()
        if not available:
            logger.warning(f"Image export may fail: {message}")
        return available


    # Main launcher
    def launch_app():
        print("🏥 Clinical Data Analysis Platform - Plan Confirmation Feature")
        port = int(os.getenv("PORT", 7860))
        print(f"🚀 Launching application on http://127.0.0.1:{port}")
        print("   Press Ctrl+C to stop the server")
        print()
        
        # Create and launch the app
        app_instance = ClinicalAnalysisApp()
        interface = app_instance.create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            debug=True,
            show_error=True
        )
    
if __name__ == "__main__":
    ClinicalAnalysisApp.launch_app()
