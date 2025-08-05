"""
DSPy-based agents for analysis planning and code generation.
"""

import dspy
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json
import logging
import re
import io
import base64
import plotly.graph_objects as go
import plotly.io as pio
import ast

logger = logging.getLogger(__name__)

class AnalysisPlanningSignature(dspy.Signature):
    """Generate a comprehensive analysis plan for clinical data that fully interprets user needs."""
    
    user_query = dspy.InputField(desc="The user's question or instruction for data analysis")
    data_metadata = dspy.InputField(desc="Metadata about the clinical dataset including columns, types, and medical concepts")
    data_preview = dspy.InputField(desc="Preview of the actual data")
    
    # user_intent_analysis = dspy.OutputField(desc="Deep analysis of what the user really wants to understand, including underlying questions they might not have explicitly asked")
    analysis_plan = dspy.OutputField(
        desc=(
            "Return a **JSON array**, where each item is a dictionary representing a step in the analysis plan. First, interpret the user's query deeply: what are they really trying to understand? What related or unspoken questions may exist?\n\n"
            "Then generate a **comprehensive plan** that may includes:\n"
            "- **Insightful visualizations or tables**, even if not explicitly requested.\n"
            "- **Supporting statistics when appropriate** that help contextualize the result.\n"
            "- **Step dependencies**, to define the flow of logic.\n\n"
            "Each step must include the following fields:\n"
            "- `step_number`: A unique identifier for the step (e.g., 1, 2, 3).\n"
            "- `title`: A short name for the step.\n"
            "- `description`: What this step does and *why* it's important.\n"
            "- `display_to_user`: Boolean. Whether to show the output to the user.\n"
            "- `output_format`: One of: text, plot, table, dataframe, interpretation, or None.\n"
            "- `outputs`: One output variable that contains all the results produced from this step.\n"
            "- `depends_on`: A list of variables from previous steps that this depends on, or null.\n\n"
            "Focus on analytical depth, good visual communication, and clinical relevance."
        )
    )

class SuggestedQuestionsSignature(dspy.Signature):
    """Generate suggested research questions that medical researchers might want to ask about clinical data."""
    
    data_metadata = dspy.InputField(desc="Metadata about the clinical dataset including columns, data types, and statistical summaries")
    data_preview = dspy.InputField(desc="Preview of the actual clinical data showing sample rows and values")
    
    suggested_questions = dspy.OutputField(
        desc=(
            "Return a **JSON array** of suggested medical research questions that medical researchers might want to ask about this clinical data. "
            "Each question should be a string.\n"
            "Generate 5 diverse questions that cover different aspects of the data including:\n"
            "- Descriptive statistics and data exploration\n"
            "- Relationships between variables\n"
            "- Clinical outcomes and patterns\n"
            "- Risk factors and predictions\n"
            "- Demographic and temporal analyses\n"
            "- Quality and completeness assessments\n\n"
            "Make questions specific to the actual data columns found in the dataset. Keep the question simple, straightforward, and don't include multiple parts in one question. Don't use any placeholders. Generate questions that can be directly asked without needing additional changes."
        )
    )

class ClinicalAnalysisPlanner(dspy.Module):
    """DSPy module for generating comprehensive analysis plans that fully interpret user needs."""
    
    def __init__(self):
        super().__init__()
        self.plan_generator = dspy.ChainOfThought(AnalysisPlanningSignature)
    
    def forward(self, user_query: str, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive analysis plan that interprets user intent and suggests helpful visualizations."""
        
        # Prepare enhanced metadata summary for deeper analysis
        metadata_summary = self._prepare_enhanced_metadata_summary(data_package)
        
        # Generate the comprehensive plan
        result = self.plan_generator(
            user_query=user_query,
            data_metadata=metadata_summary,
            data_preview=data_package["data_preview"]
        )

        print(result)
        
        # Parse and structure the plan with enhanced components
        try:
            structured_plan = json.loads(self._clean_llm_response(result.analysis_plan))
        except:
            structured_plan = ast.literal_eval(self._clean_llm_response(result.analysis_plan))

        return {
            "plan": structured_plan
        }
    def _clean_llm_response(self, operation: str) -> str:
        # Remove markdown code blocks
        if '```json' in operation:
            start_idx = operation.find('```json') + len('```json')
            end_idx = operation.find('```', start_idx)
            if end_idx != -1:
                operation = operation[start_idx:end_idx]
        elif '```' in operation:
            operation = operation.replace('```', '')
        
        operation = operation.strip()
        return operation

    def _prepare_enhanced_metadata_summary(self, data_package: Dict[str, Any]) -> str:
        """Prepare an enhanced metadata summary that includes clinical context for better planning."""
        metadata = data_package["metadata"]
        
        summary_parts = [
            f"Dataset Overview: {metadata['basic_info']['shape'][0]} rows, {metadata['basic_info']['shape'][1]} columns",
            f"Data completeness: {100 - metadata['basic_info']['missing_percentage']:.1f}% complete"
        ]
        
        # Enhanced column analysis
        for col, info in data_package["column_info"].items():
            col_type = info["type"]
            sample_vals = info["sample_values"][:3]
            unique_count = info.get("unique_count", "unknown")
            
            
            summary_parts.append(
                f"- {col}: {col_type} ({unique_count} unique values) "
                f"| Samples: {sample_vals}"
            )
        
        # Add data quality insights
        summary_parts.append("")
        summary_parts.append("Data Quality Insights:")
        if metadata['basic_info']['missing_percentage'] > 10:
            summary_parts.append("- Significant missing data may require imputation strategies")
        if metadata['basic_info']['shape'][0] < 100:
            summary_parts.append("- Small dataset size may limit statistical power")
        
        return "\n".join(summary_parts)



class ClinicalDataAnalysisSignature(dspy.Signature):
    """DSPy ReACT signature for clinical data analysis with tools."""
    objective = dspy.InputField(desc="Description of the step that needs to be performed within an analysis plan")
    output_var = dspy.InputField(desc="One output variable that contains all the results produced from this step") 
    depends_on = dspy.InputField(desc="A list of variables from previous steps that this step depends on, or null.")
    context = dspy.InputField(desc="Metadata about the available data and overall analysis plan context.")
    # plan = dspy.InputField(desc="Context of the complete analysis plan which contains the current step. This is used to provide additional context about the analysis step being executed, including the overall goal and any relevant details from previous steps. Do not perform all operations in this context, just focus on the current step.")

class DataAnalysisTool:
    """Tool for analyzing clinical data with pandas operations."""
    
    def __init__(self, raw_data: pd.DataFrame, output_var, dependent_vars):
        self.name = "data_analysis"
        self.description = "Execute pandas operations on clinical data."
        self.raw_data = raw_data
        self.output_var = output_var
        self.dependent_vars = dependent_vars if dependent_vars is not None else []
    
    def __call__(self, operation: str) -> str:
        """Execute a pandas operation on the clinical data."""
        try:
            # Clean the operation to remove any markdown formatting
            operation = self._clean_operation(operation)
            # Initialize output variable
            if self.output_var not in globals():
                globals()[self.output_var] = None
            
            # Create safe execution environment with shared variables and import capabilities
            safe_globals = self._create_safe_execution_environment()
            safe_globals.update({var: globals().get(var) for var in self.dependent_vars if var in globals()})
            # print("Safe globals for execution:", safe_globals.keys())
            
            # Execute the operation
            local_vars = {}
            exec(operation, safe_globals, local_vars)

            # extract the expected output variable and store it globally
            if self.output_var in local_vars:
                globals()[self.output_var] = local_vars[self.output_var]

            return "Operation executed successfully, result stored in variable: " + self.output_var
                
        except Exception as e:
            return f"Error executing operation: {str(e)}"
        
    def _clean_operation(self, operation: str) -> str:
        """Clean operation code by removing markdown formatting."""
        # Remove markdown code blocks
        if '```python' in operation:
            start_idx = operation.find('```python') + len('```python')
            end_idx = operation.find('```', start_idx)
            if end_idx != -1:
                operation = operation[start_idx:end_idx]
        elif '```' in operation:
            operation = operation.replace('```', '')
        
        operation = operation.strip()
        return operation

    def _create_safe_execution_environment(self):
        """Create a safe execution environment with dynamic import capabilities."""
        # Create execution environment with essential Python functionality and dynamic imports
        safe_globals = {
            # Core Python functionality - essential builtins for data analysis
            '__builtins__': {
                # Essential builtins that are safe for data analysis
                'len': len, 'max': max, 'min': min, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'reversed': reversed,
                'enumerate': enumerate, 'zip': zip, 'range': range,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'hasattr': hasattr,
                'getattr': getattr, 'setattr': setattr,
                'print': print,  # Allow printing for debugging
                'any': any, 'all': all, 'filter': filter, 'map': map,
                'ord': ord, 'chr': chr, 'bin': bin, 'hex': hex, 'oct': oct,
                'divmod': divmod, 'pow': pow,
                
                # Import functionality - this is key for dynamic imports
                '__import__': __import__,
                
                # Exception types that might be needed
                'ImportError': ImportError,
                'AttributeError': AttributeError,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'Exception': Exception,
                'RuntimeError': RuntimeError,
                'NotImplementedError': NotImplementedError,
                'NameError': NameError,
                'ZeroDivisionError': ZeroDivisionError,
            },
            
            # Only essential pre-loaded modules
            'pd': pd,
            'np': np,
            'df': self.raw_data,
            self.output_var: globals().get(self.output_var),
        }
        
        return safe_globals

class ClinicalVisualizationTool:
    """Tool for creating visualizations of clinical data. Create a list object with the specified variable name containing all visualizations as figure objects."""
    
    def __init__(self, raw_data: pd.DataFrame, output_var, dependent_vars):
        self.raw_data = raw_data
        self.output_var = f"{output_var}_results"
        self.dependent_vars = dependent_vars if dependent_vars is not None else []
        self.name = "visualization"
        self.description = "Create plots and visualizations using matplotlib/seaborn. Create a list object with the specified variable name containing all visualizations as figure objects."
        # self.visualizations = []

    def __call__(self, plot_code: str) -> str:
        """Create a visualization using matplotlib/seaborn."""
        try:
            # Clean the plot code
            plot_code = self._clean_code(plot_code)

            # Initialize output variable
            if self.output_var not in globals():
                globals()[self.output_var] = None
            
            # Create safe execution environment with import capabilities
            safe_globals = self._create_safe_execution_environment()
            safe_globals.update({var: globals()[var] for var in self.dependent_vars})
            # print("Safe globals for execution:", safe_globals.keys())
            
            # Execute the plotting code
            local_vars = {}
            exec(plot_code, safe_globals, local_vars)
            # extract the expected output variable
            if self.output_var in local_vars:
                globals()[self.output_var] = local_vars[self.output_var]
            # result = local.get('figures')
                return f"Visualization created successfully"  
            else:
                return f"Error: you must assign your figures to a variable called {self.output_var} in your code."

        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def _clean_code(self, code: str) -> str:
        """Clean code by removing markdown formatting."""
        if '```python' in code:
            start_idx = code.find('```python') + len('```python')
            end_idx = code.find('```', start_idx)
            if end_idx != -1:
                code = code[start_idx:end_idx]
        elif '```' in code:
            code = code.replace('```', '')
        
        return code.strip()

    def _create_safe_execution_environment(self):
        """Create a safe execution environment with dynamic import capabilities for visualization."""
        # Create execution environment with essential Python functionality and dynamic imports
        safe_globals = {
            # Core Python functionality - essential builtins for data analysis
            '__builtins__': {
                # Essential builtins that are safe for data analysis
                'len': len, 'max': max, 'min': min, 'sum': sum, 'abs': abs,
                'round': round, 'sorted': sorted, 'reversed': reversed,
                'enumerate': enumerate, 'zip': zip, 'range': range,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'hasattr': hasattr,
                'getattr': getattr, 'setattr': setattr,
                'print': print,
                'any': any, 'all': all, 'filter': filter, 'map': map,
                'ord': ord, 'chr': chr, 'bin': bin, 'hex': hex, 'oct': oct,
                'divmod': divmod, 'pow': pow,
                
                # Import functionality - this is key for dynamic imports
                '__import__': __import__,
                
                # Exception types that might be needed
                'ImportError': ImportError,
                'AttributeError': AttributeError,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'Exception': Exception,
                'RuntimeError': RuntimeError,
                'NotImplementedError': NotImplementedError,
                'NameError': NameError,
                'ZeroDivisionError': ZeroDivisionError,
            },
            
            # Only essential pre-loaded modules
            'pd': pd,
            'np': np,
            'df': self.raw_data,
            self.output_var: globals().get(self.output_var),
        }
        
        return safe_globals

class ClinicalSummaryInterpretationSignature(dspy.Signature):
    """Generate clinical interpretation of analysis results for summary purposes."""
    
    analysis_result = dspy.InputField(desc="Raw analysis result that needs interpretation")
    context = dspy.InputField(desc="Context about the analysis performed")

    summary = dspy.OutputField(desc="Summary and interpretation of the analysis result")


class ClinicalQuestionGenerator(dspy.Module):
    """DSPy module for generating suggested research questions based on clinical data."""
    
    def __init__(self):
        super().__init__()
        self.question_generator = dspy.Predict(SuggestedQuestionsSignature)
    
    def forward(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """Generate suggested research questions for medical researchers."""
        
        try:
            # Prepare enhanced metadata summary for deeper analysis
            metadata_summary = self._prepare_enhanced_metadata_summary(data_package)
            
            # Generate the suggested questions
            result = self.question_generator(
                data_metadata=metadata_summary,
                data_preview=data_package["data_preview"]
            )
            
            # Parse the JSON response
            try:
                questions = json.loads(result.suggested_questions)
                if not isinstance(questions, list):
                    raise ValueError("Expected a list of questions")
            except (json.JSONError, ValueError) as e:
                logger.warning(f"Failed to parse questions JSON: {e}")
                # Fallback to default questions
                questions = self._generate_fallback_questions()
            
            return {
                "success": True,
                "questions": questions,
                "total_questions": len(questions)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate suggested questions: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "questions": self._generate_fallback_questions()
            }
    
    def _prepare_enhanced_metadata_summary(self, data_package: Dict[str, Any]) -> str:
        """Prepare an enhanced metadata summary that includes clinical context for better planning."""
        metadata = data_package["metadata"]
        
        summary_parts = [
            f"Dataset Overview: {metadata['basic_info']['shape'][0]} rows, {metadata['basic_info']['shape'][1]} columns",
            f"Data completeness: {100 - metadata['basic_info']['missing_percentage']:.1f}% complete"
        ]
        
        # Enhanced column analysis
        for col, info in data_package["column_info"].items():
            col_type = info["type"]
            # sample_vals = info["sample_values"][:3]
            unique_count = info.get("unique_count", "unknown")
            summary_parts.append(
                f"- {col}: {col_type} ({unique_count} unique values) "
            )
        
        # Add data quality insights
        summary_parts.append("")
        summary_parts.append("Data Quality Insights:")
        if metadata['basic_info']['missing_percentage'] > 10:
            summary_parts.append("- Significant missing data may require imputation strategies")
        if metadata['basic_info']['shape'][0] < 100:
            summary_parts.append("- Small dataset size may limit statistical power")
        
        return "\n".join(summary_parts)
    
    def _generate_fallback_questions(self) -> List[str]:
        """Generate fallback questions when AI generation fails."""
        return [
            "What are the basic descriptive statistics for this dataset?",
            "What is the distribution of values across different variables?",
            "Are there any missing values or data quality issues?",
            "What patterns can be observed in the clinical data?",
            "How do different variables correlate with each other?"
        ]


class ClinicalReActAgent(dspy.Module):
    """DSPy ReACT agent that uses tools to analyze clinical data directly."""
    
    def __init__(self, max_iterations: int = 8):
        super().__init__()
        self.max_iterations = max_iterations
        # Don't initialize ReAct here since tools are data-specific
        self.react = None
        self.displayed_outputs = {}
        self.intermediate_outputs = {}
    
    def execute_step_with_validation(self, step: Dict[str, Any], data_package: Dict[str, Any], plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute analysis step using DSPy ReACT with tools."""
        
        # Extract data and metadata
        data = data_package["data"]
        output_var = step.get("outputs")
        dependent_vars = step.get("depends_on", [])
        
        # Create tools for this analysis with shared state
        data_tool = DataAnalysisTool(raw_data=data, output_var=output_var, dependent_vars=dependent_vars)
        viz_tool = ClinicalVisualizationTool(data, output_var=output_var, dependent_vars=dependent_vars)

        # Initialize ReAct with tools
        tools = [data_tool, viz_tool]
        
        self.react = dspy.ReAct(ClinicalDataAnalysisSignature, tools=tools, max_iters=5)

        
        # Prepare context
        display_to_user = step.get('display_to_user', True)
        # if display_to_user:
        #     # objective = f"{step.get('title', '')}: {step.get('description', '')}."
        #     # if step.get("output_format")=="table" or step.get("output_format")=="dataframe":
        #     #     objective = f"{step.get('title', '')}: {step.get('description', '')}."
        #     # elif step.get("output_format")=="text":
        #     #     # objective = f"{step.get('title', '')}: {step.get('description', '')}. Use the ClinicalSummaryTool to generate interpretation, and make sure to pass the correct analysis results (not placeholders) to the tool. Validate that the interpretation is useful for the question and accurate. If not, return an error message."
        #     #     objective = f"{step.get('title', '')}: {step.get('description', '')}."

        #     # elif step.get("output_format")=="plot":
        #     #     objective = f"{step.get('title', '')}: {step.get('description', '')}."
        # else:
        #     objective = f"{step.get('title', '')}: {step.get('description', '')}"

        if step.get("output_format")=="interpretation":
            interpreter = dspy.Predict(ClinicalSummaryInterpretationSignature)
            analysis_result = {}
            for key in dependent_vars:
                if "plot" in key:
                    analysis_result[key] = [self._convert_figure_to_base64(fig) for fig in globals().get(f"{key}_results", None)]
                else:
                    analysis_result[key] = globals().get(key, None)

            print("analysis result", analysis_result)

            if analysis_result is None:
                return {
                    "success": False,
                    "error": f"Dependent variable '{str(dependent_vars)}' not found.",
                    "step_title": step.get('title', ''),
                    "step_description": step.get('description', ''),
                    "attempts": 1
                }
            interpretation_result = interpreter(
                analysis_result=analysis_result,
                context=f"Current step description: {step.get('description', '')}.\n\n"
            )
            self.displayed_outputs[step.get('title', '')] = interpretation_result.summary
            self.intermediate_outputs[step.get('title', '')] = interpretation_result.summary
            return {
                "success": True,
                "reasoning": None,
                "trajectory": None,
                "step_title": step.get('title', ''),
                "step_description": step.get('description', ''),
                "intermediate_output": interpretation_result.summary,
            }
        else:
            if output_var:
                if step.get("output_format")=="plot":
                    objective = f"{step.get('title', '')}: {step.get('description', '')}. **Create a list called '{output_var}_results' and store all plot figures as objects in it. Do not print or show the figures.**"
                else:
                    objective = f"{step.get('title', '')}: {step.get('description', '')}. **Create a variable called '{output_var}' and store the result in it. Do not print the result.**"
            else:
                objective = f"{step.get('title', '')}: {step.get('description', '')}."

            data_context = self._prepare_data_context(data_package)
            if dependent_vars is None:
                context = data_context
            else:
                context = f"{data_context}\n\n {str(dependent_vars)} variables have been generated and can be accessed in the environment."
            # Use ReAct to analyze the data
            result = self.react(
                objective=objective,
                context=context,
                output_var=output_var,
                depends_on=dependent_vars
            )

            if display_to_user:
                if step.get("output_format")=="text":
                #     # objective = f"{step.get('title', '')}: {step.get('description', '')}. Use the ClinicalSummaryTool to generate interpretation, and make sure to pass the correct analysis results (not placeholders) to the tool. Validate that the interpretation is useful for the question and accurate. If not, return an error message."
                #     objective = f"{step.get('title', '')}: {step.get('description', '')}."
                    interpreter = dspy.Predict(ClinicalSummaryInterpretationSignature)
                    analysis_result = globals().get(output_var, None)
                    if analysis_result is None:
                        return {
                            "success": False,
                            "error": f"Output variable '{output_var}' not found. Ensure the analysis step produces a valid result.",
                            "step_title": step.get('title', ''),
                            "step_description": step.get('description', ''),
                            "attempts": 1
                        }
                    interpretation_result = interpreter(
                        analysis_result=analysis_result,
                        context=step.get('description', '')
                    )
                    self.displayed_outputs[step.get('title', '')] = interpretation_result.summary
                elif step.get("output_format")=="plot":
                    self.displayed_outputs[step.get('title', '')] = globals()[f"{output_var}_results"]
                else:
                    self.displayed_outputs[step.get('title', '')] = globals()[output_var]

            # save intermediate outputs
            if step.get("output_format")=="text":
                self.intermediate_outputs[step.get('title', '')] = globals()[output_var]
            elif step.get("output_format")=="plot":
                self.intermediate_outputs[step.get('title', '')] = globals()[f"{output_var}_results"]
            else:
                self.intermediate_outputs[step.get('title', '')] = globals()[output_var]

            return {
                "success": True,
                "reasoning": str(result.reasoning),
                "trajectory": str(result.trajectory),
                "step_title": step.get('title', ''),
                "step_description": step.get('description', ''),
                "intermediate_output": self.intermediate_outputs.get(step.get('title', ''), None),
                # "display_to_user": display_to_user,
                # "attempts": 1  # ReAct handles its own iterations
            }
        
    def _convert_figure_to_base64(self, fig) -> str:
        """Convert a matplotlib figure to a base64 string for display."""
        if(isinstance(fig, go.Figure)):
            img_bytes = pio.to_image(fig, format='png')
            img_str = base64.b64encode(img_bytes).decode('utf-8')
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
        return img_str
    
    def _prepare_data_context(self, data_package: Dict[str, Any]) -> str:
        """Prepare data context for ReACT reasoning."""
        data = data_package["data"]
        context_parts = [
            f"Dataset shape: {data.shape}",
            f"Available columns: {', '.join(data.columns[:10])}" + ("..." if len(data.columns) > 10 else ""),
            "Tools available:",
            "- data_analysis: Execute pandas operations on the data",
            "- visualization: Create plots and visualizations", 
        ]
        # Add sample data
        context_parts.append(f"\nSample data:\n{data.head(3).to_string()}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear all displayed outputs"""
        self.displayed_outputs.clear()
        self.intermediate_outputs.clear()
    