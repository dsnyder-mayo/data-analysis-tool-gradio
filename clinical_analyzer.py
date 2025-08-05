"""
Main clinical data analyzer that orchestrates the entire analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from datetime import datetime
import json

from data_processor import ClinicalDataProcessor
from dspy_agents import (
    ClinicalAnalysisPlanner,
    ClinicalReActAgent,
    ClinicalQuestionGenerator
)

logger = logging.getLogger(__name__)

class ClinicalDataAnalyzer:
    """Main orchestrator for clinical data analysis."""
    
    def __init__(self):
        """Initialize the analyzer with DSPy configuration."""
        
        # Configure DSPy based on provider
        import dspy

        self.lm = dspy.LM(
            "vertex_ai/gemini-2.5-flash",
            temperature=0,
            top_p=1,
            top_k=1
        )

        dspy.configure(lm=self.lm)
        # dspy.configure(adapter=dspy.JSONAdapter()) 

        # Initialize components
        self.data_processor = ClinicalDataProcessor()
        self.planner = ClinicalAnalysisPlanner()
        self.react_agent = ClinicalReActAgent(max_iterations=8)
        self.question_generator = ClinicalQuestionGenerator()
        
        # Analysis state
        self.current_analysis = {
            "data_package": None,
            "plan": None,
            "step_results": {},
            "dependencies": {},
            "outputs_for_display": [],
            "analysis_id": None
        }
    
    def analyze_data(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Main method to analyze clinical data based on user query."""
        
        try:
            # Initialize analysis session
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Clear previous analysis state to prevent persistence across questions
            self.current_analysis = {
                "data_package": None,
                "plan": None,
                "step_results": {},
                "dependencies": {},
                "outputs_for_display": [],
                "analysis_id": analysis_id
            }
            
            # Clear previous analysis outputs to prevent persistence across questions
            self.react_agent.clear_history()
            
            logger.info(f"Starting analysis {analysis_id}")
            logger.info(f"User query: {user_query}")
            
            # Step 1: Load and process data
            logger.info("Step 1: Loading and processing data...")
            df, metadata = self.data_processor.load_data(file_path)
            data_package = self.data_processor.prepare_data_for_analysis(df, metadata)
            self.current_analysis["data_package"] = data_package
            
            # Step 2: Generate analysis plan
            logger.info("Step 2: Generating analysis plan...")
            plan_result = self.planner(user_query, data_package)
            self.current_analysis["plan"] = plan_result
            
            # Step 3: Execute analysis steps
            logger.info("Step 3: Executing analysis steps...")
            execution_results = self._execute_analysis_plan(plan_result["plan"], data_package)
            
            # Step 4: Compile final results
            logger.info("Step 4: Compiling final results...")
            final_results = self._compile_results(
                user_query, 
                data_package, 
                plan_result, 
                execution_results
            )
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis_id": self.current_analysis.get("analysis_id", "unknown")
            }
    
    def _execute_analysis_plan(self, plan_steps: List[Dict[str, Any]], data_package: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute each step of the analysis plan with enhanced visualization guidance."""
        
        execution_results = []
        
        for i, step in enumerate(plan_steps):
            logger.info(f"Executing step {i+1}: {step['title']}")
            
            # Execute step using ReAct agent with enhanced context
            step_result = self.react_agent.execute_step_with_validation(
                step, 
                data_package,
                plan_steps
            )

            execution_results.append(step_result)
        return execution_results
    
    
    def _compile_results(self, user_query: str, data_package: Dict[str, Any], 
                        plan_result: Dict[str, Any], execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile final results for presentation."""
        
        # Separate successful and failed steps
        successful_steps = [r for r in execution_results if r["success"]]
        failed_steps = [r for r in execution_results if not r["success"]]

        outputs = self.react_agent.displayed_outputs
        
        # Combine plan details with execution trajectories
        plan_details_with_trajectories = []
        for i, plan_step in enumerate(plan_result["plan"]):
            step_with_trajectory = plan_step.copy()
            if i < len(execution_results):
                execution_result = execution_results[i]
                step_with_trajectory["execution_trajectory"] = {
                    "success": execution_result.get("success", False),
                    "reasoning": execution_result.get("reasoning", ""),
                    "trajectory": execution_result.get("trajectory", ""),
                    "step_title": execution_result.get("step_title", ""),
                    "step_description": execution_result.get("step_description", ""),
                    "intermediate_output": execution_result.get("intermediate_output", None)
                }
            plan_details_with_trajectories.append(step_with_trajectory)
        
        return {
            "success": True,
            "analysis_id": self.current_analysis["analysis_id"],
            "user_query": user_query,
            "analysis_plan": {
                "total_steps": len(plan_result["plan"]),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "plan_details": plan_details_with_trajectories
            },
            "displayed_outputs": outputs,
        }

    def generate_suggested_questions(self) -> Dict[str, Any]:
        """Generate suggested research questions based on uploaded clinical data."""
        
        try:
            # Generate suggested questions using the DSPy component
            questions_result = self.question_generator(
                self.current_analysis["data_package"]
            )
            logger.info(f"Generated {questions_result.get('total_questions', 0)} suggested questions")
            return questions_result
            
        except Exception as e:
            logger.error(f"Failed to generate suggested questions: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "questions": []
            }
