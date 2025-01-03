"""Module for processing research results."""
import logging
from typing import Dict, Any
from .json_handler import parse_json_safely

logger = logging.getLogger(__name__)

def process_task_output(task_output: Any, task_name: str) -> Dict:
    """Process individual task output with error handling."""
    try:
        result = parse_json_safely(task_output)
        logger.info(f"Successfully processed {task_name} output")
        return result
    except Exception as e:
        logger.error(f"Error processing {task_name} output: {str(e)}")
        return {
            "error": f"Failed to process {task_name}",
            "details": str(e)
        }

def process_research_results(result: Any) -> Dict[str, Any]:
    """Process all research results with comprehensive error handling."""
    processed_results = {
        "research": {},
        "use_cases": {},
        "resources": {},
        "proposal": ""
    }
    
    try:
        if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 4:
            processed_results["research"] = process_task_output(
                result.tasks_output[0], "research"
            )
            processed_results["use_cases"] = process_task_output(
                result.tasks_output[1], "use cases"
            )
            processed_results["resources"] = process_task_output(
                result.tasks_output[2], "resources"
            )
            processed_results["proposal"] = str(result.tasks_output[3]).strip()
            
        return processed_results
    except Exception as e:
        logger.error(f"Error processing research results: {str(e)}")
        return processed_results