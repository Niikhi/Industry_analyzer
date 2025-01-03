"""Module for handling JSON parsing and validation."""
import json
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

def clean_json_string(input_str: str) -> str:
    """Clean and prepare JSON string for parsing."""
    if not isinstance(input_str, str):
        return str(input_str)
    
    # Remove markdown code fences
    cleaned = input_str.replace('```json', '').replace('```', '')
    
    # Remove whitespace and newlines
    cleaned = ' '.join(cleaned.split())
    
    # Fix common JSON formatting issues
    cleaned = (
        cleaned
        .replace(',}', '}')
        .replace(',]', ']')
        .replace(',,', ',')
        .replace('}.{', '},{')
    )
    
    return cleaned.strip()

def extract_json_object(text: str) -> str:
    """Extract JSON object from text by finding matching braces."""
    stack = []
    start = text.find('{')
    
    if start == -1:
        return text
        
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            if stack:
                start_pos = stack.pop()
                if not stack:  # Found complete object
                    return text[start_pos:i+1]
    
    return text[start:]  # Return partial if no complete object found

def parse_json_safely(input_data: Union[str, Dict, Any]) -> Dict:
    """Parse JSON data with enhanced error handling."""
    try:
        # If already a dict, return as is
        if isinstance(input_data, dict):
            return input_data
            
        # Convert to string if necessary
        if hasattr(input_data, 'raw'):
            input_str = str(input_data.raw)
        else:
            input_str = str(input_data)
        
        # Clean the input string
        cleaned_str = clean_json_string(input_str)
        
        try:
            # First attempt: direct parsing
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Second attempt: extract JSON object
            json_obj = extract_json_object(cleaned_str)
            return json.loads(json_obj)
            
    except Exception as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Problematic input:\n{input_data[:500]}...")  # Log first 500 chars
        
        # Return a minimal valid JSON object
        return {
            "error": "Failed to parse JSON",
            "raw_data": str(input_data)[:1000]  # Include first 1000 chars of raw data
        }