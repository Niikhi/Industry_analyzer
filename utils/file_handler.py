import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_safe_filename(name: str) -> str:
    """Create a safe filename from input string."""
    return "".join(
        c for c in name.lower() if c.isalnum() or c in (' ', '-', '_')
    ).strip().replace(' ', '_')

def ensure_directories(*paths: str) -> None:
    """Ensure all specified directories exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def save_json_file(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {str(e)}")
        raise

def save_markdown_file(content: str, filepath: str) -> None:
    """Save markdown content to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully saved markdown to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save markdown file {filepath}: {str(e)}")
        raise