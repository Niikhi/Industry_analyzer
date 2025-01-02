import json
from typing import Dict, Any

def generate_markdown_report(company_name: str, research_results: Dict[str, Any], 
                           use_cases: Dict[str, Any], resources: Dict[str, Any], 
                           proposal: str) -> str:
    """Generate a formatted markdown report from the research results."""
    report = f"""# AI Implementation Proposal for {company_name}

## Executive Summary
{json.dumps(research_results, indent=2)}

## Use Cases and Solutions
"""
    # Add use cases section
    if isinstance(use_cases, dict) and 'UseCases' in use_cases:
        for use_case in use_cases['UseCases']:
            report += f"\n### {use_case.get('Trend', 'Use Case')}\n"
            report += f"{use_case.get('UseCase', '')}\n\n"
            report += f"**Expected Outcome:** {use_case.get('Outcome', '')}\n\n"

            # Add resources for this use case if available
            if isinstance(resources, dict):
                for category, res_list in resources.items():
                    if 'Resources' in res_list:
                        report += f"**{category} Resources:**\n"
                        for resource in res_list['Resources']:
                            report += f"- [{resource['Name']}]({resource['URL']})\n  {resource['Description']}\n"

    # Add the proposal section
    report += "\n## Implementation Plan\n"
    report += proposal

    return report