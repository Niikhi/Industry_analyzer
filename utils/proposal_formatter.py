"""Utility functions for formatting proposal sections."""

def format_research_summary(research_data: dict) -> str:
    """Format research data into a readable summary."""
    try:
        company_background = research_data.get("companyBackground", {})
        market_position = research_data.get("marketPosition", {})
        industry_trends = research_data.get("industryTrends", {})

        summary = f"""# Comprehensive AI Implementation Proposal for {company_background.get('name')}

## 1. Company Analysis

### Company Overview
{company_background.get('name')} is a {company_background.get('industry')} company headquartered in {company_background.get('headquarters')}, 
with annual revenue of {company_background.get('revenue')} under the leadership of {company_background.get('ceo')}.

### Market Position
{market_position.get('overview')}

**Key Strengths:**
{_format_list(market_position.get('strengths', []))}

**Current Challenges:**
{_format_list(market_position.get('weaknesses', []))}

### Industry Landscape & Trends
{industry_trends.get('overview')}

**Key Industry Trends:**
"""
        # Add detailed trends
        for trend in industry_trends.get('trends', []):
            summary += f"\n**{trend.get('name')}**\n{trend.get('description')}\n"
            
        return summary
    except Exception as e:
        return f"Error formatting research summary: {str(e)}"

def format_use_cases(use_cases: dict, resources: dict) -> str:
    """Format use cases and their resources into a structured section."""
    try:
        section = "\n## 2. Proposed AI Implementation Strategy\n\n"
        
        for domain, cases in use_cases.items():
            section += f"\n### {domain}\n"
            for case in cases:
                case_title = case.get('Use Case')
                section += f"\n#### {case_title}\n"
                section += f"{case.get('Description')}\n"
                
                # Add related resources
                case_resources = resources.get(case_title, {})
                if case_resources:
                    section += "\n**Implementation Resources:**\n"
                    for resource in case_resources.get('Resources', []):
                        section += f"- [{resource.get('Name')}]({resource.get('URL')}): {resource.get('Description')}\n"
        
        return section
    except Exception as e:
        return f"Error formatting use cases: {str(e)}"

def _format_list(items: list) -> str:
    """Helper function to format list items as bullet points."""
    return '\n'.join(f"- {item}" for item in items)
