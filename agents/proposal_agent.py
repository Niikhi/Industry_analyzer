from crewai import Agent
from typing import Dict, List, Any
import json
import requests

class ProposalAgent:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_proposal(self, industry_data: Dict, use_cases: Dict, resources: Dict) -> str:
        company_name = industry_data.get("company_name", "the company")
        
        # Generate industry summary
        industry_summary = self._generate_industry_summary(industry_data)
        
        # Format use cases with resources
        use_cases_section = self._format_use_cases_with_resources(use_cases, resources)
        
        return self._compile_markdown_report(company_name, industry_summary, use_cases_section)

    def _generate_industry_summary(self, industry_data: Dict) -> str:
        prompt = f"""Based on this industry data:
        {json.dumps(industry_data, indent=2)}
        
        Create a concise summary focusing on:
        1. Industry overview
        2. Current technological state
        3. Key opportunities for AI/ML implementation
        """
        return self._query_groq(prompt)

    def _format_use_cases_with_resources(self, use_cases: Dict, resources: Dict) -> str:
        formatted_content = []
        
        for domain, cases in use_cases.items():
            formatted_content.append(f"## {domain} Use Cases\n")
            
            for idx, use_case in enumerate(cases, 1):
                case_title = use_case.get("Use Case", "Unnamed Use Case")
                
                formatted_content.append(f"""
**Use Case {idx}: {case_title}**
* **Objective/Use Case**: {use_case.get('Description', 'Not specified')}
* **AI Application**: {use_case.get('Solution', 'Not specified')}
* **Cross-Functional Benefits**:""")
                
                # Add benefits as bullet points
                benefits = use_case.get('Benefits', '').split('\n')
                for benefit in benefits:
                    if benefit.strip():
                        formatted_content.append(f"   * {benefit.strip()}")
                
                # Add relevant resources if available
                if case_title in resources:
                    formatted_content.append("\n**Relevant Resources:**")
                    for resource in resources[case_title]:
                        formatted_content.append(
                            f"* [{resource['Name']}]({resource['URL']}) - {resource['Description']}"
                        )
                
                formatted_content.append("\n")  # Add spacing between use cases
                
        return "\n".join(formatted_content)

    def _compile_markdown_report(self, company_name: str, industry_summary: str, use_cases_section: str) -> str:
        return f"""# AI Implementation Proposal for {company_name}

## Industry Analysis and Opportunities
{industry_summary}

## Proposed AI/ML Solutions
{use_cases_section}
"""

    def _query_groq(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are an AI expert specializing in proposal generation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        try:
            response = requests.post(self.groq_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content")
        except requests.exceptions.RequestException as e:
            print(f"Error querying Groq API: {e}")
            raise

def create_proposal_agent(llm: Any) -> Agent:
    return Agent(
        role="Proposal Specialist",
        goal="Create clear and actionable AI implementation proposals",
        backstory="Expert in creating concise, well-structured AI implementation proposals",
        llm=llm,
        verbose=True,
    )