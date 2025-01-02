from crewai import Agent
from typing import Dict, List, Any
import json
import requests


class ProposalAgent:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_proposal(
        self,
        industry_data: Dict[str, Any],
        use_cases: List[Dict[str, Any]],
        resources: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate a comprehensive proposal document."""
        proposal_sections = {
            "executive_summary": self._generate_executive_summary(industry_data),
            "industry_analysis": self._format_industry_analysis(industry_data),
            "use_cases": self._format_use_cases(use_cases),
            "resources": self._format_resources(resources),
            "implementation_roadmap": self._generate_implementation_roadmap(use_cases),
        }

        return self._compile_proposal(proposal_sections)

    def _generate_executive_summary(self, industry_data: Dict[str, Any]) -> str:
        """Generate an executive summary using Groq."""
        industry_data_json = json.dumps(industry_data, indent=2)
        prompt = (
            f"Create an executive summary for an AI implementation proposal based on:\n"
            f"{industry_data_json}\n\n"
            "Focus on:\n"
            "1. Industry context\n"
            "2. Key opportunities\n"
            "3. Proposed value proposition\n"
            "4. Expected outcomes"
        )
        return self._query_groq(prompt)

    def _generate_implementation_roadmap(self, use_cases: List[Dict[str, Any]]) -> str:
        """Generate an implementation roadmap."""
        use_cases_json = json.dumps(use_cases, indent=2)
        prompt = (
            f"Create a phased implementation roadmap for these AI use cases:\n"
            f"{use_cases_json}\n\n"
            "Include:\n"
            "1. Phase-wise implementation plan\n"
            "2. Timeline estimates\n"
            "3. Key milestones\n"
            "4. Resource requirements\n"
            "5. Risk mitigation strategies"
        )
        return self._query_groq(prompt)

    def _query_groq(self, prompt: str) -> str:
        """Query the Groq API for generating content."""
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

    def _format_industry_analysis(self, industry_data: Dict[str, Any]) -> str:
        """Format industry data into a readable section."""
        return (
            f"### Industry Overview\n"
            f"{industry_data.get('industry_overview', 'No industry overview provided.')}\n\n"
            f"### Market Trends\n"
            f"{industry_data.get('market_trends', 'No market trends provided.')}\n\n"
            f"### Technology Adoption\n"
            f"{industry_data.get('technology_adoption', 'No technology adoption information provided.')}\n"
        )

    def _format_use_cases(self, use_cases: List[Dict[str, Any]]) -> str:
        """Format use case details into a readable section."""
        formatted = []
        for use_case in use_cases:
            formatted.append(
                f"#### {use_case.get('title', 'No Title')}\n"
                f"- **Business Problem**: {use_case.get('business_problem', 'Not specified')}\n"
                f"- **Proposed Solution**: {use_case.get('proposed_solution', 'Not specified')}\n"
                f"- **Expected Benefits**: {use_case.get('expected_benefits', 'Not specified')}\n"
                f"- **Implementation Approach**: {use_case.get('implementation_approach', 'Not specified')}\n"
            )
        return "\n".join(formatted)

    def _format_resources(self, resources: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format resources into a readable section."""
        formatted = []
        for title, resource_list in resources.items():
            resource_lines = "\n".join(
                f"- **{res.get('title', 'No Title')}**: {res.get('url', 'No URL')} - {res.get('description', 'No description provided.')}"
                for res in resource_list
            )
            formatted.append(f"#### {title}\n{resource_lines}\n")
        return "\n".join(formatted)

    def compile_proposal(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Compile all sections into a structured dictionary."""
        return {
            "research_results": sections['industry_analysis'],
            "use_cases": sections['use_cases'],
            "resources": sections['resources'],
            "proposal": sections['implementation_roadmap']
        }

def create_proposal_agent(llm: Any) -> Agent:
    """Create a CrewAI agent for proposal generation."""
    return Agent(
        role="Proposal Specialist",
        goal="Create comprehensive and actionable AI implementation proposals",
        backstory=(
            "You are a skilled proposal writer with expertise in AI/ML technologies. "
            "You excel at creating clear, compelling proposals that effectively communicate "
            "technical solutions to business stakeholders."
        ),
        llm=llm,
        verbose=True,
    )
