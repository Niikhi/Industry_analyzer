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
    ) -> Dict[str, str]:
        """Generate a comprehensive proposal with research summary and use cases."""
        try:
            research_summary = self._generate_research_summary(industry_data)
            
            strategic_analysis = self._generate_strategic_analysis(industry_data, use_cases)
            
            implementation_plan = self._format_implementation_plan(use_cases, resources)
            
            executive_summary = self._generate_executive_summary(
                research_summary, 
                strategic_analysis, 
                use_cases
            )
            
            full_report = f"""# AI Implementation Proposal

{executive_summary}

{research_summary}

{strategic_analysis}

{implementation_plan}

## Next Steps and Timeline
{self._generate_next_steps(use_cases)}"""

            return {
                "executive_summary": executive_summary,
                "research_summary": research_summary,
                "strategic_analysis": strategic_analysis,
                "implementation_plan": implementation_plan,
                "full_report": full_report
            }
        except Exception as e:
            print(f"Error in generate_proposal: {str(e)}")
            return {"error": f"Error generating proposal: {str(e)}"}

    def _generate_research_summary(self, industry_data: Dict[str, Any]) -> str:
        """Generate a comprehensive research summary using company research data."""
        try:
            company_info = industry_data.get("companyBackground", {})
            market_position = industry_data.get("marketPosition", {})
            industry_trends = industry_data.get("industryTrends", {})

            company_overview = f"""## Company Overview

    **{company_info.get('name')}** is a {company_info.get('industry')} company headquartered in {company_info.get('headquarters')}. 
    With annual revenue of {company_info.get('revenue')}, the company is led by CEO {company_info.get('ceo')}.

    ### Market Position
    {market_position.get('overview', '')}

    **Key Strengths:**
    {self._format_bullet_points(market_position.get('strengths', []))}

    **Areas for Improvement:**
    {self._format_bullet_points(market_position.get('weaknesses', []))}

    ### Industry Landscape
    {industry_trends.get('overview', '')}

    **Key Industry Trends:**"""

            for trend in industry_trends.get('trends', []):
                company_overview += f"\n\n**{trend.get('name')}**\n{trend.get('description')}"

            return company_overview

        except Exception as e:
            print(f"Error in research summary generation: {str(e)}")
            return "Error generating research summary"

    def _format_bullet_points(self, items: List[str]) -> str:
        """Helper method to format list items as bullet points."""
        return '\n'.join(f"- {item}" for item in items)


    def _generate_strategic_analysis(self, industry_data: Dict[str, Any], use_cases: List[Dict[str, Any]]) -> str:
        """Generate strategic analysis using Groq."""
        prompt = f"""
        Based on the following company data and proposed use cases, provide a strategic analysis:
        
        Company Data:
        {json.dumps(industry_data, indent=2)}
        
        Proposed Use Cases:
        {json.dumps(use_cases, indent=2)}
        
        Focus on:
        1. Strategic fit of AI initiatives
        2. Expected business impact
        3. Risk assessment
        4. Critical success factors
        
        Format the response in markdown with clear sections and bullet points.
        """
        
        analysis = self._query_groq(prompt)
        return f"## Strategic Analysis\n\n{analysis}"

    def _format_implementation_plan(self, use_cases: Dict[str, Any], resources: Dict[str, Any]) -> str:
        """Format detailed implementation plan with use cases and resources."""
        sections = ["## Implementation Plan"]
        
        grouped_cases = self._group_use_cases(use_cases)
        
        for domain, cases in grouped_cases.items():
            sections.append(f"\n### {domain}")
            for i, use_case in enumerate(cases, 1):
                sections.append(self._format_detailed_use_case(i, use_case, resources))
        
        return "\n\n".join(sections)

    def _format_detailed_use_case(self, index: int, use_case: Dict[str, Any], resources: Dict[str, Any]) -> str:
        """Format a detailed use case with implementation details."""
        title = use_case.get('Use Case', f'Use Case {index}')
        
        implementation_prompt = f"""
        For the following AI use case, provide detailed implementation guidance:
        {json.dumps(use_case, indent=2)}
        
        Include:
        1. Technical requirements
        2. Implementation steps
        3. Success metrics
        4. Potential challenges
        
        Format the response in markdown with clear sections and bullet points.
        """
        
        implementation_details = self._query_groq(implementation_prompt)
        
        resources_section = self._format_resources_section(title, resources)
        
        return f"""#### {index}. {title}

{implementation_details}

{resources_section}"""

    def _generate_executive_summary(self, research_summary: str, strategic_analysis: str, use_cases: List[Dict[str, Any]]) -> str:
        """Generate executive summary using Groq."""
        prompt = f"""
        Create an executive summary based on the following information:
        
        Research Summary:
        {research_summary}
        
        Strategic Analysis:
        {strategic_analysis}
        
        Number of Use Cases: {len(use_cases)}
        
        Focus on:
        1. Key opportunities
        2. Expected business impact
        3. Resource requirements
        4. Timeline overview
        
        Format the response in markdown, keep it concise and impactful.
        """
        
        summary = self._query_groq(prompt)
        return f"## Executive Summary\n\n{summary}"

    def _generate_next_steps(self, use_cases: List[Dict[str, Any]]) -> str:
        """Generate next steps and timeline using Groq."""
        prompt = f"""
        Based on these use cases, provide a detailed next steps and timeline plan:
        {json.dumps(use_cases, indent=2)}
        
        Include:
        1. Immediate next steps (30 days)
        2. Short-term milestones (90 days)
        3. Long-term objectives (12 months)
        4. Key dependencies and prerequisites
        
        Format the response in markdown with clear sections and timelines.
        """
        
        return self._query_groq(prompt)

    def _group_use_cases(self, use_cases: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group use cases by domain or category."""
        if isinstance(use_cases, dict) and any(isinstance(v, list) for v in use_cases.values()):
            return use_cases
        
        cases = use_cases if isinstance(use_cases, list) else use_cases.get('use_cases', [])
        grouped = {}
        for case in cases:
            category = case.get('Category', 'General AI Initiatives')
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(case)
        return grouped

    def _format_resources_section(self, use_case_title: str, resources: Dict[str, Any]) -> str:
        """Format resources section for a use case."""
        resources_for_case = resources.get(use_case_title, {})
        if not resources_for_case:
            return ""
        
        sections = ["\n**Resources & Implementation Guidance:**"]
        
        if "Resources" in resources_for_case:
            sections.append("\n*Key Resources:*")
            for resource in resources_for_case["Resources"]:
                sections.append(f"- [{resource.get('Name', 'Resource')}]({resource.get('URL', '#')}) - {resource.get('Description', 'No description')}")
        
        if "Implementation Examples" in resources_for_case:
            sections.append("\n*Implementation Examples:*")
            for example in resources_for_case["Implementation Examples"]:
                sections.append(f"- [{example.get('Name', 'Example')}]({example.get('URL', '#')}) - {example.get('Description', 'No description')}")
        
        return "\n".join(sections)

    def _query_groq(self, prompt: str) -> str:
        """Query Groq API with improved error handling."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an AI implementation specialist creating detailed, 
                    practical proposals. Focus on clear, actionable insights and maintain 
                    professional markdown formatting."""
                },
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
            return f"Error generating content: {str(e)}"

def create_proposal_agent(llm: Any) -> Agent:
    """Create a CrewAI agent for proposal generation."""
    return Agent(
        role="AI Implementation Specialist",
        goal="Create comprehensive AI implementation proposals with practical use cases",
        backstory="Expert in analyzing companies and creating targeted AI implementation strategies",
        llm=llm,
        verbose=True
    )
