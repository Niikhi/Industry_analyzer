import logging
from crewai import Agent
from typing import List, Dict, Any
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketStandardsAgent:
    def __init__(self, tavily_api_key: str, groq_api_key: str):
        self.tavily_api_key = tavily_api_key
        self.groq_api_key = groq_api_key
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.tavily_url = "https://api.tavily.com/search"

        if not self.groq_api_key or not self.tavily_api_key:
            raise ValueError("Missing required API keys")

    def analyze_trends(self, industry_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self._create_analysis_prompt(industry_data)
            analysis = self._query_groq(prompt)
            return self._structure_analysis(analysis)
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            raise

    def generate_use_cases(self, industry_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            prompt = self._create_use_case_prompt(industry_analysis)
            use_cases_text = self._query_groq(prompt)
            return self._parse_use_cases(use_cases_text)
        except Exception as e:
            logger.error(f"Error generating use cases: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _query_groq(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are an AI/ML solutions architect specializing in industry applications."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.groq_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Groq API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Groq query: {e}")
            raise

    def _create_analysis_prompt(self, industry_data: Dict[str, Any]) -> str:
        return f"""Analyze the following industry data and identify key AI/ML trends:
        
        Industry Overview: {industry_data.get('industry_overview', 'No data')}
        Market Trends: {industry_data.get('market_trends', 'No data')}
        Technology Adoption: {industry_data.get('technology_adoption', 'No data')}
        
        Format your response as JSON with the following structure:
        {{
            "current_trends": [list of current AI/ML adoption trends],
            "emerging_technologies": [list of emerging technologies],
            "opportunities": [list of market opportunities],
            "challenges": [list of implementation challenges]
        }}"""

    def _create_use_case_prompt(self, analysis: Dict[str, Any]) -> str:
        return f"""Based on this industry analysis, generate 5 specific AI/ML use cases.
        
        Industry Analysis: {json.dumps(analysis, indent=2)}
        
        Format each use case as JSON with:
        - title
        - business_problem
        - proposed_solution
        - expected_benefits
        - implementation_approach
        - estimated_complexity (High/Medium/Low)
        - estimated_impact (High/Medium/Low)"""

    def _structure_analysis(self, analysis_text: str) -> Dict[str, Any]:
        try:
            return json.loads(analysis_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing analysis JSON: {e}")
            return {
                "current_trends": [],
                "emerging_technologies": [],
                "opportunities": [],
                "challenges": []
            }

    def _parse_use_cases(self, use_cases_text: str) -> List[Dict[str, Any]]:
        try:
            use_cases = json.loads(use_cases_text)
            return [self._validate_use_case(uc) for uc in use_cases]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing use cases JSON: {e}")
            return []

    def _validate_use_case(self, use_case: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = [
            "title", "business_problem", "proposed_solution",
            "expected_benefits", "implementation_approach"
        ]
        
        return {
            field: use_case.get(field, "Not specified")
            for field in required_fields
        }

def create_market_standards_agent(llm: Any) -> Agent:
    return Agent(
        role="AI Solutions Architect",
        goal="Analyze industry trends and generate actionable AI/ML use cases",
        backstory="""You are an experienced AI solutions architect who specializes in 
        identifying and developing AI/ML opportunities across various industries. 
        You excel at translating market trends into practical use cases.""",
        llm=llm,
        verbose=True
    )