import logging
from crewai import Agent
from typing import List, Dict, Optional, Any
import requests
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndustryInsight:
    industry_name: str
    key_trends: List[str]
    ai_applications: List[Dict[str, str]]
    market_size: Optional[str] = None

class IndustryResearchAgent:
    def __init__(self, tavily_api_key: str, groq_api_key: str, max_results: int = 5):
        self.tavily_api_key = tavily_api_key
        self.groq_api_key = groq_api_key
        self.tavily_url = "https://api.tavily.com/search"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.max_results = max_results

        if not self.tavily_api_key or not self.groq_api_key:
            raise ValueError("Missing required API keys")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_industry(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            'Authorization': f'Bearer {self.tavily_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'query': query,
            'max_results': self.max_results,
            'search_depth': 'advanced',
            'include_raw_content': True
        }

        try:
            response = requests.post(self.tavily_url, headers=headers, json=payload)
            response.raise_for_status()
            return self._parse_results(response.json())
        except requests.RequestException as e:
            logger.error(f"Tavily API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise

    def _parse_results(self, results: dict) -> List[Dict[str, Any]]:
        parsed_results = []
        try:
            for item in results.get("results", []):
                description = self._get_best_description(item)
                parsed_results.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", "No URL"),
                    "snippet": self._clean_description(description),
                    "displayed_url": item.get("url", "No URL")
                })
            return parsed_results
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            return []

    @staticmethod
    def _clean_description(description: str, max_length: int = 500) -> str:
        if not description:
            return "No description available"
        description = " ".join(description.split())
        return description[:max_length - 3] + "..." if len(description) > max_length else description

    @staticmethod
    def _get_best_description(item: dict) -> str:
        return (
            item.get("content") or
            item.get("snippet") or
            item.get("description") or
            "No description available"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No results to summarize."

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an industry analyst specializing in AI/ML applications. "
                    "Summarize search results into clear, actionable insights."
                )
            },
            {
                "role": "user",
                "content": "Summarize the following search results:\n" + 
                "\n".join([
                    f"- {result['title']}\n"
                    f"Source: {result['displayed_url']}\n"
                    f"Content: {result['snippet']}\n"
                    for result in results
                ])
            }
        ]

        try:
            response = requests.post(
                self.groq_url,
                headers=headers,
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": messages,
                    "max_tokens": 400,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Groq API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise

    def analyze_company(self, company_name: str) -> Dict[str, Any]:
        try:
            # Search for company information
            company_results = self.search_industry(f"{company_name} company overview industry analysis")
            industry_results = self.search_industry(f"{company_name} industry trends AI ML adoption")
            
            # Summarize findings
            company_summary = self.summarize_results(company_results)
            industry_summary = self.summarize_results(industry_results)
            
            return {
                "company_overview": company_summary,
                "industry_analysis": industry_summary,
                "raw_data": {
                    "company_results": company_results,
                    "industry_results": industry_results
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing company {company_name}: {e}")
            raise

def create_research_agent(llm) -> Agent:
    return Agent(
        role='Industry Research Specialist',
        goal='Gather comprehensive industry data and insights using AI-powered search and analysis.',
        backstory="""You are an experienced industry analyst specializing in AI/ML technologies.
        Your expertise lies in identifying market trends, technological innovations, and strategic opportunities.""",
        llm=llm,
        verbose=True
    )