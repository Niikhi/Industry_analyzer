import logging
from crewai import Agent
from typing import List, Dict, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceAssetAgent:
    def __init__(self, tavily_api_key: str):
        self.tavily_api_key = tavily_api_key
        self.tavily_url = "https://api.tavily.com/search"
        
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required")

        self.resource_domains = {
            'datasets': ['kaggle.com', 'huggingface.co', 'data.world', 'google.com/datasetsearch'],
            'github': ['github.com', 'gitlab.com'],
            'documentation': ['docs.', 'documentation.', 'learn.', 'tutorial.']
        }

    def find_resources(self, use_cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        resources = {}
        for use_case in use_cases:
            try:
                use_case_title = use_case.get('title', 'Unnamed Use Case')
                resources[use_case_title] = {
                    'datasets': self._search_datasets(use_case),
                    'implementations': self._search_implementations(use_case),
                    'documentation': self._search_documentation(use_case)
                }
            except Exception as e:
                logger.error(f"Error finding resources for use case '{use_case.get('title', 'Unknown')}': {e}")
                resources[use_case_title] = {'error': str(e)}
        return resources

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _tavily_search(self, query: str, domain_filter: List[str] = None) -> List[Dict[str, Any]]:
        headers = {
            'Authorization': f'Bearer {self.tavily_api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'query': query,
            'max_results': 5,
            'search_depth': 'advanced',
            'include_raw_content': True
        }

        try:
            response = requests.post(self.tavily_url, headers=headers, json=payload)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if domain_filter:
                results = [r for r in results if any(domain in r.get('url', '').lower() for domain in domain_filter)]
            
            return self._format_results(results)
        except requests.RequestException as e:
            logger.error(f"Tavily API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            return []

    def _search_datasets(self, use_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = f"{use_case.get('title', '')} {use_case.get('business_problem', '')} dataset ML AI"
        return self._tavily_search(query, self.resource_domains['datasets'])

    def _search_implementations(self, use_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = f"{use_case.get('title', '')} {use_case.get('proposed_solution', '')} github implementation"
        return self._tavily_search(query, self.resource_domains['github'])

    def _search_documentation(self, use_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = f"{use_case.get('title', '')} {use_case.get('proposed_solution', '')} documentation tutorial"
        return self._tavily_search(query, self.resource_domains['documentation'])

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_results = []
        for result in results:
            try:
                formatted_results.append({
                    'title': self._clean_title(result.get('title', 'Untitled')),
                    'url': result.get('url', ''),
                    'description': self._clean_description(result.get('content', '')[:300]),
                    'source_type': self._determine_source_type(result.get('url', ''))
                })
            except Exception as e:
                logger.error(f"Error formatting result: {e}")
        return formatted_results

    @staticmethod
    def _clean_title(title: str) -> str:
        return re.sub(r'\s+', ' ', title).strip()

    @staticmethod
    def _clean_description(description: str) -> str:
        description = re.sub(r'\s+', ' ', description).strip()
        return f"{description}..." if len(description) == 300 else description

    def _determine_source_type(self, url: str) -> str:
        url_lower = url.lower()
        for source_type, domains in self.resource_domains.items():
            if any(domain in url_lower for domain in domains):
                return source_type
        return 'other'

def create_resource_agent(llm: Any) -> Agent:
    return Agent(
        role='Resource Specialist',
        goal='Find and validate relevant AI/ML implementation resources',
        backstory="""You are a technical resource specialist who excels at finding 
        and evaluating AI/ML resources, datasets, and implementation examples.""",
        llm=llm,
        verbose=True
    )