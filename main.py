import os
import json
import logging
import streamlit as st
from dotenv import load_dotenv
from crewai import Task, Crew, Process
from langchain_groq import ChatGroq
from typing import Dict, List, Optional
from utils.report_generator import generate_markdown_report
from utils.file_handler import (
    ensure_directories,
    create_safe_filename,
    save_json_file,
    save_markdown_file
)

from agents.research_agent import IndustryResearchAgent, create_research_agent
from agents.market_standards_agent import MarketStandardsAgent, create_market_standards_agent
from agents.resource_asset_agent import ResourceAssetAgent, create_resource_agent
from agents.proposal_agent import ProposalAgent, create_proposal_agent

from utils.json_handler import parse_json_safely
from utils.result_processor import process_research_results



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_json(input_str: str) -> dict:
    """Sanitize and validate JSON string from task output with industry-specific formatting."""
    if isinstance(input_str, dict):
        return input_str
        
    # If it's a TaskOutput object, convert to string
    if hasattr(input_str, 'raw'):
        input_str = str(input_str.raw)
    else:
        input_str = str(input_str)
    
    # Remove any leading/trailing whitespace
    input_str = input_str.strip()
    
    try:
        # First, try to parse as-is
        return json.loads(input_str)
    except json.JSONDecodeError:
        try:
            # If that fails, try to fix the structure
            
            # Split the input into industry sections
            sections = input_str.split('"], "')
            
            formatted_data = {}
            current_industry = None
            
            for section in sections:
                # Look for industry markers
                if '"' in section and ':' in section:
                    # Extract industry name
                    industry_split = section.split('":')
                    if len(industry_split) >= 2:
                        current_industry = industry_split[0].strip().strip('"')
                        use_cases_str = industry_split[1]
                else:
                    use_cases_str = section
                
                if current_industry:
                    # Clean up the use cases string
                    use_cases_str = use_cases_str.strip()
                    if use_cases_str.startswith('['):
                        use_cases_str = use_cases_str
                    if use_cases_str.endswith(']'):
                        use_cases_str += ','
                    
                    # Try to parse the use cases
                    try:
                        # Ensure the string is a valid JSON array
                        if not use_cases_str.startswith('['):
                            use_cases_str = '[' + use_cases_str
                        if not use_cases_str.endswith(']'):
                            use_cases_str = use_cases_str + ']'
                            
                        use_cases = json.loads(use_cases_str)
                        formatted_data[current_industry] = use_cases
                    except json.JSONDecodeError:
                        formatted_data[current_industry] = []
            
            if not formatted_data:
                # If no industry sections found, try parsing as a simple array
                input_str = input_str.strip()
                if input_str.startswith('[') and input_str.endswith(']'):
                    return {"use_cases": json.loads(input_str)}
                else:
                    # Try to extract any array from the string
                    start = input_str.find('[')
                    end = input_str.rfind(']') + 1
                    if start != -1 and end != -1:
                        array_str = input_str[start:end]
                        return {"use_cases": json.loads(array_str)}
            
            return formatted_data

        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.error(f"Raw input:\n{input_str[:500]}")
            return {
                "error": "Failed to parse JSON",
                "raw_data": input_str[:500]
            }


class AIResearchCrew:
    def __init__(self):

        load_dotenv()
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')

        if not self.tavily_api_key or not self.groq_api_key:
            raise ValueError("Missing required API keys in .env file")

        try:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="groq/mixtral-8x7b-32768",
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        try:
            self.research_agent_instance = IndustryResearchAgent(
                tavily_api_key=self.tavily_api_key,
                groq_api_key=self.groq_api_key
            )
            self.market_standards_instance = MarketStandardsAgent(
                tavily_api_key=self.tavily_api_key,
                groq_api_key=self.groq_api_key
            )
            self.resource_asset_instance = ResourceAssetAgent(
                tavily_api_key=self.tavily_api_key
            )
            self.proposal_agent_instance = ProposalAgent(
                groq_api_key=self.groq_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

        self.research_agent = create_research_agent(self.llm)
        self.market_agent = create_market_standards_agent(self.llm)
        self.resource_agent = create_resource_agent(self.llm)
        self.proposal_agent = create_proposal_agent(self.llm)

    def create_tasks(self, company_name: str) -> List[Task]:
        """Create sequential tasks with proper data passing between them."""

        # Task 1: Industry Research
        research_task = Task(
            description=f"Research {company_name} and its industry thoroughly, including company background, market position, and industry trends.",
            agent=self.research_agent,
            expected_output="JSON containing industry research results",
            context=[
                {
                    "key": "company_name",
                    "value": company_name,
                    "description": "Company name for research",
                    "expected_output": "Company details and industry insights in JSON format"
                }
            ]
        )

        # Task 2: Market Standards Analysis
        market_task = Task(
            description="Analyze industry research and generate AI/ML use cases.",
            agent=self.market_agent,
            expected_output="JSON containing use cases",
            context=[
                {
                    "key": "research_results",
                    "value": lambda: research_task.output,
                    "description": "Results from industry research task",
                    "expected_output": "Generated AI/ML use cases"
                }
            ]
        )

        # Task 3: Resource Collection
        resource_task = Task(
            description="Find relevant resources for each use case.",
            agent=self.resource_agent,
            expected_output="JSON containing resources",
            context=[
                {
                    "key": "use_cases",
                    "value": lambda: market_task.output,
                    "description": "Generated use cases from market standards task",
                    "expected_output": "List of resources for use cases"
                }
            ]
        )

        # Task 4: Proposal Generation
        proposal_task = Task(
        description="Create a comprehensive AI implementation proposal using company-specific data.",
            agent=self.proposal_agent,
            expected_output="Markdown formatted proposal",
            context=[
                {
                    "key": "research_results",
                    "value": lambda: research_task.output,
                    "description": "Industry research results",
                    "expected_output": "Industry research details"
                },
                {
                    "key": "use_cases",
                    "value": lambda: market_task.output,
                    "description": "Generated use cases",
                    "expected_output": "Proposed AI/ML use cases"
                },
                {
                    "key": "resources",
                    "value": lambda: resource_task.output,
                    "description": "Relevant resources",
                    "expected_output": "Details of resources for use cases"
                }
            ]
        )

        return [research_task, market_task, resource_task, proposal_task]

    def save_results(self, result, company_name: str) -> None:
        """Save research results to files with improved error handling."""
        try:
            ensure_directories('outputs', 'reports')
            safe_company_name = create_safe_filename(company_name)
            
            # Initialize default values
            research_results = {}
            use_cases = {}
            resources = {}
            proposal = ""
            
            # Process each task output with better error handling
            if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 4:
                # Research Results (Task 0)
                research_results = sanitize_json(result.tasks_output[0])
                
                # Use Cases (Task 1)
                use_cases = sanitize_json(result.tasks_output[1])
                
                # Resources (Task 2)
                resources = sanitize_json(result.tasks_output[2])
                
                # Proposal (Task 3) - Keep as string
                proposal = str(result.tasks_output[3]).strip()
            
            # Prepare the complete output
            json_output = {
                "company_info": research_results,
                "use_cases": use_cases,
                "resources": resources,
                "proposal": proposal
            }
            
            # Save JSON output
            output_path = os.path.join('outputs', f'{safe_company_name}_research_output.json')
            save_json_file(json_output, output_path)
            
            # Generate and save markdown report
            report_content = generate_markdown_report(
                company_name, research_results, use_cases, resources, proposal
            )
            report_path = os.path.join('reports', f'{safe_company_name}_ai_proposal.md')
            save_markdown_file(report_content, report_path)
            
        except Exception as e:
            logger.error(f"Error in save_results: {str(e)}")
            if hasattr(result, 'tasks_output'):
                for i, output in enumerate(result.tasks_output):
                    logger.error(f"Task {i} output preview: {str(output)[:200]}...")
            raise

    def _generate_markdown_report(self, data: Dict, company_name: str) -> str:
        """Generate a formatted markdown report."""
        report = f"""# AI Implementation Proposal for {company_name}

    ## Executive Summary
    {json.dumps(data['company_info'], indent=2)}

    ## Use Cases and Solutions

    """
        # Add use cases by domain
        for domain, use_cases in data['use_cases'].items():
            report += f"\n### {domain}\n"
            for use_case in use_cases:
                report += f"\n#### {use_case['Use Case']}\n"
                report += f"{use_case['Description']}\n\n"

                # Add resources for this use case
                matching_resources = data['resources'].get(use_case['Use Case'], {})
                if matching_resources:
                    report += "**Relevant Resources:**\n"
                    for res_key, resource in matching_resources.items():
                        report += f"- [{resource['Name']}]({resource['URL']})\n  {resource['Description']}\n"

        # Add the proposal section
        report += "\n## Implementation Plan\n"
        report += data['proposal']

        return report

    def run_research(self, company_name: str) -> Dict:
        """Execute the research process with proper error handling."""
        try:
            if not company_name or not isinstance(company_name, str):
                raise ValueError("Invalid company name provided")

            tasks = self.create_tasks(company_name)
            for task in tasks:
                if not task.expected_output:
                    raise ValueError(f"Missing expected output for task: {task.description}")

            crew = Crew(
                agents=[
                    self.research_agent,
                    self.market_agent,
                    self.resource_agent,
                    self.proposal_agent
                ],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )

            # Execute tasks
            result = crew.kickoff()

            # Debugging: Log raw result and task outputs
            logger.debug(f"Raw result from crew.kickoff(): {result}")

            for i, task in enumerate(tasks):
                print(f"DEBUG: Output of Task {i+1} ({task.description}):", task.output)


            # Save results
            self.save_results(result, company_name)

            return result

        except Exception as e:
            logger.error(f"Error in research process: {e}")
            raise

def initialize_session():
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'download_ready' not in st.session_state:
        st.session_state.download_ready = False

def main():
    st.set_page_config(page_title="AI Implementation Research Tool", layout="wide")
    
    st.title("AI Implementation Research Tool")
    initialize_session()

    try:
        crew = AIResearchCrew()
        
        with st.form("research_form"):
            company_name = st.text_input("Enter Company Name:")
            submit_button = st.form_submit_button("Start Research")

        if submit_button and company_name:
            st.info("Research in progress... Please wait.")
            
            with st.spinner("Conducting research..."):
                progress_bar = st.progress(0)
                
                # Update progress for each step
                steps = ["Industry Research", "Market Analysis", "Resource Collection", "Proposal Generation"]
                for i, step in enumerate(steps):
                    progress_bar.progress((i + 1) * 25)
                    st.write(f"Step {i+1}: {step}")
                    
                try:
                    result = crew.run_research(company_name)
                    st.session_state.research_results = result
                    st.session_state.download_ready = True
                    progress_bar.progress(100)
                except Exception as e:
                    st.error(f"Error during research: {str(e)}")
                    return

            if st.session_state.download_ready:
                st.success("Research completed successfully!")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["Company Research", "Use Cases", "Resources", "Proposal"])
                
                try:
                    processed_results = process_research_results(result)
                    
                    with tab1:
                        st.json(processed_results["research"])
                    
                    with tab2:
                        st.json(processed_results["use_cases"])
                    
                    with tab3:
                        st.json(processed_results["resources"])
                    
                    with tab4:
                        st.markdown(processed_results["proposal"])
                    
                    # Add download buttons
                    st.download_button(
                        label="Download JSON Results",
                        data=json.dumps({
                            "company_info": processed_results["research"],
                            "use_cases": processed_results["use_cases"],
                            "resources": processed_results["resources"],
                            "proposal": processed_results["proposal"]
                        }, indent=2),
                        file_name=f"{create_safe_filename(company_name)}_research_output.json",
                        mime="application/json"
                    )
                    
                    report_content = generate_markdown_report(
                        company_name,
                        processed_results["research"],
                        processed_results["use_cases"],
                        processed_results["resources"],
                        processed_results["proposal"]
                    )
                    
                    st.download_button(
                        label="Download Markdown Report",
                        data=report_content,
                        file_name=f"{create_safe_filename(company_name)}_ai_proposal.md",
                        mime="text/markdown"
                    )
                
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")

    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()