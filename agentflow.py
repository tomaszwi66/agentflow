# ==============================================================================
# Autonomous AI Market Research Agency (CrewAI & Google Gemini)
# ==============================================================================

# Uncomment the line below if running directly in Google Colab
# !pip install -U ddgs crewai tavily-python

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

# ==========================================
# 1. LLM & TOOL CONFIGURATION
# ==========================================
# ⚠️ WARNING: Never hardcode your actual API key in a public repository!
MY_API_KEY = "" # Add your Google Gemini API Key here
os.environ["GEMINI_API_KEY"] = MY_API_KEY

# Optional: Set TAVILY_API_KEY to use Tavily search instead of DuckDuckGo.
# When TAVILY_API_KEY is present, Tavily is used; otherwise DuckDuckGo is the default.
# os.environ["TAVILY_API_KEY"] = ""  # Add your Tavily API key here (optional)

# Initialize Tavily client once at module level if the key is available.
_tavily_client = None
_tavily_api_key = os.environ.get("TAVILY_API_KEY")
if _tavily_api_key:
    from tavily import TavilyClient
    _tavily_client = TavilyClient(api_key=_tavily_api_key)

advanced_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=MY_API_KEY
)

@tool("Internet_Search")
def search_engine(query: str) -> str:
    """Searches the internet for the latest and most accurate information."""
    try:
        if _tavily_client:
            response = _tavily_client.search(query=query, max_results=4)
            results = response.get("results", [])
        else:
            results = list(DDGS().text(query, max_results=4))
        if not results:
            return "Empty results. Try using a broader or different search query."
        return str(results)
    except Exception as e:
        return f"Error occurred during search: {str(e)}"

# ==========================================
# 2. AGENT CREATION
# ==========================================
# NOTE: allow_delegation is set to True so the Manager can delegate tasks to them!

internet_researcher = Agent(
    role='Chief Internet Researcher',
    goal='Search for the most accurate and up-to-date information online.',
    backstory='You are an expert in web searching. You never guess; you always verify facts using live internet data.',
    verbose=True,
    tools=[search_engine],
    allow_delegation=True, 
    llm=advanced_llm
)

market_analyst = Agent(
    role='Market Analyst',
    goal='Analyze raw data from the internet and extract actionable insights.',
    backstory='You have a highly analytical mind. You flawlessly identify trends, key players, and market obstacles from provided texts.',
    verbose=True,
    allow_delegation=True, 
    llm=advanced_llm
)

report_director = Agent(
    role='Director of Reporting',
    goal='Create a final, professional report in Markdown format.',
    backstory='You are a perfectionist. You transform raw strategies and analyses into beautiful, well-structured, and executive-ready reports.',
    verbose=True,
    allow_delegation=True, 
    llm=advanced_llm
)

# ==========================================
# 3. TASKS & HIERARCHICAL CREW SETUP
# ==========================================

RESEARCH_TOPIC = "The Potential of AI in Waste Management"

# In hierarchical mode, tasks can be broad. The Manager will handle the details and delegation.
main_task = Task(
    description=f'Conduct comprehensive research on the following topic: {RESEARCH_TOPIC}. Search the internet, identify key trends and companies, and then create a professional, structured report.',
    expected_output='A comprehensive, well-formatted business report in Markdown.',
    # No agent is assigned upfront! The Manager will decide who does what.
)

# ASSEMBLE THE CREW WITH A MANAGER
hierarchical_agency = Crew(
    agents=[internet_researcher, market_analyst, report_director],
    tasks=[main_task],
    process=Process.hierarchical, # <--- The AI Manager is enabled here
    manager_llm=advanced_llm,     # <--- Assigning the brain to the Manager
    verbose=True
)

print(f"🚀 Launching the mission under Manager's supervision: {RESEARCH_TOPIC}\n")
final_result = hierarchical_agency.kickoff()

print("\n" + "="*60)
print("🎯 FINAL REPORT APPROVED BY THE MANAGER:")
print("="*60 + "\n")
print(final_result)

# ==========================================
# 4. REQUIREMENTS GENERATOR (Optional Utility)
# ==========================================
# This block automatically generates a clean requirements.txt file for easy installation.
requirements_content = """crewai
ddgs
tavily-python
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content)

print("\n✅ requirements.txt file has been successfully created!")
