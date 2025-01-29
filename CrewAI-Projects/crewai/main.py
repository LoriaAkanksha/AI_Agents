import os
import yaml
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import time
from tenacity import retry, wait_exponential, stop_after_attempt

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
groq_llm = "groq/llama-3.1-70b-versatile"


import warnings
warnings.filterwarnings('ignore')


class SocialMediaPost(BaseModel):
    platform: str = Field(..., description="The social media platform where the post will be published (e.g., Twitter, LinkedIn).")
    content: str = Field(..., description="The content of the social media post, including any hashtags or mentions.")

class ContentOutput(BaseModel):
    article: str = Field(..., description="The article, formatted in markdown.")
    social_media_posts: List[SocialMediaPost] = Field(..., description="A list of social media posts related to the article.")

# Define file paths for YAML configurations
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']


# Creating Agents
market_news_monitor_agent = Agent(
    config=agents_config['market_news_monitor_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()]
    # llm=groq_llm,
)

data_analyst_agent = Agent(
    config=agents_config['data_analyst_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()]
    # llm=groq_llm,
)

content_creator_agent = Agent(
    config=agents_config['content_creator_agent'],
    tools=[SerperDevTool(), WebsiteSearchTool()],
)

quality_assurance_agent = Agent(
    config=agents_config['quality_assurance_agent'],
)


# Creating Tasks
monitor_financial_news_task = Task(
    config=tasks_config['monitor_financial_news'],
    agent=market_news_monitor_agent
)

analyze_market_data_task = Task(
    config=tasks_config['analyze_market_data'],
    agent=data_analyst_agent
)

create_content_task = Task(
    config=tasks_config['create_content'],
    agent=content_creator_agent,
    context=[monitor_financial_news_task, analyze_market_data_task]
)

quality_assurance_task = Task(
    config=tasks_config['quality_assurance'],
    agent=quality_assurance_agent,
    output_pydantic=ContentOutput
)

# Add retry decorator for rate limit handling
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=30),
    stop=stop_after_attempt(3)
)
def execute_with_retry(crew, inputs):
    return crew.kickoff(inputs=inputs)

# Creating Crew
content_creation_crew = Crew(
    agents=[
        market_news_monitor_agent,
        data_analyst_agent,
        content_creator_agent,
        quality_assurance_agent
    ],
    tasks=[
        monitor_financial_news_task,
        analyze_market_data_task,
        create_content_task,
        quality_assurance_task
    ],
    verbose=True
)

# Wrap the crew execution in the retry function
try:
    result = execute_with_retry(content_creation_crew, inputs={
        'subject': 'AI and the impact of AI on human jobs in 2025'
    })
    print("Result: ", result)
except Exception as e:
    print(f"Failed after retries: {str(e)}")

import textwrap

posts = result.pydantic.dict()['social_media_posts']
for post in posts:
    platform = post['platform']
    content = post['content']
    print("Platform: ", platform)
    wrapped_content = textwrap.fill(content, width=50)
    print("Content: ",  wrapped_content)
    print('-' * 50)