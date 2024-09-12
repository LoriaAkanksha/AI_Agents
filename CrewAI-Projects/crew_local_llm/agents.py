from crewai import Agent
from tools import tool
from dotenv import load_dotenv
load_dotenv()


from langchain_ollama import ChatOllama

import os
os.environ["OPENAI_API_KEY"] = "NA"
os.environ['OPENAI_API_BASE']='http://127.0.0.1:11434'
os.environ['OPENAI_MODEL_NAME']='llama3.1'  # Adjust based on available model

llm = ChatOllama(
    model = "llama3.1",
    base_url = "http://127.0.0.1:11434")

# Creating a senior researcher agent with memory and verbose mode

news_researcher=Agent(
    role="Senior Researcher",
    goal='Unccover ground breaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."

    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True

)

## creating a write agent with custom tools responsible in writing news blog

news_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[tool],
  llm=llm,
  allow_delegation=False
)

