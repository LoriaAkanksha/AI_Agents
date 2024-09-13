from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END  
from fastapi.responses import RedirectResponse
from typing import Literal
from langchain.schema import Document
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# API keys and database credentials from environment variables
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')

# Initialize Cassandra
import cassio
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Embedding using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cassandra VectorStore
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

astra_vector_store.add_documents(doc_splits)

# Retriever setup
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()

# Data model for routing query
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Route the query to the appropriate datasource."
    )

llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="llama-3.1-8b-instant")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Chat prompt template
system_prompt = """You are an expert at routing user questions to either a vectorstore or Wikipedia."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
question_router = route_prompt | structured_llm_router


# State model for graph workflow
class GraphState(BaseModel):
    input: str
    generation: str
    documents: list

# Define the functions for graph nodes
def wiki_search(state):
    question = state["question"]
    docs = wiki.invoke({"query": question})
    return {"documents": [docs], "question": question}

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def route_question(state):
    try:
        question = state["question"]
        source = question_router.invoke({"question": question})
        datasource = source.datasource if hasattr(source, 'datasource') else None

        if datasource == "wiki_search":
            return "wiki_search"
        elif datasource == "vectorstore":
            return "retrieve"
        else:
            raise ValueError(f"Unknown datasource: {datasource}")
    except Exception as e:
        print(f"Error in routing: {e}")
        raise

# Initialize the graph workflow
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node("wiki_search", wiki_search)  # Wiki search node
workflow.add_node("retrieve", retrieve)  # Document retrieval node

# Ensure that the START node is correctly initialized
workflow.add_conditional_edges(
    START,  # The predefined START node should be explicitly mentioned here
    route_question,  # Router function
    {
        "wiki_search": "wiki_search",  # If routed to wiki_search, execute wiki_search node
        "vectorstore": "retrieve",     # If routed to vectorstore, execute retrieve node
    }
)

# Safeguard for checkpoint initialization in _prepare_next_tasks
def _prepare_next_tasks(checkpoint):
    if "versions_seen" not in checkpoint:
        checkpoint["versions_seen"] = {}
    if "__start__" not in checkpoint["versions_seen"]:
        checkpoint["versions_seen"]["__start__"] = 1  # Initialize the START node
    # Proceed with the usual task preparation logic
    # ...

# Add terminal edges leading to the END node
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

# Compile the chatbot workflow
try:
    chatbot = workflow.compile()  # Ensure that START and END nodes are well-connected
except Exception as e:
    print(f"Error during compilation: {e}")

def _prepare_next_tasks(checkpoint):
    if "versions_seen" not in checkpoint:
        checkpoint["versions_seen"] = {}
    if "__start__" not in checkpoint["versions_seen"]:
        checkpoint["versions_seen"]["__start__"] = 1  # Initialize the START node
    # Proceed with the usual task preparation logic


# FastAPI setup
app = FastAPI(title="Chatbot", version="1.0")

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: dict

# Add routes to FastAPI app
add_routes(
    app,
    chatbot.with_types(input_type=Input, output_type=Output),
    path="/chat",
)
@app.post("/invoke-graph/")
async def invoke_graph(input_data: Input):
    try:
        result = chatbot.invoke(input_data.dict())
        return result
    except Exception as e:
        print(f"Error in FastAPI route: {e}")
        return {"error": str(e)}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8013)
