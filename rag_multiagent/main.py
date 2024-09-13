import os
import cassio
from fastapi import FastAPI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Environment Variables
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
groq_api_key = os.getenv('GROQ_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# Initialize Astra DB Connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Docs to Index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load Documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split Documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)
print("doc split done.......................................")
# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print('embedding done ............................')
# Vector Store
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

# Add Documents to Vector Store
astra_vector_store.add_documents(doc_splits)
print(f"Inserted {len(doc_splits)} documents.")

# Create Index and Retriever
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_store.as_retriever()

'''demo = retriever.invoke("What is agent",ConsistencyLevel="LOCAL_ONE")
print(demo[0].metadata['description'])'''


# FastAPI app
app = FastAPI(
    title="Conversational Chatbot API",
    version="1.0",
    description="An API for answering questions using a vector store and LLM."
)

class Input(BaseModel):
    question: str

class Output(BaseModel):
    response: str


@app.post("/ask", response_model=Output)
async def ask_question(input: Input):
    # Use the retriever to fetch the response
    documents = retriever.invoke(input.question, ConsistencyLevel="LOCAL_ONE")
    # If no response found, return a default message
    if not documents:
        documents = "No answer found."
    # Return the result as a FastAPI Output model
    return Output(response=documents[0].metadata['description'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.60.175", port=5000)

