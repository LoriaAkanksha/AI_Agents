from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from cassandra.cluster import Cluster
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
### from langchain_cohere import CohereEmbeddings
import cassio
import os
from dotenv import load_dotenv
load_dotenv()

#groq_api_key = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)


# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://www.cloudflare.com/learning/ai/what-is-large-language-model/",
    "https://www.ibm.com/topics/machine-learning"   
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)


embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None

)


astra_vector_store.add_documents(doc_splits)
print("Inserted %i headlines." % len(doc_splits))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

retriever=astra_vector_store.as_retriever()

retriever.invoke("What is agent",ConsistencyLevel="LOCAL_ONE")

rag_result = retriever.invoke("What is agent",ConsistencyLevel="LOCAL_ONE")

print(rag_result)