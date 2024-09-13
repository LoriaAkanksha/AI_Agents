import streamlit as st
from pprint import pprint
from langgraph.graph import END, StateGraph, START  # Ensure START and END are imported
from router import question_router, llm
from tools import wiki
from rag_init import retriever
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from router import question_router, llm
from tools import wiki
from rag_init import retriever
from langchain.schema import Document
# Define the GraphState and GraphNodes as per your logic
class GraphState(TypedDict):
    input: str
    generation: str
    documents: str  # List[str]

class GraphNodes:
    def __init__(self, retriever, wiki):
        self.retriever = retriever
        self.wiki_search = wiki

    def retrieve(self, state):
        """Retrieve documents based on the question."""
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def wiki_search(self, state):
        """Wikipedia search based on the rephrased question."""
        question = state["question"]
        docs = self.wiki_search.invoke({"query": question})
        wiki_results = Document(page_content=docs)
        return {"documents": wiki_results, "question": question}

    def route_question(self, state):
        """Route question to either wiki search or retrieval (RAG)."""
        question = state["question"]
        source = question_router.invoke({"question": question})
        datasource = source.datasource if hasattr(source, 'datasource') else None

        if datasource == "wiki_search":
            return "wiki_search"
        elif datasource == "vectorstore":
            return "retrieve"
        else:
            raise ValueError(f"Unknown datasource: {datasource}")

# Create and compile the graph
workflow = StateGraph(GraphState)
nodes = GraphNodes(retriever=retriever, wiki=wiki)
workflow.add_node("wiki_search", nodes.wiki_search)  # web search
workflow.add_node("retrieve", nodes.retrieve)  # retrieval

# Define conditional routing between the nodes
workflow.add_conditional_edges(
    START,  # Starting node
    nodes.route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

# Compile the workflow
app = workflow.compile()

# Streamlit app to interact with the graph and display the answers
st.title('Welcome to Chatbot with Document Retrieval')

input_text = st.text_input('Ask me anything')

if input_text:
    with st.spinner("Processing..."):
        try:
            inputs = {"question": input_text}
            output_text = ""

            # Stream through the output nodes and display the documents/results
            for output in app.stream(inputs):
                for key, value in output.items():
                    st.write(f"Node '{key}':")
                    pprint(f"Node '{key}':")
                
                output_text = value['documents']
                st.write("\n---\n")
            
            # Final output generation
            if output_text:
                st.write("Final Documents:")
                st.write(output_text)
            else:
                st.write("No documents found.")
        
        except Exception as e:
            st.error(f"Error: {e}")
