from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from graph import *
from rag import retriever
from tools import *
from router import *
from pprint import pprint

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

### Edges ###


def route_question(state):
    """
    Route question to wiki search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


## WORKFLOW

workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)
# Compile
chatbot = workflow.compile()

# Run
inputs = {
    "question": "who is pm of india?"
}
final_output = None
source_type = None

for output in chatbot.stream(inputs):
    for key, value in output.items():
        # Store the final output and source type based on routing
        if key == "retrieve":
            source_type = "RAG"
        elif key == "wiki_search":
            source_type = "wiki_search"
        final_output = value

# Final generation
if source_type == "RAG":
    # Print metadata description if routed to vectorstore (RAG)
    print("--- Final Generation from RAG ---")
    pprint(final_output['documents'][0].dict()['metadata']['description'])
elif source_type == "wiki_search":
    # Print full document results if routed to Wikipedia search
    print("--- Final Generation from Wikipedia Search ---")
    pprint(final_output['documents'])
