from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from graph import *
from rag import retriever
from tools import *
from router import *
from pprint import pprint
from langserve import add_routes
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
import uvicorn

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

# FastAPI setup
app = FastAPI(title="Chatbot", version="1.0")


# Pydantic models for input and output validation
class Input(BaseModel):
    question: str

class Output(BaseModel):
    documents: dict

# Add route for root redirect to Swagger UI
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# add routes
add_routes(
   app,
   chatbot.with_types(input_type=Input, output_type=Output),
   path="/chat",
)

@app.post("/chat", response_model=Output)
async def chat(input: Input):
    inputs = {
        "question": input.question
    }

    final_output = None
    source_type = None

    # Run the workflow for the provided input
    for output in chatbot.stream(inputs):
        for key, value in output.items():
            # Store the final output and source type based on routing
            if key == "retrieve":
                source_type = "RAG"
            elif key == "wiki_search":
                source_type = "wiki_search"
            final_output = value

    # Final response based on the source type
    if source_type == "RAG":
    # Check if 'documents' exist and is not empty
        if 'documents' in final_output and final_output['documents']:
            # Safely access the metadata description
            metadata_description = final_output['documents'][0].dict().get('metadata', {}).get('description', None)
            
            if metadata_description is None:
                # If description is not found, return a suitable error message or default response
                raise HTTPException(status_code=500, detail="No description found in metadata")
            
            # Return a valid dictionary with the description
            return {"documents": {"description": metadata_description}}
        else:
            # Handle case where 'documents' key is missing or empty
            raise HTTPException(status_code=500, detail="No documents found in final output")
    elif source_type == "wiki_search":
        # Return full document results from Wikipedia search
        return {"documents": final_output['documents']}
    else:
        raise HTTPException(status_code=404, detail="No documents found")


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=7003)
