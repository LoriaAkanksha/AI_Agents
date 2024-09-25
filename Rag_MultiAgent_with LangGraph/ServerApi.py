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

    print("---RETRIEVE---")
    question = state["question"]
    print(question)

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):

    print("---wikipedia---")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})

    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}


def route_question(state):
    """
    Route question to wiki search or RAG.
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
            # Access the metadata description
            metadata_description = final_output['documents'][0].dict().get('metadata', {}).get('description', None)

            # Return a valid dictionary with the description
            return {"documents": {"description": metadata_description}}
        else:
            raise HTTPException(status_code=500, detail="No documents found in final output")
    elif source_type == "wiki_search":
        # Return document results from Wikipedia search
        return {"documents": final_output['documents']}
    else:
        raise HTTPException(status_code=404, detail="No documents found")


# FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=7001)
