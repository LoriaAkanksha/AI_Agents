from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from langserve import add_routes
from langchain.schema import Document

# Your chatbot imports here
from router import question_router
from tools import wiki
from rag_init import retriever
from pipeline import GraphState



# Your chatbot logic here (Define retrieve, wiki_search, route_question, etc.)
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki)  # web search
workflow.add_node("retrieve", retriever)  # retrieve

# Debugging print
print("Adding conditional edges...")
workflow.add_conditional_edges(
    START,
    question_router,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

print('Compiled workflow:')
pprint(workflow.compile())  # Print the compiled workflow

# Define the chatbot
chatbot = workflow.compile()  # Compile the workflow

# Initialize FastAPI
app = FastAPI()

class Input(BaseModel):
    question: str

# Define response model (if needed)
class Output(BaseModel):
    response: str
    
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Add routes
print("Adding routes.........................................................")
#add_routes(app, chatbot, path="/chat")
print("Routes added successfully................................................")
chain = add_routes(app, chatbot, path="/docs/chat")
print("chain created....................................................")

'''@app.post("/ask", response_model=Output)
async def ask_question(input: Input):
    # Use the retriever to fetch the response
    documents = chain.invoke(input.question, ConsistencyLevel="LOCAL_ONE")
    # If no response found, return a default message
    if not documents:
        documents = "No answer found."
    # Return the result as a FastAPI Output model
    return Output(response=documents[0].metadata['description'])'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.10.60.175", port=8017)
