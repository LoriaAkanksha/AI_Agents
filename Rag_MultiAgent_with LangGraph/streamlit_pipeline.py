import streamlit as st
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from router import question_router, llm
from tools import wiki
from rag_init import retriever
from langchain.schema import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    input: str
    generation: str
    documents: str #List[str]

class GraphNodes:
    def __init__(self, retriever, wiki):
        self.retriever = retriever
        self.wiki_search = wiki
    

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
        
        # Extract datasource string from RouteQuery object
        datasource = source.datasource if hasattr(source, 'datasource') else None
        
        if datasource == "wiki_search":
            print("---ROUTE QUESTION TO Wiki SEARCH---")
            return "wiki_search"
        elif datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            raise ValueError(f"Unknown datasource: {datasource}")
    

    
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
app = workflow.compile()



'''# RUN
inputs = {
    "question": "What is agent?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value['documents'][0].dict()['metadata']['description'])'''

'''# Run
inputs = {
    "question": "Who is modi"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value['documents'])'''

# Streamlit UI
st.title('Question Answering System')

# Input question
question = st.text_input("Enter your question:", "")

if st.button("Submit"):
    if question:
        inputs = {"input": question, "generation": "", "documents": []}
        with st.spinner("Processing..."):
            try:
                # Process the input through the workflow
                results = list(app.stream(inputs))
                
                # Display results
                for result in results:
                    for key, value in result.items():
                        st.write(f"Node '{key}':")
                        documents = value.get("documents", [])
                        if documents:
                            st.write(documents[0].page_content)
                        else:
                            st.write("No results found.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
