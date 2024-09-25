## Graph

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]