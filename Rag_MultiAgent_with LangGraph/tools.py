from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun


## Arxiv and wikipedia Tools
'''arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=800)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)'''

api_wrapper=WikipediaAPIWrapper(top_k_results=1)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

