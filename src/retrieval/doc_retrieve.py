import os
from dotenv import load_dotenv
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def load_faiss_index(index_path="./faiss_index"):
    """
    Load FAISS index and enable tracing.
    """
    embeddings = OpenAIEmbeddings()  # Attach callback to embeddings
    index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return index

def retrieve_docs(query: str, k: int = 5):
    """
    Retrieve the top-k document chunks based on semantic similarity to the query.
    """
    index = load_faiss_index()
    
    # Use a Runnable to trace retrieval
    retrieve_fn = RunnableLambda(lambda x: index.similarity_search(x, k=k))

    # Run and trace the retrieval
    docs = retrieve_fn.invoke(query) 
    return docs
