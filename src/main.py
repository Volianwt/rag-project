import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.retrieval.doc_retrieve import retrieve_docs

def generate_answer(full_query: str, retrieved_docs, num_questions: int = 3):
    """
    Use retrieved document chunks to generate an answer with quiz questions.
    """
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Construct a detailed prompt for the LLM
    prompt = f"""
    You are a helpful teaching assistant. Based on the following context, please explain the concept and generate {num_questions} multiple-choice quiz questions.
    
    Question: {full_query}

    Context:
    {context}
    """
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if __name__ == "__main__":
    load_dotenv()  # Load environment variables

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is not set.")

    # **STEP 1: Define Queries**
    full_query = "What is risk management?"
    retrieval_query = full_query.split("?")[0] 

    # **STEP 2: Retrieve Relevant Documents**
    retrieved_docs = retrieve_docs(retrieval_query, k=10)

    # **STEP 3: Use LLM to Generate Answers**
    answer = generate_answer(full_query, retrieved_docs, num_questions=3)

    print("\n=== Generated Answer and Quiz ===")
    print(answer)