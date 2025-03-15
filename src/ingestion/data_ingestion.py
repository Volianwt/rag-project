import os
import re
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def extract_unit_metadata(filename: str) -> str:
    """
    Extracts the unit information from the filename.
    For example, "Unit 3_Cost_and_Time_Estimation (1).pdf" returns "Unit 3".
    """
    match = re.search(r'(Unit\s*\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Unknown"

def load_pdf_documents(data_dir: str):
    """
    Traverse the given directory to load all PDF files using PDFPlumberLoader.
    Then split the text and attach metadata (unit + source filename).
    """
    all_docs = []
    for file in os.listdir(data_dir):
        if file.lower().endswith('.pdf'):
            file_path = os.path.join(data_dir, file)
            print(f"Loading {file_path} with PDFPlumberLoader...")

            # 1) Load the PDF into a list of Document objects (one Document per page by default)
            loader = PDFPlumberLoader(file_path)
            pages = loader.load()  # This returns a list of Document objects

            # 2) If you want to further chunk each page, use a text splitter
            #    Otherwise, you can skip chunking and just attach metadata directly.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # You can adjust
                chunk_overlap=100
            )

            docs = text_splitter.split_documents(pages)

            # 3) Attach metadata: unit info + source filename
            unit = extract_unit_metadata(file)
            for d in docs:
                d.metadata["unit"] = unit
                d.metadata["source"] = file

            # 4) Debug print: show the first 200 chars of each chunk
            print(f"--- After splitting {file}, total chunks = {len(docs)} ---")
            for i, d in enumerate(docs):
                preview = d.page_content[:200].replace("\n", " ")
                print(f"Chunk {i+1} (Unit: {unit}): {preview}...")

            all_docs.extend(docs)

    return all_docs

def build_faiss_index(data_dir: str, persist_directory: str = "./faiss_index"):
    """
    Build a FAISS index from the PDF documents in 'data_dir', using OpenAI embeddings.
    """
    print("Loading documents from:", data_dir)
    docs = load_pdf_documents(data_dir)
    print(f"Total documents/chunks after loading: {len(docs)}")

    # 5) Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(docs, embeddings)
    index.save_local(persist_directory)
    print(f"FAISS index built and saved to: {persist_directory}")

if __name__ == "__main__":
    data_directory = "data/sample_docs"  # or wherever your PDFs are
    build_faiss_index(data_directory)