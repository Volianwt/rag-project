# build_index.py
import os
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def parse_pdf_to_docs(file_path, chunk_size=1500, chunk_overlap=200):
    """
    Parse a single PDF file into a list of Document objects, each representing
    a chunk of text. We use pdfplumber + RecursiveCharacterTextSplitter.
    """
    docs = []
    with pdfplumber.open(file_path) as pdf:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        for page_idx, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                continue
            # Wrap this page's text as one Document
            page_doc = Document(
                page_content=raw_text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page_number": page_idx + 1
                }
            )
            # Further split if the page is large
            splitted = text_splitter.split_documents([page_doc])
            docs.extend(splitted)
    return docs

def build_faiss_index(data_dir: str = "data/sample_docs", persist_dir: str = "./faiss_index"):
    """
    1) Parse all PDFs in data_dir into Document chunks
    2) Use OpenAIEmbeddings to create vector embeddings
    3) Build FAISS index and save locally
    """
    all_docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, fname)
            print(f"Parsing {file_path} ...")
            docs = parse_pdf_to_docs(file_path)
            all_docs.extend(docs)
    print(f"Total chunks after parsing: {len(all_docs)}")

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(all_docs, embeddings)
    index.save_local(persist_dir)
    print(f"FAISS index saved to {persist_dir}")

if __name__ == "__main__":
    build_faiss_index()