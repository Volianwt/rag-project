# 📌 RAG-Based Document Retrieval and Q&A System

## 🔍 Overview
This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using LangChain, FAISS, and OpenAI's GPT model. It allows users to **retrieve relevant information** from a collection of documents and generate **quiz questions** for learning and assessment.(If your **professor provides a large number of slides** and you’re struggling with **how to efficiently review**, or if you **want to reinforce your learning** by generating **relevant quiz questions**, but can’t find **questions tailored to your slides**, this project is the solution.)

## 🚀 Features
- **Document Ingestion & Indexing**: Parses PDF files and indexes them using FAISS.
- **Semantic Search (Retrieval)**: Retrieves the most relevant document chunks using vector embeddings.
- **Question Generation**: Uses GPT-3.5/4 to generate multiple-choice questions based on retrieved content.
- **Tracing with LangSmith**: Tracks queries and responses for debugging & evaluation.

---

## 🛠 Installation

### 1️⃣ **Clone the repository**
```bash
 git clone https://github.com/Volianwt/rag-project.git
 cd rag-project
```

### 2️⃣ **Set up a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### 3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Set up environment variables**
Create a `.env` file and add your OpenAI & LangSmith API keys:
```ini
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
```

---

## 📂 Project Structure
```
rag-project/
│── data/sample_docs/          # Directory for PDF files to be indexed
│── faiss_index/               # FAISS vector store
│── src/
│   ├── indexing/
│   │   ├── build_index.py     # Parses PDFs and builds FAISS index
│   ├── retrieval/
│   │   ├── doc_retrieve.py    # Retrieves relevant document chunks
│   ├── main.py                # Orchestrates retrieval and Q&A generation
│── requirements.txt           # Dependencies
│── .env                       # API keys (not included in repo)
│── README.md                  # This file
```

---

## 📌 Usage Guide

### **1️⃣ Build the FAISS Index** (Run once to process documents)
```bash
python src/indexing/build_index.py
```
This will parse the PDFs in `data/sample_docs/` and create an index.

### **2️⃣ Retrieve & Generate Questions**
```bash
python src/main.py
```
This will:
1. Retrieve **top-k relevant document chunks** based on the query.
2. Generate **multiple-choice quiz questions** using GPT.
3. Display the output in the terminal.

#### **Example Input (inside `main.py`)**
```python
full_query = "What is risk management?"
```

#### **Example Output**
```
=== Generated Answer and Quiz ===
1. What is the primary goal of risk management?
   A) Minimize profits
   B) Reduce uncertainties in projects
   C) Eliminate all risks
   D) Maximize cost overruns
   Correct Answer: B
```

---

## 📊 LangSmith Tracing (Optional)
This project integrates **LangSmith** to trace execution.

To check logs, visit **[smith.langchain.com](https://smith.langchain.com)** and ensure:
- `LANGSMITH_API_KEY` is correctly set.
- Queries appear in the dashboard.

---

## 🔧 Future Enhancements
- [ ] Add support for **additional file formats** (Word, CSV, TXT)
- [ ] Implement **UI for better user interaction**
- [ ] Optimize FAISS indexing for **faster retrieval**
- [ ] Add **more evaluation metrics** for quiz quality

---


## 🤝 Contributions
Pull requests are welcome! If you find issues, open an **issue** or contribute directly.

🚀 **Happy coding!**

