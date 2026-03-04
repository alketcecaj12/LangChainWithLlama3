# 📄 Simple RAG with LangChain

A beginner-friendly **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain. Ask questions about any PDF and get answers powered by a local or cloud LLM.

---

## 🧠 How It Works

```
PDF
 │
 ▼
PyPDFLoader          ← loads the PDF into Document objects
 │
 ▼
RecursiveCharacterTextSplitter   ← splits into overlapping chunks
 │
 ▼
OllamaEmbeddings / OpenAIEmbeddings   ← converts chunks into vectors
 │
 ▼
Chroma Vector Store  ← stores and indexes the vectors
 │
 ▼
User Query → similarity search → relevant chunks → LLM → Answer
```

The key idea behind RAG is that instead of feeding an entire document to an LLM (which has a context limit), we:
1. Pre-process the document into searchable vector chunks
2. At query time, retrieve only the **most relevant chunks**
3. Pass those chunks + the question to the LLM for an answer

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a conda environment
```bash
conda create -n rag_env python=3.11
conda activate rag_env
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-openai langchain-chroma langchain-ollama langchain-text-splitters python-dotenv
```

### 4. Set up your API key (if using OpenAI)
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 5. (Optional) Use Ollama for free local embeddings
Install Ollama from [ollama.com](https://ollama.com) and pull a model:
```bash
ollama pull llama3
```

---

## 🚀 Example Usage

```python
# Load and process the PDF
loader = PyPDFLoader('rag-paper.pdf')
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Embed and store (local with Ollama)
from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(model="llama3")

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)

# Query
results = vector_store.similarity_search("What is RAG?", k=3)
for r in results:
    print(r.page_content)
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| [LangChain](https://langchain.com) | RAG pipeline framework |
| [Chroma](https://trychroma.com) | Vector database |
| [Ollama](https://ollama.com) | Local LLM & embeddings |
| [OpenAI](https://openai.com) | Cloud embeddings (optional) |
| Python 3.11 | Runtime |

---

## 📁 Project Structure

```
.
├── SimpleRAG.py          # Main RAG pipeline
├── rag-paper.pdf         # Source document
├── .env                  # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 📚 What I Learned

- How to load and chunk documents for LLM consumption
- The difference between `split_text()` and `split_documents()`
- How vector embeddings enable semantic search
- Managing API keys securely with `.env`
- Switching between local (Ollama) and cloud (OpenAI) embeddings

---

## ⚠️ Notes

- Never commit your `.env` file — it's in `.gitignore`
- Delete the `chroma_db/` folder if you switch embedding models (dimension mismatch)
- OpenAI embeddings = 1536 dims | Llama3 embeddings = 4096 dims — they are not interchangeable
