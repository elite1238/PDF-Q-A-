# 📄 PDF-Q-A

**Upload any PDF and ask questions!**  

Get instant answers based on the content of your PDF.

---

## 🚀 Features

- Upload PDF files and interact with them using natural language
- Answers are generated based on the actual content of your PDF
- Powered by advanced language models (Llama 3, Phi 3, Gemma 2, Qwen 2.5)
- Runs 100% locally — no cloud API keys or costs required
- Simple, intuitive web interface with Streamlit

---

## 🗂️ Codebase Structure

```
PDF-Q-A-/
├── app.py              # Primary application — clean RAG implementation
├── optimizedapp.py     # Near-identical copy of app.py with a minor UI label change
├── RAGup.py            # Advanced variant using Qdrant cloud vector database
├── week2.py            # Full-featured variant with MongoDB Q&A history and Groq LLM support
├── requirements.txt    # All Python dependencies
├── .devcontainer/
│   └── devcontainer.json   # VS Code Dev Container config (auto-starts app on port 8501)
└── README.md           # This file
```

### File Purposes

| File | Description |
|------|-------------|
| **app.py** | Main entry point. Implements the complete RAG pipeline: PDF upload → text chunking → local embedding → cosine-similarity search → Ollama LLM answer. |
| **optimizedapp.py** | Near-identical copy of `app.py`. The only difference is a minor UI label (`"Answer:"` vs `"*Answer:*"`). |
| **RAGup.py** | Replaces the in-memory vector store with [Qdrant](https://qdrant.tech/) (cloud-hosted). Adds text preprocessing and TextBlob spell correction on user queries. |
| **week2.py** | Extends the pipeline with [MongoDB](https://www.mongodb.com/) for persisting Q&A pairs, semantic de-duplication of previously answered questions, and optional [Groq](https://groq.com/) cloud LLM support. |
| **requirements.txt** | Pinned dependency list for reproducible installs. |

---

## 🔑 Key Technologies

| Technology | Role |
|-----------|------|
| [Streamlit](https://streamlit.io/) | Web UI framework — renders the upload widget, model selector, and chat interface |
| [LangChain](https://python.langchain.com/) | RAG orchestration — text splitting, prompt construction, LLM invocation |
| [Ollama](https://ollama.com/) | Local LLM inference server — runs Llama 3.2, Phi 3.5, Gemma 2, and Qwen 2.5 on your machine |
| [PyPDF2](https://pypdf2.readthedocs.io/) | PDF text extraction |
| [Sentence Transformers](https://sbert.net/) | Generates dense vector embeddings (`all-MiniLM-L6-v2`) for text chunks and queries |
| [scikit-learn](https://scikit-learn.org/) | Cosine similarity computation for nearest-chunk retrieval |
| [ChromaDB](https://www.trychroma.com/) | In-process vector database (used in `app.py` / `optimizedapp.py`) |
| [Qdrant](https://qdrant.tech/) | Cloud-hosted vector database (used in `RAGup.py`) |
| [MongoDB / PyMongo](https://www.mongodb.com/) | Persistent storage for Q&A history (used in `week2.py`) |
| [TextBlob](https://textblob.readthedocs.io/) | Spell correction on user queries (used in `RAGup.py`) |
| [NumPy](https://numpy.org/) | Array operations for embedding similarity |
| [PyTorch](https://pytorch.org/) | Deep learning backend for Sentence Transformers |

---

## 🏗️ Code Architecture

All variants implement the same **Retrieval-Augmented Generation (RAG)** pipeline:

```
┌─────────────┐    ┌───────────────┐    ┌──────────────────┐
│  PDF Upload │───▶│ Text Chunking │───▶│ Embed Chunks     │
│  (PyPDF2)   │    │ (LangChain    │    │ (Sentence        │
└─────────────┘    │  Splitter)    │    │  Transformers)   │
                   └───────────────┘    └────────┬─────────┘
                                                 │ store
                                        ┌────────▼─────────┐
                                        │  Vector Store     │
                                        │  (ChromaDB /      │
                                        │   Qdrant /        │
                                        │   in-memory)      │
                                        └────────┬─────────┘
                                                 │
┌─────────────┐    ┌───────────────┐    ┌────────▼─────────┐
│ User Query  │───▶│ Embed Query   │───▶│ Similarity Search│
└─────────────┘    │ (same model)  │    │ (cosine, top-k)  │
                   └───────────────┘    └────────┬─────────┘
                                                 │ relevant chunks
                                        ┌────────▼─────────┐
                                        │ Prompt Builder   │
                                        │ (LangChain)      │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │   Ollama LLM     │
                                        │ (local inference)│
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │  Answer shown    │
                                        │  in Streamlit UI │
                                        └──────────────────┘
```

### Key Classes & Functions (`app.py`)

```python
class LocalEmbeddings:
    """Wraps SentenceTransformer to produce document and query embeddings."""
    def embed_documents(self, texts) -> np.ndarray
    def embed_query(self, query) -> np.ndarray

class LocalVectorStore:
    """In-memory vector store with cosine-similarity search."""
    def similarity_search(self, query, k=4) -> list[str]

def create_qa_chain(llm, docs, question) -> str:
    """Builds a prompt from retrieved chunks and invokes the Ollama LLM."""

def main():
    """Streamlit entry point — renders UI and wires the pipeline together."""
```

### Session State (Streamlit)

`app.py` uses `st.session_state` to avoid re-processing the same PDF on every interaction:

| Key | Contents |
|-----|----------|
| `pdf_hash` | MD5 of the uploaded file (detects file changes) |
| `embeddings` | `LocalEmbeddings` instance |
| `knowledge_base` | `LocalVectorStore` with all chunk embeddings |
| `chunks` | Raw text chunks from the current PDF |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/elite1238/PDF-Q-A-.git
cd PDF-Q-A-
```

### 2. Install Python Packages

```bash
pip install -r requirements.txt
```

### 3. Install Ollama Locally

Follow the [Ollama installation guide](https://docs.ollama.com/installation) to install Ollama on your system.

### 4. Pull LLM Models

```bash
ollama pull llama3.2
ollama pull phi3.5
ollama pull gemma2:2b
ollama pull qwen2.5:7b
```

Then start the Ollama server (keep this terminal open):

```bash
ollama serve
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open your browser at **http://localhost:8501**.

> **Other variants:**
> ```bash
> python -m streamlit run optimizedapp.py   # performance-tuned
> python -m streamlit run RAGup.py          # Qdrant cloud vector DB
> python -m streamlit run week2.py          # MongoDB + Groq LLM support
> ```

---

## ❓ How to Use

1. Open the PDF-Q-A web app in your browser.
2. Select an LLM model from the dropdown.
3. Upload your PDF file using the upload button.
4. Ask questions about the content of the PDF using the input box.
5. Read the answers generated by the AI model.

---

## 📚 FAQ

**Q: What types of PDF files can I upload?**  
A: You can upload any PDF file that is not password-protected. The PDF should contain text-based content for the best results.

**Q: How does the AI generate answers?**  
A: The app uses Retrieval-Augmented Generation (RAG): it splits the PDF into chunks, embeds them as vectors, finds the chunks most relevant to your question, and passes them as context to the local LLM to generate an answer.

**Q: Can I trust the accuracy of the answers?**  
A: The answers are generated by advanced AI models, but they may not always be 100% accurate. It's always good to double-check important information.

**Q: Do I need an internet connection or API key?**  
A: No. All inference runs locally via Ollama. No data leaves your machine.

---

## 🎉 Acknowledgments

- This project is powered by [Streamlit](https://streamlit.io/), [Ollama](https://ollama.com/), and advanced language models from [Hugging Face](https://huggingface.co/).
- Special thanks to the contributors and the open-source community.

---

Enjoy exploring your PDFs in a whole new way! 🚀
