# 🔍 RAG API — CV Matching System

A production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI, FAISS, and OpenRouter LLM. Upload your documents (CVs, reports, manuals) and ask questions — the system retrieves the most relevant chunks and returns accurate, context-grounded answers.

---

## ✨ Features

- 📄 **Multi-format document ingestion** — PDF, DOCX, HTML, TXT
- 🧠 **Semantic search** using `all-MiniLM-L6-v2` sentence embeddings
- ⚡ **FAISS vector store** for fast similarity search
- 💬 **Chat with memory** — maintains session-based conversation history
- 🤖 **LLM generation** via OpenRouter API
- 🐳 **Fully containerized** with Docker & Docker Compose
- 🔁 **Auto-indexing** on startup if documents exist in `/data`

---

## 🏗️ Architecture

```
Rag_project/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── routes/
│   │   └── rag_routes.py    # API endpoints (Controller)
│   └── services/
│       └── rag_service.py   # Core RAG logic (Embeddings, FAISS, LLM)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

**Flow:**
```
Upload Docs → Extract Text → Chunk → Embed → Store in FAISS
Query → Embed Query → FAISS Search → Build Prompt → LLM → Answer
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) & Docker Compose
- An [OpenRouter](https://openrouter.ai/) API key

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Rag_project.git
cd Rag_project
```

### 2. Set your API key

Open `docker-compose.yml` and replace the placeholder with your key:

```yaml
environment:
  - OPENROUTER_API_KEY=your_api_key_here
```

> ⚠️ **Never commit real API keys.** See [Security](#-security) below.

### 3. (Optional) Pre-load documents

Drop any PDF, DOCX, HTML, or TXT files into a `./data` folder — they'll be indexed automatically on startup:

```bash
mkdir data
cp your_cvs/*.pdf data/
```

### 4. Build and run

```bash
docker-compose up --build
```

The API will be live at `http://localhost:8000`

---

## 📡 API Endpoints

### `GET /`
Health check — returns server status and number of indexed vectors.

```json
{ "status": "ok", "index_size": 142 }
```

---

### `POST /upload`
Upload and index one or more documents.

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@cv1.pdf" \
  -F "files=@cv2.docx"
```

**Response:**
```json
{ "message": "Files uploaded and indexed successfully", "files_processed": 2 }
```

---

### `POST /query`
Ask a question against the indexed documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{ "query": "Who has experience with Python and machine learning?", "top_k": 5, "session_id": "user_1" }'
```

**Response:**
```json
{
  "query": "Who has experience with Python and machine learning?",
  "answer": "Based on the provided CVs, Ahmed Hassan has 3 years of Python experience...",
  "full_prompt": "..."
}
```

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Your question |
| `top_k` | int | 5 | Number of document chunks to retrieve |
| `session_id` | string | `"default"` | Session ID for chat history |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Store | FAISS (CPU) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | OpenRouter API |
| PDF Parsing | pdfplumber |
| DOCX Parsing | python-docx |
| HTML Parsing | BeautifulSoup4 |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Containerization | Docker + Docker Compose |

---

## 🔒 Security

The `docker-compose.yml` currently has a hardcoded API key — **remove it before pushing to GitHub.**

Use a `.env` file instead:

1. Create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_real_key_here
   ```

2. Update `docker-compose.yml`:
   ```yaml
   environment:
     - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
   ```

3. Add `.env` to `.gitignore`:
   ```
   .env
   ```

---

## 📋 Supported File Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word Document | `.docx` |
| HTML | `.html` |
| Plain Text | `.txt` |

---

## 📄 License

This project is open source. Feel free to use and modify.
