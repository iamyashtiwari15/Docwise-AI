# DocWise AI

DocWise AI is a document-first assistant with a FastAPI backend, a LangGraph orchestration layer, and a React/Vite frontend. It supports session-scoped document Q&A for PDF, DOCX, and TXT files, can analyze uploaded images, and falls back to Tavily web search when document retrieval is weak or when a question needs live information.

## Documentation

- Full end-to-end project guide: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
- RAG deep dive: [RAG_ALGORITHM_DETAILS.md](RAG_ALGORITHM_DETAILS.md)

## Key Features

- Session-scoped upload and retrieval for PDF, DOCX, and TXT
- Hybrid RAG with dense embeddings, BM25, reciprocal rank fusion, cross-encoder reranking, and HyDE
- LangGraph orchestration for guardrails, image handling, routing, and response delivery
- Web-search fallback for current or low-confidence document questions
- OCR-based image analysis for uploaded user images
- Conversation history restored per browser session
- Chunk-debug artifacts and evaluation utilities for RAG quality work

## Tech Stack

- Backend: Python, FastAPI, Pydantic, LangChain, LangGraph, Groq, Tavily, HuggingFace Embeddings, sentence-transformers, rank-bm25, PyMuPDF4LLM, python-docx, Pillow, pytesseract
- Frontend: React 18, Vite 5, react-markdown, plain CSS
- State: browser `localStorage`, LangGraph `InMemorySaver`, in-memory uploaded document store
- Testing and evaluation: `unittest`, FastAPI `TestClient`, custom multi-layer RAG evaluator

## Runtime Architecture

```text
backend/
  main.py                             FastAPI app factory + CORS
  api/routes/chat.py                  HTTP routes
  api/schemas.py                      request/response models
  services/chat_service.py            request orchestration entry point
  agents/agent_decision.py            LangGraph workflow
  agents/workflow_manager.py          document/web/general routing policy
  agents/uploaded_document_store.py   in-memory RAG index + retrieval
  agents/document_ingestion.py        extraction + chunking
  agents/rag_agent/response_generator.py
  agents/web_search/tavily_search.py
  agents/vision_agents/image_analysis_agent.py

frontend/
  src/App.jsx                         single-page UI and API client
  src/main.jsx                        React bootstrap
  src/styles.css                      UI styling
```

## API Routes

- `GET /api/health` -> liveness check
- `POST /api/history` -> restore chat history and uploaded files for a session
- `POST /api/ingest` -> background document indexing as soon as a file is selected
- `POST /api/chat` -> text-only requests
- `POST /api/upload` -> file upload flow for documents and images

## Quick Start

### Backend

```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Required `.env`

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

## Important Runtime Defaults

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Main chat and response model |
| `EMBEDDING_MODEL_NAME` | `microsoft/harrier-oss-v1-270m` | Default embedding model |
| `DOCUMENT_RELEVANCE_THRESHOLD` | `0.20` | Minimum document score before web fallback |
| `RAG_TOP_K` | `6` | Chunks passed into answer generation |
| `RAG_CANDIDATE_K` | `50` | Candidate pool before reranking |
| `DOCUMENT_CHUNK_SIZE` | `1200` | Target chunk length |
| `DOCUMENT_CHUNK_OVERLAP` | `300` | Overlap between adjacent chunks |
| `RAG_WINDOW_SIZE` | `2` | Neighbor chunk expansion around retrieved chunks |

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the complete configuration table and end-to-end logic.

## Tests

```bash
cd backend
python -m unittest discover -s tests
```
