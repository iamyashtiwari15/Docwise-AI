# docwise-ai

A modular document-first AI assistant. Upload PDF/DOCX files and ask grounded questions. Automatically falls back to web search when the answer is not in your documents.

## Features

- PDF, DOCX, TXT ingestion with per-session retrieval
- 3-layer routing: document Q&A → web fallback → general chat
- Confidence-based routing (configurable threshold)
- Image analysis / OCR support
- Session-based conversation history
- Modular FastAPI backend (api / services / agents)
- React + Vite frontend

## Architecture

```
backend/
  api/routes/chat.py        ← /api/chat  /api/upload  /api/history  /api/health
  core/config.py            ← env-based centralised config
  services/
    chat_service.py         ← request orchestration
    session_store.py        ← bounded in-memory session store (Redis-ready)
  agents/
    agent_decision.py       ← LangGraph pipeline
    workflow_manager.py     ← document / web / general routing
    query_router.py         ← fast rule-based classification
    uploaded_document_store.py  ← per-session doc indexing + retrieval
    document_ingestion.py   ← PDF/DOCX/TXT extraction + chunking
    general_chat_agent.py   ← general conversation handler
    guardrails.py           ← fast local safety filter
    rag_agent/
      response_generator.py ← grounded answer synthesis
  tests/

frontend/src/
  App.jsx / main.jsx / styles.css
```

## Getting started

### Backend

```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload
```

Create `.env`:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Optional env vars:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq chat model |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SESSION_MAX_MESSAGES` | `30` | Max messages per session |
| `DOCUMENT_RELEVANCE_THRESHOLD` | `0.35` | Min cosine similarity to use doc answer |
| `WEB_SEARCH_MAX_RESULTS` | `3` | Tavily results per search |
| `RAG_TOP_K` | `4` | Final number of document chunks passed to generation |
| `RAG_CANDIDATE_K` | `8` | Retrieval candidate pool size before reranking |
| `RAG_MAX_QUERY_VARIANTS` | `3` | Max multi-query variants per request |
| `DOCUMENT_CHUNK_SIZE` | `900` | Target chunk size for uploaded documents |
| `DOCUMENT_CHUNK_OVERLAP` | `150` | Overlap budget when building document chunks |
| `ENABLE_MULTI_QUERY_RETRIEVAL` | `true` | Enable low-cost multi-query retrieval for compound questions |
| `ENABLE_LLM_MULTI_QUERY` | `true` | Use the LLM to generate retrieval variants for complex questions |
| `ENABLE_LIGHTWEIGHT_RERANK` | `true` | Blend semantic and lexical signals before generation |
| `LLM_MULTI_QUERY_MIN_TERMS` | `5` | Minimum extracted query terms before LLM multi-query is attempted |
| `CORS_ORIGINS` | localhost dev ports | Comma-separated allowed origins |

### Frontend

```bash
cd frontend
npm install
npm run dev   # proxies /api to http://127.0.0.1:8000
```

## Tests

```bash
python -m unittest discover -s backend/tests
```
