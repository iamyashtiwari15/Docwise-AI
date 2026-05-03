# RAG Algorithm Details

This document describes the exact Retrieval-Augmented Generation (RAG) implementation currently used in this project.

## 0) Design Intent

This RAG stack is optimized for dynamic, session-scoped uploads rather than a static enterprise corpus. The design goals are:

- Fast indexing per uploaded file
- Strong retrieval precision through multi-stage reranking
- Conversation-aware query rewriting for follow-up questions
- Safe fallback to web when document confidence is low
- Fully in-memory operation by default

## 1) Orchestration ("Orchestra")

The system is orchestrated with a LangGraph workflow and a service/API layer.

### End-to-end flow

1. Request enters FastAPI (`/api/chat`, `/api/upload`, `/api/ingest`).
2. `ChatService` invokes `assistant_graph` (LangGraph) with `session_id` and current input.
3. Graph runs these stages:
   - `Guardrails` (safety check for text input)
   - `ImageDetection` (route based on image presence)
   - `ImageAnalysis` (if image present; OCR/vision context extraction)
   - `ContextMerger` (merge text + image context)
   - `RouteAgent` (document vs web vs general)
   - `CallAgent` (execute selected branch and return response)
4. For document route, `WorkflowManager` calls document retrieval in `UploadedDocumentStore` and then answer generation.
5. If document confidence is too low, workflow falls back to web search.

### Graph node-level behavior

- `Guardrails`
  - Triggered only for text input.
  - If blocked, graph returns immediately with `GUARDRAILS_BLOCK` response.
- `ImageDetection`
  - If `image` is present in state -> route to `ImageAnalysis`.
  - Else -> skip image analysis and continue.
- `ImageAnalysis`
  - Runs OCR/vision analysis and produces `image_context`.
  - On failure, stores an error marker text (`❌ ...`) used downstream.
- `ContextMerger`
  - If valid `image_context` exists, merges it with user query.
- `RouteAgent`
  - Calls `WorkflowManager.process_query(...)`.
  - Maps result route to `DOCUMENT_AGENT`, `WEB_AGENT`, or `CONVERSATION_AGENT`.
- `CallAgent`
  - For conversation route, calls `general_chat_agent`.
  - For document/web routes, returns workflow response and appends AI message to graph history.

### Routing logic

- `QueryRouter` classifies into:
  - `document`
  - `web`
  - `general`

Detailed routing order (important):

1. Empty query -> `general`
2. Match any general pattern (greetings/meta-chat) -> `general`
3. Match any document pattern (`document`, `pdf`, `upload`, `summar...`, `analyze`, etc.) -> `document`
4. Match any web pattern (`latest`, `today`, `2024/2025/2026`, `price`, etc.) -> `web`
5. Else, if session has uploaded docs -> `document`
6. Else -> `web`

This means document keywords are checked before web keywords.

## 1.1) Query Rewrite / Merge (Before Routing)

`AdvancedQueryMerger.merge(...)` runs before routing and retrieval.

### Trigger rules

- If no chat history -> no rewrite.
- If query has no follow-up marker -> no rewrite.

Follow-up markers checked:

- `this`, `that`, `it`, `they`, `them`, `these`, `those`, `above`, `earlier`, `previous`

### Rewrite branches

1. If follow-up marker exists and query length <= 6 words:
  - Returns:
    - `current_query + "\n\nRelated context: " + last_user_message`
2. If follow-up marker exists and query length > 6 words:
  - Calls LLM role `query_rewriter` with last `max_history` turns (default 4)
  - Returns rewritten self-contained query
3. If rewrite fails or empty output:
  - Falls back to original `current_query`

## 2) Ingestion Pipeline

Supported document types:

- PDF
- DOCX
- TXT

### Extraction

- PDF: `pymupdf4llm` + `fitz` (PyMuPDF)
  - Page text is extracted in markdown-like structure.
  - Page markers are injected as `[PAGE N]`.
  - Embedded images are extracted and optionally described by a Groq vision model.
  - Image descriptions are injected as `[IMAGE SUMMARY: ...]` blocks.
- DOCX: `python-docx`
- TXT: UTF-8 decode with ignore errors

### Chunking strategy

Chunking is not naive fixed-size splitting. It is structure-aware:

- Normal text is split with sentence-aware logic.
- Overlap is applied between adjacent chunks.
- Markdown tables are atomic (never split).
- `[IMAGE SUMMARY: ...]` blocks are atomic (never split).
- Metadata includes:
  - `source`
  - `source_path`
  - `chunk_index`
  - `page_number` (when detectable from `[PAGE N]`)
  - flags like `is_table`, `is_image_summary`

Default chunk settings:

- `document_chunk_size = 1200`
- `document_chunk_overlap = 300`

### Exact chunking mechanics

1. Normalize text:
  - Collapse repeated spaces
  - Preserve table lines and special markers
  - Collapse 3+ newlines into double newline
2. Split into paragraph blocks (double-newline boundaries)
3. For each block:
  - If markdown table -> atomic unit
  - If `[IMAGE SUMMARY: ...]` -> atomic unit
  - Else split via sentence-aware `_split_long_unit(...)`
4. Build chunks with max length budget:
  - When current chunk would exceed `chunk_size`, flush current chunk
  - Carry overlap units up to `document_chunk_overlap`
5. Metadata extraction:
  - `page_number` is parsed via regex from `[PAGE N]`

Debug output:

- Every ingestion attempts `save_chunks_debug(...)`
- Writes human-readable chunk dump at `backend/chunk_debug/.../chunks.txt`

## 3) Embedding Model Used

Embeddings are generated through `langchain_huggingface` with normalization enabled.

### Default embedding model (from config)

- `microsoft/harrier-oss-v1-270m`
- Device default: `cpu`

### Query-time behavior

- For Harrier models: query embeddings use `prompt_name="web_search_query"`.
- For BGE models (if configured): query instruction prefix is applied:
  - `Represent this sentence for searching relevant passages: `

### Normalization and vector behavior

- `encode_kwargs={"normalize_embeddings": True}` is enabled.
- Because vectors are normalized, cosine similarity becomes stable for ranking.
- Query embeddings and document embeddings are both stored/used as float lists.

## 4) Retrieval Algorithm

The retrieval stack is hybrid and multi-stage.

### Stage A: Query expansion

- Base query is always used.
- Optional multi-query generation:
  - Heuristic splitting for compound queries.
  - LLM-generated variants for sufficiently complex queries.

### Exact multi-query trigger gates

Global gate in retrieval:

1. `enable_multi_query_retrieval` must be `true`
2. If false: only base query is used

LLM gate (in addition to global gate):

1. `enable_llm_multi_query` must be `true`
2. `should_use_llm_multi_query(query, min_terms)` must be `true`
3. `min_terms` comes from `llm_multi_query_min_terms` (default 5)

`should_use_llm_multi_query(...)` returns `true` if any are true:

- extracted term count >= `min_terms`
- query contains compound connectors: `and|or|also|plus|vs|versus`
- query contains any of `? ; \n :`

If LLM multi-query is enabled and triggered:

- role `multi_query` model is called
- prompt requests up to `max_variants - 1` additional short retrieval queries
- output parser accepts JSON list or newline list
- candidates are filtered by:
  - remove question prefix (`what is`, `tell me about`, etc.)
  - deduplicate by lowercase string
  - lexical overlap guard: must share at least one term with base query
  - hard cap at `rag_max_query_variants`

Heuristic fallback (always attempted after LLM attempt unless early return):

- If base terms < 4 and no `? ; \n` -> skip heuristic generation
- Else candidate sources:
  - split on `? ; \n`
  - if compound query and terms >= 4, split on `and|or|also|plus|vs|versus`
  - if terms >= 6, add compressed term query (`first up to 8 terms`)
- Candidate filters:
  - dedupe
  - require at least 2 extracted terms OR at least 3 words
  - require lexical overlap with base query
  - cap at `rag_max_query_variants`

Default toggles:

- `enable_multi_query_retrieval = true`
- `enable_llm_multi_query = true`
- `llm_multi_query_min_terms = 5`
- `rag_max_query_variants = 5`

### Stage B: HyDE (Hypothetical Document Embeddings)

- If enabled, an LLM writes a synthetic factual passage for the user query.
- That synthetic passage is embedded and added as an extra dense retrieval query vector.

HyDE trigger gate:

1. `enable_hyde` must be `true`
2. HyDE LLM call must succeed and return non-empty passage
3. If failure/empty output -> HyDE is skipped, retrieval continues normally

Default toggle:

- `enable_hyde = true`

### Stage C: Dense retrieval

- Each query variant (and HyDE vector if present) is embedded.
- Cosine similarity is computed against all stored chunk embeddings.

Dense ranking behavior:

- A full ranking is produced per query embedding.
- Each ranking contributes top `candidate_k` items to fusion.

### Stage D: Sparse retrieval (BM25)

- BM25 index is built from tokenized chunk terms (`rank_bm25`).
- Each query variant contributes a BM25 ranking.

BM25 trigger and fallback:

- BM25 index is lazily rebuilt when session docs change.
- If `rank_bm25` import fails, BM25 stage is skipped (dense-only still works).

### Stage E: Fusion (RRF)

- Dense rankings + BM25 rankings are fused using Reciprocal Rank Fusion:
  - score contribution: `1 / (k + rank + 1)` with `k = 60`

For each candidate chunk index `i`:

$$
RRF(i) = \sum_{r \in rankings} \frac{1}{60 + rank_r(i) + 1}
$$

- Candidates are sorted by fused RRF score.
- Top `candidate_k` proceed to reranking.

### Stage F: Cross-encoder rerank

- Top fused candidates are reranked using cross-encoder logits, sigmoid-normalized.

Exact behavior:

1. Trigger only if `enable_cross_encoder_rerank = true` and candidate list non-empty
2. Build pairs: `(original_user_query, candidate_chunk_content)`
3. Predict logits via cross-encoder
4. Convert each logit to score using sigmoid
5. Reorder candidates by score descending
6. If any exception occurs -> keep prior RRF order

Default model:

- `cross-encoder/ms-marco-MiniLM-L-12-v2`

Default toggle:

- `enable_cross_encoder_rerank = true`

### Stage G: Top-k selection + context window

- Final top-k chunks selected.
- Optional neighboring chunk expansion to improve local context continuity.

Detailed rules:

- `resolved_candidate_k = max(rag_candidate_k, rag_top_k * 4)`
- `combined_score` is currently:
  - cross-encoder score when available
  - else RRF score
- `score` field stores best semantic cosine among query embeddings
- If `rag_window_size > 0`, retrieved chunk text is expanded with neighboring chunk contents from same source using `chunk_index` adjacency

Defaults:

- `rag_candidate_k = 50`
- `rag_top_k = 6`
- `rag_window_size = 2`

## 5) Generation

- Retrieved chunks are formatted into `<passage ...>` blocks with relevance labels and metadata.
- Role-specific LLM prompt (`rag_answerer`) generates grounded response.
- If context is insufficient, model is instructed to output `INSUFFICIENT_INFORMATION`.
- Sources are attached in output when available.

### Prompting and context formatting details

- Each passage is tagged with:
  - `id`
  - `relevance` label based on ranking score
  - optional `page` and `source` attributes
- Relevance labels are assigned as:
  - `primary` if score > 0.75
  - `supporting` if score > 0.45
  - `background` otherwise
- The user prompt explicitly instructs:
  - teacher-style structured answer
  - markdown headings
  - preserve numbers/conditions/exceptions
  - reproduce tables as markdown tables
  - cite page numbers inline

Insufficient context handling:

- If model returns `INSUFFICIENT_INFORMATION` (or short equivalent), system returns a safe fallback message with confidence `0.0`.

Default chat model:

- `llama-3.3-70b-versatile` (Groq)

## 6) Database / Storage Used

Current implementation uses in-memory stores at runtime.

### Document retrieval store

- `UploadedDocumentStore` keeps per-session chunks in Python memory:
  - Chunk text
  - Metadata
  - Embedding vectors
  - Lexical term sets
- BM25 indexes are also cached in memory.
- No persistent vector database is currently active in this retrieval path.

Data structures used:

- `StoredChunk` dataclass:
  - `content: str`
  - `metadata: Dict`
  - `embedding: List[float]`
  - `terms: frozenset`
- Session maps:
  - `_session_chunks[session_id] -> List[StoredChunk]`
  - `_session_files[session_id] -> List[filename]`
  - `_bm25_cache[session_id] -> BM25 index`
  - `_bm25_dirty[session_id] -> bool`

### Conversation state store

- LangGraph checkpointer: `InMemorySaver`
- Session message history: in-memory (`InMemoryChatMessageHistory`)

Ingestion state store:

- `IngestionTracker` stores per-file async ingestion status:
  - `pending`
  - `done`
  - `error`
- `/api/ingest` marks and runs background indexing
- `/api/upload` can wait for pending ingestion completion

### Note on ChromaDB

- `chromadb` is present in dependencies/logging noise filters, but the active RAG retrieval path shown above does not currently use a Chroma collection for indexing/querying.

## 7) Confidence and Fallback Behavior

- Document route returns RAG answer only when top document score passes threshold.
- If score is below threshold, system falls back to web search + generation.

### Exact fallback gate

In document route:

1. Retrieve docs from uploaded store
2. Compute top score using first available key in priority:
  - `combined_score`
  - `rerank_score`
  - `score`
  - metadata equivalents
3. If `top_score >= document_relevance_threshold` -> generate document-grounded response
4. Else -> run Tavily web search and generate web-grounded response
5. If web also returns nothing -> return no-information fallback

Default threshold:

- `document_relevance_threshold = 0.20`

## 8) Key Config Defaults (Current)

- `llm_model_name = llama-3.3-70b-versatile`
- `embedding_model_name = microsoft/harrier-oss-v1-270m`
- `embedding_device = cpu`
- `document_chunk_size = 1200`
- `document_chunk_overlap = 300`
- `rag_top_k = 6`
- `rag_candidate_k = 50`
- `rag_max_query_variants = 5`
- `rag_window_size = 2`
- `enable_multi_query_retrieval = true`
- `enable_llm_multi_query = true`
- `enable_cross_encoder_rerank = true`
- `enable_hyde = true`
- `cross_encoder_model_name = cross-encoder/ms-marco-MiniLM-L-12-v2`
- `document_relevance_threshold = 0.20`

## 9) Step-by-step Runtime Summary (Single Query)

1. User sends query.
2. Query merge checks follow-up markers and may rewrite.
3. Router picks general/document/web.
4. For document route:
  - Build query variants (LLM + heuristic rules)
  - Optional HyDE synthetic embedding
  - Dense retrieval rankings
  - BM25 rankings (if available)
  - RRF fusion
  - Optional cross-encoder rerank
  - Top-k selection
  - Optional context window expansion
5. Generator builds teacher-style answer from selected passages.
6. Confidence gate decides document answer vs web fallback.

## 10) Failure Modes and Current Safeguards

- LLM multi-query failure -> heuristic query variants still run.
- HyDE failure -> retrieval continues without HyDE.
- BM25 unavailable (`rank_bm25` missing) -> dense retrieval continues.
- Cross-encoder failure -> fallback to RRF ordering.
- Insufficient context from generator -> safe low-confidence response.
- Document relevance too low -> web fallback path.