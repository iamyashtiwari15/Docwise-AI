# Backend Implementation Approval

## Goal
Make the backend production-ready, modular, cheaper to run, and faster for:

- normal chatbot requests
- RAG-based answers
- web-search-assisted answers

This document is a proposal only. No active implementation changes are left in the codebase.

## Current Issues Found

### 1. High request cost on the hot path
The backend currently stacks multiple expensive steps in a single request path:

1. guardrails LLM call
2. query rewrite LLM call
3. query classification LLM fallback
4. query expansion / planning in RAG
5. final response generation
6. optional verification pass
7. optional web search

This makes latency unpredictable and increases token/API usage.

### 2. Hidden overengineering in retrieval
The retriever includes agentic query planning and expansion, but the current retrieval flow does not fully benefit from that complexity. This adds architecture weight and model cost without strong measurable value.

### 3. Web search is too eager
For some query classes, web and local retrieval are both used too early. In production, web search should be a fallback or a policy-driven branch, not a common default.

### 4. Shared resources are initialized in multiple places
Embedding models and Chroma access are created from separate modules instead of one shared resource layer. That increases memory usage, startup cost, and maintenance overhead.

### 5. Session memory is not production-safe
Chat history is stored in a global in-memory dictionary. That causes:

- no TTL or eviction
- no concurrency protection
- no horizontal scalability
- memory growth risk

### 6. Response handling is brittle
Some responses are stringified dictionaries and then reparsed later. That is fragile, slower than necessary, and easy to break when output shape changes.

### 7. Module boundaries are weak
The current backend mixes:

- API transport concerns
- workflow orchestration
- retrieval policy
- session management
- model loading

This makes the system harder to test and harder to evolve.

## What Is Overengineered

The main overengineering points are:

1. multi-step query expansion/planning for ordinary user questions
2. too many LLM-assisted decisions before actual answer generation
3. separate loader patterns for similar LLM/embedding responsibilities
4. workflow branching that is heavier than the current domain really needs

## What Is Actually Needed

The backend should move to a simpler decision model:

1. **Fast path first**
   - rule-based safety filter
   - rule-based query classification for obvious queries
   - direct RAG for most domain questions

2. **Escalate only when needed**
   - web search only when local confidence is weak or the query needs live/external data
   - LLM rewrite only for follow-up references like "what about this?"
   - verification only for high-risk responses

3. **Shared infrastructure**
   - one config layer
   - one model loader strategy
   - one retrieval resource layer
   - one session store abstraction

## Proposed Target Structure

```text
backend/
  api/
    routes/
      chat.py
      history.py
      upload.py
    schemas.py

  core/
    config.py
    logging.py

  services/
    chat_service.py
    session_store.py
    safety_service.py

  orchestration/
    workflow_manager.py
    routing_policy.py

  retrieval/
    vector_store.py
    document_retriever.py
    section_retriever.py
    web_retriever.py
    ranking.py

  llm/
    loaders.py
    prompts.py
    response_generator.py

  agents/
    agent_decision.py
    consumer_rights_chat_agent.py
    vision_agents/

  tests/
    test_workflow.py
    test_guardrails.py
    test_retrieval_policy.py
    test_session_store.py
```

## Proposed Implementation

### Phase 1: Cost and latency reduction

#### A. Replace always-on LLM guardrails with a fast local gate
- Use regex/rule filters for obvious unsafe or off-domain requests.
- Keep optional LLM guardrails only as fallback for ambiguous cases.

**Expected effect**
- lower average latency
- lower token cost
- more stable throughput

#### B. Remove agentic query expansion from the default RAG path
- Use the original query for most retrieval.
- Only enable query rewrite for follow-up/context-dependent user turns.
- Disable multi-step planning/expansion unless a query is genuinely complex.

**Expected effect**
- fewer LLM calls per request
- simpler retrieval behavior
- easier debugging

#### C. Make web search fallback-based
- Run local retrieval first.
- Compute a lightweight confidence score from retrieved documents.
- Trigger web search only if:
  - local confidence is weak, or
  - the query explicitly requires fresh/external information, or
  - it is a contact-info / real-time policy case

**Expected effect**
- lower Tavily usage
- lower end-to-end latency
- stronger control of spend

#### D. Disable verification by default
- Keep answer verification behind config.
- Enable it only for high-risk flows like legal procedure guidance or refund escalation content.

**Expected effect**
- removes extra model pass from most requests

### Phase 2: Production modularity

#### A. Add a shared config layer
- Centralize env parsing and feature flags.
- Put all performance-sensitive toggles in one place.

Suggested flags:

- `ENABLE_LLM_GUARDRAILS_FALLBACK`
- `ENABLE_LLM_QUERY_REWRITE`
- `ENABLE_LLM_QUERY_EXPANSION`
- `ENABLE_RESPONSE_VERIFICATION`
- `RAG_TOP_K`
- `MAX_CONTEXT_DOCS`
- `MIN_RAG_CONFIDENCE_FOR_WEB_FALLBACK`

#### B. Add shared retrieval resources
- Single cached embedding loader
- Single cached Chroma loader
- Reuse across RAG and section retrieval

#### C. Introduce a session store abstraction
- Replace the global dict with a bounded store
- Add TTL and max-history trimming
- Keep interface swappable so Redis can be added later

#### D. Normalize response objects
- Keep one response schema from workflow to API
- stop converting dicts to strings and reparsing them later

### Phase 3: Production hardening

#### A. Add focused tests
- routing policy tests
- guardrail fast-path tests
- retrieval fallback tests
- session store tests

#### B. Improve observability
- structured logs for:
  - query type
  - retrieval path used
  - web fallback used or not
  - response generation latency
  - source count

#### C. Prepare for horizontal scale
- session backend interface ready for Redis
- no critical state in process memory only

## Recommended First Implementation Scope

The safest first delivery is:

1. add config layer
2. add session store abstraction
3. simplify query merge behavior
4. reduce unnecessary LLM calls in guardrails/retrieval
5. make web search fallback-only
6. add regression tests

This gives the best cost/latency return without forcing a full rewrite.

## Expectation Alignment and Learning Loop

This section ensures the implementation is not only delivered, but also teaches us how to improve the next version of the system.

### What must be true for this implementation to meet expectations

1. A user can upload a PDF or DOCX and ask grounded questions against that content.
2. The system prefers uploaded documents first, not the web, when document context is available.
3. Web search is used only when document confidence is weak or the question is clearly external/current.
4. Responses stay traceable to the underlying source chunks or web results.
5. The system remains modular enough that routing, ingestion, retrieval, and answer generation can be improved independently.

### What we should learn from the first implementation

We should treat the first rollout as a feedback system, not just a feature release.

Key learning signals:

- which query types are routed to document, web, and general paths
- how often document retrieval fails because of poor chunking versus poor routing
- what top similarity/confidence scores look like for good answers versus bad answers
- how often web fallback is triggered after document retrieval
- which file types and document structures produce weak extraction quality
- where latency is spent: upload parsing, embedding, retrieval, routing, or response generation
- which answers users reformulate, retry, or abandon

### How those learnings should improve the system

The implementation should be instrumented so we can improve it based on evidence:

1. **Routing improvement**
   - refine document/web/general routing using real query distributions
   - tighten or relax confidence thresholds based on false-positive and false-negative retrieval cases

2. **Retrieval improvement**
   - improve chunking strategy for policies, reports, contracts, manuals, and mixed-format documents
   - add metadata-aware retrieval once we know which metadata actually helps

3. **Answer quality improvement**
   - identify where grounded responses still hallucinate or miss important context
   - improve prompts only after observing repeated failure patterns

4. **System design improvement**
   - decide when to move from in-memory session storage to Redis-backed storage based on usage patterns
   - decide whether session embeddings should stay in-memory or move to a persistent/vector-store-backed strategy

### Recommended metrics for the first rollout

- document upload success rate
- extraction success rate by file type
- top retrieved chunk confidence score
- document-to-web fallback rate
- no-answer rate
- average response latency by route
- average cost per route
- user follow-up rate after an answer

This gives us a practical loop:

**implement -> observe -> learn -> tune -> modularize further**

## Approval Points

Please approve these design decisions before implementation:

1. **Guardrails strategy**: local-first with optional LLM fallback
2. **Web search policy**: fallback-only, not default parallel retrieval
3. **Session strategy**: bounded in-memory abstraction now, Redis-ready interface next
4. **Verification policy**: disabled by default, enabled only for high-risk flows
5. **Refactor scope**: incremental modular refactor, not full rewrite

## Final Recommendation

I recommend an **incremental refactor** rather than replacing the backend all at once.

That will:

- reduce latency fastest
- cut model and search cost fastest
- keep the current app usable during changes
- make the system modular enough for production rollout
