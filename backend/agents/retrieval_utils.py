import re
import json
import logging
from typing import Collection, List


logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}")
_QUESTION_PREFIXES = (
    "what is",
    "what are",
    "who is",
    "who are",
    "when is",
    "when are",
    "where is",
    "where are",
    "tell me about",
    "explain",
    "summarize",
    "summarise",
    "describe",
    "review",
    "analyze",
    "analyse",
    "list",
    "outline",
)
_COMPOUND_SPLIT_PATTERN = re.compile(r"\b(?:and|or|also|plus|vs\.?|versus)\b", re.IGNORECASE)

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "this",
    "to",
    "us",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_query_terms(text: str) -> List[str]:
    terms: List[str] = []
    seen = set()

    for match in _TOKEN_PATTERN.finditer((text or "").lower()):
        token = match.group(0).strip("-_")
        if not token or token in STOPWORDS:
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)

    return terms


def lexical_overlap_score(query_terms: Collection[str], content_terms: Collection[str]) -> float:
    query_set = {term for term in query_terms if term}
    content_set = {term for term in content_terms if term}
    if not query_set or not content_set:
        return 0.0

    overlap = query_set & content_set
    if not overlap:
        return 0.0

    recall = len(overlap) / len(query_set)
    precision = len(overlap) / len(content_set)
    return min(1.0, recall * 0.85 + precision * 0.15)


def _strip_question_prefix(text: str) -> str:
    cleaned = normalize_whitespace(text.strip(" ,.;:"))
    lowered = cleaned.lower()
    for prefix in _QUESTION_PREFIXES:
        if lowered.startswith(f"{prefix} "):
            return cleaned[len(prefix):].strip(" ,.;:")
    return cleaned


def should_use_llm_multi_query(query: str, min_terms: int = 5) -> bool:
    cleaned_query = normalize_whitespace(query)
    if not cleaned_query:
        return False

    terms = extract_query_terms(cleaned_query)
    if len(terms) >= min_terms:
        return True
    if _COMPOUND_SPLIT_PATTERN.search(cleaned_query):
        return True
    if re.search(r"[?;\n:]", cleaned_query):
        return True
    return False


def _parse_llm_query_variants(raw_output: str) -> List[str]:
    cleaned_output = (raw_output or "").strip()
    if not cleaned_output:
        return []

    try:
        payload = json.loads(cleaned_output)
        if isinstance(payload, list):
            return [normalize_whitespace(str(item)) for item in payload if normalize_whitespace(str(item))]
    except Exception:
        pass

    variants: List[str] = []
    for line in cleaned_output.splitlines():
        candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not candidate:
            continue
        variants.append(normalize_whitespace(candidate.strip("\"'")))
    return variants


def _generate_llm_query_variants(query: str, max_variants: int, llm) -> List[str]:
    prompt = f"""
Generate up to {max(1, max_variants - 1)} short retrieval queries for the following user question.

Rules:
- Preserve the original meaning exactly.
- Prefer concrete retrieval phrasing over conversational wording.
- If the request has multiple parts, split them into focused sub-queries.
- Keep each query on its own line.
- Return only the queries, no explanation.

User question:
{query}
"""
    result = llm.invoke(prompt)
    content = result.content if hasattr(result, "content") else str(result)
    return _parse_llm_query_variants(content)


def generate_query_variants(query: str, max_variants: int = 3, llm=None) -> List[str]:
    cleaned_query = normalize_whitespace(query)
    if not cleaned_query:
        return []

    variants = [cleaned_query]
    seen = {cleaned_query.lower()}
    base_terms = extract_query_terms(cleaned_query)
    if llm and should_use_llm_multi_query(cleaned_query):
        try:
            for candidate in _generate_llm_query_variants(cleaned_query, max_variants, llm):
                normalized_candidate = _strip_question_prefix(candidate)
                if not normalized_candidate:
                    continue
                lowered = normalized_candidate.lower()
                if lowered in seen:
                    continue

                candidate_terms = extract_query_terms(normalized_candidate)
                shared_terms = set(candidate_terms) & set(base_terms)
                if candidate_terms and not shared_terms:
                    continue

                variants.append(normalized_candidate)
                seen.add(lowered)
                if len(variants) >= max(1, max_variants):
                    return variants
        except Exception as error:
            logger.warning("[RETRIEVE] LLM multi-query generation failed, using heuristic variants: %s", error)

    if len(base_terms) < 4 and not re.search(r"[?;\n]", cleaned_query):
        return variants

    candidates: List[str] = []
    candidates.extend(part for part in re.split(r"[?;\n]+", cleaned_query) if part.strip())

    if len(base_terms) >= 4 and _COMPOUND_SPLIT_PATTERN.search(cleaned_query):
        candidates.extend(part for part in _COMPOUND_SPLIT_PATTERN.split(cleaned_query) if part.strip())

    if len(base_terms) >= 6:
        candidates.append(" ".join(base_terms[: min(8, len(base_terms))]))

    for candidate in candidates:
        normalized_candidate = _strip_question_prefix(candidate)
        if not normalized_candidate:
            continue

        lowered = normalized_candidate.lower()
        if lowered in seen:
            continue

        candidate_terms = extract_query_terms(normalized_candidate)
        if len(candidate_terms) < 2 and len(normalized_candidate.split()) < 3:
            continue

        shared_terms = set(candidate_terms) & set(base_terms)
        if candidate_terms and len(shared_terms) < 1:
            continue

        variants.append(normalized_candidate)
        seen.add(lowered)
        if len(variants) >= max(1, max_variants):
            break

    return variants
