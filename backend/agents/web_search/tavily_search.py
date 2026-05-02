


import logging
import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from core.config import get_settings

load_dotenv()  # Load TAVILY_API_KEY from .env file

TRUSTED_SITES = [
    "wikipedia.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
]

class FlexibleTavilySearchAgent:
    """
    Flexible Tavily search agent supporting trusted-only and unrestricted search.
    """
    def __init__(self):
        settings = get_settings()
        self.tavily_search = TavilySearch(max_results=settings.web_search_max_results)
        self.logger = logging.getLogger(__name__)

    def search_results(self, query: str, trusted_sites_only: bool = False) -> list[dict]:
        """
        Returns a list of dicts with keys: title, url, content.
        Empty list on failure.
        """
        clean_query = query.strip('"\'')
        final_query = clean_query

        if trusted_sites_only:
            site_filter = " OR ".join([f"site:{site}" for site in TRUSTED_SITES])
            final_query = f"{clean_query} ({site_filter})"

        if len(final_query) > 400:
            final_query = final_query[:400]

        self.logger.info("[TAVILY] Searching | trusted_only=%s | query=%.80r", trusted_sites_only, final_query)
        try:
            result_dict = self.tavily_search.invoke({"query": final_query})
            results_list = result_dict.get("results", [])
            self.logger.info("[TAVILY] Got %d results", len(results_list))
            return [
                {
                    "title": item.get("title", "Web result"),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
                for item in results_list
                if item.get("content")
            ]
        except Exception as e:
            self.logger.warning("[TAVILY] Search failed: %s", e)
            return []

    def search(self, query: str, trusted_sites_only: bool = False) -> str:
        """Returns formatted string for backward compatibility."""
        results = self.search_results(query, trusted_sites_only)
        if not results:
            return "No relevant results were found from the web search."
        return "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:600]}{'...' if len(r['content']) > 600 else ''}"
            for r in results
        ])
