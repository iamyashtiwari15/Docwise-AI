


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

    def search(self, query: str, trusted_sites_only: bool = False) -> str:
        """
        Perform a web search using Tavily API.
        If trusted_sites_only is True, restrict to trusted sites.
        Returns a clean, readable string of results.
        """
        self.logger.info(f"Starting search for query: {query} | trusted_sites_only={trusted_sites_only}")
        clean_query = query.strip('"\'')
        final_query = clean_query

        if trusted_sites_only:
            self.logger.info(f"Performing TRUSTED site search for: {clean_query}")
            site_filter = " OR ".join([f"site:{site}" for site in TRUSTED_SITES])
            final_query = f"{clean_query} ({site_filter})"
        else:
            self.logger.info(f"Performing GENERAL web search for: {clean_query}")

        try:
            self.logger.info(f"Executing query: {final_query}")
            if len(final_query) > 400:
                final_query = final_query[:400]

            # The API returns a dictionary, so we store it as such.
            # Add timeout handling for better performance
            import asyncio
            try:
                result_dict = self.tavily_search.invoke({"query": final_query})
            except Exception as api_error:
                self.logger.warning(f"Tavily API call failed: {api_error}")
                return "Web search temporarily unavailable. Please try again."
            
            self.logger.info(f"Raw result: {str(result_dict)[:200]}")

            # The actual search results are in the "results" key of the dictionary.
            results_list = result_dict.get("results", [])

            # Now we check and process the list we extracted.
            if results_list and isinstance(results_list, list):
                # Truncate content for faster processing and limit to 2 results max
                formatted_results = "\n\n".join([
                    f"Title: {item.get('title', 'N/A')[:100]}{'...' if len(item.get('title', '')) > 100 else ''}\nURL: {item.get('url', 'N/A')}\nSnippet: {item.get('content', 'N/A')[:200]}{'...' if len(item.get('content', '')) > 200 else ''}" 
                    for item in results_list[:2]  # Limit to first 2 results
                ])
                return formatted_results.strip()

            return "No relevant results were found from the web search."

        except Exception as e:
            self.logger.error(f"Error during web search: {e}")
            return f"Error retrieving web search results: {e}"
