"""
Web search tool using Tavily API.
"""
import requests
from typing import Optional
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient

from langgraph_chatbot.config import settings
from langgraph_chatbot.utils.logger import setup_logger

logger = setup_logger(__name__)


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web for information using Tavily API.
    
    Parameters
    ----------
    query : str
        Search query
    max_results : int, optional
        Maximum number of results to return, by default 5
        
    Returns
    -------
    dict
        Search results with titles, URLs, and snippets
        
    Examples
    --------
    >>> web_search("latest AI news", max_results=3)
    {'status': 'success', 'query': 'latest AI news', 'results': [...]}
    """
    try:
        logger.info(f"Web search: '{query}' (max_results={max_results})")
        
        if not settings.tavily_api_key:
            logger.error("Tavily API key not configured")
            return {
                "status": "error",
                "error_type": "api_key_missing",
                "message": "Web search API key not configured. Please add TAVILY_API_KEY to .env file"
            }
        
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )
        
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                # "extracted_text": _extract_full_content_from_url(item.get("url", "")),
                "score": item.get("score", 0)
            })
        
        logger.info(f"Found {len(results)} search results")
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Web search error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_type": "search_failed",
            "message": str(e)
        }
    

def _extract_full_content_from_url(url: str, timeout: int = 15) -> Optional[str]:
    """
    Extract the full textual content from a given URL.

    This function fetches the HTML content of the provided URL,
    removes non-textual elements (scripts, styles, etc.), and
    returns the cleaned visible text.

    Parameters
    ----------
    url : str
        The URL from which the content needs to be extracted.
    timeout : int, optional
        Timeout for the HTTP request in seconds (default is 15).

    Returns
    -------
    Optional[str]
        Extracted textual content from the webpage as a single string.
        Returns None if extraction fails.

    Raises
    ------
    ValueError
        If the URL is empty or invalid.
    """

    if not url or not isinstance(url, str):
        raise ValueError("A valid URL string must be provided.")

    logger.info("Starting content extraction from URL: %s", url)

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        if not text:
            logger.warning("No textual content found at URL: %s", url)
            return None

        logger.info("Successfully extracted content from URL: %s", url)
        return text

    except requests.exceptions.RequestException as exc:
        logger.error("HTTP error while fetching URL %s: %s", url, exc, exc_info=True)

    except Exception as exc:
        logger.exception("Unexpected error while extracting content from URL %s", url)

    return None
