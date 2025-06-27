# FusedChatbot/server/ai_core_service/tools/web_search.py

import logging
from duckduckgo_search import DDGS
from newspaper import Article, ArticleException

logger = logging.getLogger(__name__)

def _fetch_and_parse_url(url: str) -> str:
    """
    Fetches content from a URL and parses it to get clean text.
    
    Args:
        url (str): The URL to scrape.
        
    Returns:
        str: The clean text content of the article, or an empty string if it fails.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except ArticleException as e:
        logger.warning(f"Could not process article at {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching {url}: {e}", exc_info=True)
        return ""

def perform_search(query: str, max_results: int = 3) -> str:
    """
    Performs a web search using DuckDuckGo, scrapes the top results, and
    returns a formatted string of the content. This function does not require an API key.

    Args:
        query (str): The search query.
        max_results (int): The number of top search results to process.

    Returns:
        str: A formatted string of search results, or an empty string if it fails.
    """
    logger.info(f"Performing key-less web search for query: '{query}'")
    
    try:
        # 1. Get search results from DuckDuckGo
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
        
        if not search_results:
            logger.info("DuckDuckGo search returned no results.")
            return ""

        # 2. Scrape and parse the content from the top URLs
        formatted_results = []
        for i, result in enumerate(search_results):
            url = result.get('href')
            if not url:
                continue

            logger.info(f"Scraping content from URL: {url}")
            content = _fetch_and_parse_url(url)
            
            if content:
                formatted_results.append(
                    f"[{i+1}] Source: {url}\nContent: {content[:1500]}..." # Truncate content to keep context concise
                )
        
        if not formatted_results:
            logger.warning("Web search found URLs but failed to scrape any content.")
            return ""

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"An error occurred during DuckDuckGo search: {e}", exc_info=True)
        return "" # Return empty string on failure to not break the chain