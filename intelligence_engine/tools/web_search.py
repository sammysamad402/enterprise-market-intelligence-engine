"""
tools/web_search.py
-------------------
Asynchronous live-web search tool backed by the Tavily API.

The Tavily Python SDK exposes a synchronous client.  We run it inside
``asyncio.get_event_loop().run_in_executor`` so it integrates cleanly with
the rest of the async pipeline without blocking the event loop.

Returned snippets are pre-cleaned: leading/trailing whitespace stripped,
empty strings discarded, and duplicates removed while preserving order.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from tavily import TavilyClient

from intelligence_engine.config import get_settings

logger = logging.getLogger("intelligence_engine.tools.web_search")


def _run_tavily_search(query: str, max_results: int, api_key: str) -> List[str]:
    """
    Blocking Tavily search executed in a thread-pool worker.

    Parameters
    ----------
    query:
        The search query string.
    max_results:
        Maximum number of result snippets to return.
    api_key:
        Tavily API key (plain string, already unwrapped from SecretStr).

    Returns
    -------
    List[str]
        Cleaned text snippets extracted from Tavily's ``results`` payload.
    """
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )

    snippets: List[str] = []

    # Tavily returns a top-level "answer" field with a synthesised summary.
    top_answer: str = response.get("answer", "")
    if top_answer and top_answer.strip():
        snippets.append(top_answer.strip())

    # Individual search results each carry a "content" field with the page
    # excerpt that is most relevant to the query.
    for result in response.get("results", []):
        content: str = result.get("content", "")
        url: str = result.get("url", "")

        if content and content.strip():
            # Prefix the URL so the Critic and Generator can cite it directly.
            entry = f"[{url}] {content.strip()}" if url else content.strip()
            snippets.append(entry)

    # Deduplicate while preserving insertion order.
    seen: set[str] = set()
    unique: List[str] = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    logger.info(
        "Tavily search complete — query=%r  snippets_returned=%d",
        query[:80],
        len(unique),
    )
    return unique


async def async_web_query(query: str) -> List[str]:
    """
    Execute a live web search asynchronously via the Tavily API.

    The synchronous Tavily client is offloaded to a thread-pool executor so
    the coroutine is non-blocking from the event loop's perspective.

    Parameters
    ----------
    query:
        Research question or keyword string to search for.

    Returns
    -------
    List[str]
        A list of clean text snippets (possibly empty on API failure).
        Each snippet from a search result is prefixed with its source URL.

    Notes
    -----
    * All exceptions are caught and logged; the function never raises so that
      a Tavily outage cannot crash the upstream ``asyncio.gather`` call.
    * The function logs a warning when the list is empty so operators can
      detect retrieval degradation in production monitoring dashboards.
    """
    cfg = get_settings()

    try:
        loop = asyncio.get_running_loop()
        snippets: List[str] = await loop.run_in_executor(
            None,  # use the default ThreadPoolExecutor
            _run_tavily_search,
            query,
            cfg.max_search_results,
            cfg.tavily_api_key.get_secret_value(),
        )
        if not snippets:
            logger.warning(
                "Tavily returned zero snippets for query=%r — "
                "check your API key and quota.",
                query[:80],
            )
        return snippets

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Tavily web search failed for query=%r — %s: %s  "
            "Returning empty list so the pipeline remains operational.",
            query[:80],
            type(exc).__name__,
            exc,
        )
        return []
