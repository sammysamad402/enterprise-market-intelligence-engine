"""
tools/vector_search.py
----------------------
Asynchronous semantic search tool backed by a Qdrant vector database.

Compatible with qdrant-client==1.18.0 and fastembed.
Uses query_points() with explicit local FastEmbed embeddings — the
recommended modern API after query() was deprecated in 1.17.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import ScoredPoint

from intelligence_engine.config import get_settings

logger = logging.getLogger("intelligence_engine.tools.vector_search")

# ---------------------------------------------------------------------------
# Payload field names — adjust if your collection uses different keys
# ---------------------------------------------------------------------------
_TEXT_PAYLOAD_FIELD = "text"
_ALT_TEXT_FIELDS = ("content", "chunk", "page_content", "body")

# FastEmbed model used at index time.  Must match the model you used when
# populating the collection.  The default (BAAI/bge-small-en-v1.5) is what
# qdrant-client's own FastEmbed integration uses.
_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Module-level singleton so the ONNX model is loaded once per process.
_embedding_model: Optional[TextEmbedding] = None


def _get_embedding_model() -> TextEmbedding:
    """Return (and lazily initialise) the module-level FastEmbed model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading FastEmbed model: %s", _FASTEMBED_MODEL)
        _embedding_model = TextEmbedding(model_name=_FASTEMBED_MODEL)
    return _embedding_model


def _embed_query(query: str) -> List[float]:
    """
    Embed a single query string with FastEmbed.

    TextEmbedding.query_embed() is the query-specific path (adds the
    query prefix that BGE models require); embed() is the document path.
    Falls back to embed() if query_embed() is unavailable.
    """
    model = _get_embedding_model()
    try:
        vectors = list(model.query_embed(query))
    except AttributeError:
        vectors = list(model.embed([query]))
    return vectors[0].tolist()


def _extract_text(point: ScoredPoint) -> Optional[str]:
    """
    Pull a text string out of a ScoredPoint's payload.

    Tries the canonical field first, then common alternatives, then
    falls back to the full payload repr so no data is silently lost.
    """
    if not point.payload:
        return None

    for field in (_TEXT_PAYLOAD_FIELD, *_ALT_TEXT_FIELDS):
        value = point.payload.get(field)
        if value and isinstance(value, str) and value.strip():
            return value.strip()

    raw = str(point.payload)
    return raw if raw.strip() else None


async def async_vector_query(query: str) -> List[str]:
    """
    Execute an asynchronous semantic search against the Qdrant collection.

    Parameters
    ----------
    query:
        Natural-language research question.  Embedded locally with FastEmbed.

    Returns
    -------
    List[str]
        Text chunks ranked by cosine similarity (highest first).
        Returns an empty list on any connection or query error so that the
        upstream asyncio.gather() call is never disrupted.

    Implementation notes
    --------------------
    * client.query() was deprecated in qdrant-client 1.17 and removed in
      1.18.  We call client.query_points() directly and embed the query
      ourselves with fastembed.TextEmbedding so no server-side inference
      infrastructure is required.
    * with_payload is passed ONLY to query_points(), never through **kwargs,
      which is what caused the "multiple values" TypeError in the old code.
    * The embedding step is CPU-bound; we run it in the default executor so
      the event loop is not blocked.
    """
    cfg = get_settings()

    qdrant_api_key: Optional[str] = (
        cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
    )

    client: AsyncQdrantClient = AsyncQdrantClient(
        url=cfg.qdrant_url,
        api_key=qdrant_api_key,
        timeout=10.0,
    )

    try:
        # ------------------------------------------------------------------
        # Step 1 — embed the query locally (non-blocking via executor)
        # ------------------------------------------------------------------
        loop = asyncio.get_running_loop()
        query_vector: List[float] = await loop.run_in_executor(
            None, _embed_query, query
        )

        # ------------------------------------------------------------------
        # Step 2 — search with the modern query_points() API
        #
        # Key differences from the old client.query() call:
        #   • query=   accepts a float list (dense vector), not query_text=
        #   • with_payload is a direct parameter of query_points(), NOT a
        #     **kwargs passthrough — passing it via kwargs caused the
        #     "multiple values" TypeError.
        #   • Returns a QueryResponse whose .points attribute holds the list.
        # ------------------------------------------------------------------
        response = await client.query_points(
            collection_name=cfg.qdrant_collection,
            query=query_vector,
            limit=cfg.max_vector_results,
            with_payload=True,
        )

        results: List[ScoredPoint] = response.points

        chunks: List[str] = []
        for point in results:
            text = _extract_text(point)
            if text:
                chunks.append(text)

        logger.info(
            "Qdrant vector search complete — collection=%r  query=%r  "
            "hits=%d  chunks_extracted=%d",
            cfg.qdrant_collection,
            query[:80],
            len(results),
            len(chunks),
        )
        return chunks

    except (UnexpectedResponse, ConnectionError, TimeoutError) as conn_exc:
        logger.warning(
            "Qdrant connection error for collection=%r — %s: %s  "
            "Returning empty list; web-search context will be used exclusively.",
            cfg.qdrant_collection,
            type(conn_exc).__name__,
            conn_exc,
        )
        return []

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected Qdrant error — %s: %s  Returning empty list.",
            type(exc).__name__,
            exc,
        )
        return []

    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass