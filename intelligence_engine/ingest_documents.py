"""
ingest_documents.py
-------------------
One-shot ingestion pipeline for the enterprise_intel Qdrant collection.

What this script does
---------------------
1. Reads every .txt file from the sample_documents/ directory.
2. Chunks each document into overlapping windows so long files produce
   multiple searchable segments.
3. Embeds all chunks locally with FastEmbed (BAAI/bge-small-en-v1.5, 384-dim).
4. Creates the "enterprise_intel" collection in Qdrant if it does not exist,
   using cosine similarity and the correct vector dimension.
5. Upserts all vectors and payloads in batches.

Compatibility
-------------
- qdrant-client==1.18.0   (uses create_collection + upsert, no deprecated APIs)
- fastembed               (local ONNX inference, no external embedding API needed)
- Matches async_vector_query() in tools/vector_search.py:
    * Collection name  : enterprise_intel  (cfg.qdrant_collection default)
    * Payload field    : "text"            (_TEXT_PAYLOAD_FIELD constant)
    * Vector dimension : 384
    * Distance metric  : Cosine

Usage
-----
    python ingest_documents.py

    # Override Qdrant URL or collection:
    QDRANT_URL=http://localhost:6333 QDRANT_COLLECTION=enterprise_intel python ingest_documents.py

    # Point at a different documents directory:
    python ingest_documents.py --docs-dir /path/to/my/docs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Generator, List

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_documents")

# ---------------------------------------------------------------------------
# Constants — must match tools/vector_search.py
# ---------------------------------------------------------------------------

COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "enterprise_intel")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")

FASTEMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
VECTOR_DIM: int = 384          # bge-small-en-v1.5 output dimension
DISTANCE_METRIC: Distance = Distance.COSINE

# Chunking parameters
CHUNK_SIZE: int = 512          # tokens / words (approximate; we split on words)
CHUNK_OVERLAP: int = 64        # words of overlap between consecutive chunks
BATCH_SIZE: int = 32           # points per upsert call

# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[str, None, None]:
    """
    Split text into overlapping word-based windows.

    Word splitting is cheaper than token splitting and produces chunks that
    are close enough to 512 tokens for bge-small-en-v1.5 (which truncates
    at 512 subword tokens). Each chunk is yielded as a plain string.
    """
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += chunk_size - overlap  # advance with overlap


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Load all .txt files from docs_dir.

    Returns a list of dicts with keys:
        source   — relative filename
        company  — derived from filename stem (title-cased)
        chunks   — list of text chunk strings
    """
    docs = []
    txt_files = sorted(docs_dir.glob("*.txt"))

    if not txt_files:
        logger.error("No .txt files found in %s", docs_dir)
        sys.exit(1)

    for path in txt_files:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            logger.warning("Skipping empty file: %s", path.name)
            continue

        chunks = list(chunk_text(raw))
        docs.append(
            {
                "source": path.name,
                "company": path.stem.replace("_", " ").title(),
                "chunks": chunks,
            }
        )
        logger.info("Loaded %-30s → %d chunk(s)", path.name, len(chunks))

    return docs


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create the collection if it does not already exist.

    Uses COSINE distance and the 384-dimensional bge-small-en-v1.5 vectors.
    Safe to call on an already-existing collection — skips creation silently.
    """
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        info = client.get_collection(collection_name)
        existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
        logger.info(
            "Collection '%s' already exists (dim=%d). Skipping creation.",
            collection_name,
            existing_dim,
        )
        if existing_dim != VECTOR_DIM:
            logger.error(
                "Dimension mismatch: collection has dim=%d but model outputs dim=%d. "
                "Drop the collection and re-run, or use the matching model.",
                existing_dim,
                VECTOR_DIM,
            )
            sys.exit(1)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=DISTANCE_METRIC,
        ),
    )
    logger.info(
        "Created collection '%s' (dim=%d, distance=%s).",
        collection_name,
        VECTOR_DIM,
        DISTANCE_METRIC.value,
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(
    chunks: list[str], model: TextEmbedding
) -> list[list[float]]:
    """
    Embed a list of text chunks.

    Uses model.passage_embed() (adds document-side prefix for BGE) if
    available, falling back to embed() for older fastembed versions.
    """
    try:
        vectors = list(model.passage_embed(chunks))
    except AttributeError:
        vectors = list(model.embed(chunks))
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest(
    client: QdrantClient,
    docs: list[dict],
    model: TextEmbedding,
    collection_name: str,
) -> int:
    """
    Embed all chunks and upsert them into the collection.

    Points are uploaded in batches of BATCH_SIZE to avoid hitting
    Qdrant's request body size limits. Each point carries:
        text    — the raw chunk string  (matches _TEXT_PAYLOAD_FIELD)
        source  — originating filename
        company — human-readable company name

    Returns the total number of points upserted.
    """
    all_points: list[PointStruct] = []

    for doc in docs:
        chunks = doc["chunks"]
        logger.info(
            "Embedding %d chunk(s) for '%s' …", len(chunks), doc["company"]
        )
        vectors = embed_chunks(chunks, model)

        for chunk_text_str, vector in zip(chunks, vectors):
            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk_text_str,      # ← must match _TEXT_PAYLOAD_FIELD
                        "source": doc["source"],
                        "company": doc["company"],
                    },
                )
            )

    total = len(all_points)
    logger.info("Uploading %d point(s) in batches of %d …", total, BATCH_SIZE)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_points[batch_start : batch_start + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "  Upserted points %d–%d / %d",
            batch_start + 1,
            batch_start + len(batch),
            total,
        )

    return total


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def verify(client: QdrantClient, collection_name: str) -> None:
    """Print a concise post-ingestion summary for quick sanity-checking."""
    info = client.get_collection(collection_name)
    vectors_cfg = info.config.params.vectors  # type: ignore[union-attr]

    point_count = info.points_count
    dim = vectors_cfg.size
    distance = vectors_cfg.distance.value

    logger.info("─" * 60)
    logger.info("✓ Collection : %s", collection_name)
    logger.info("✓ Points     : %s", point_count)
    logger.info("✓ Dimensions : %d", dim)
    logger.info("✓ Distance   : %s", distance)
    logger.info("─" * 60)

    # Spot-check: retrieve 1 point and confirm payload structure
    results = client.query_points(
        collection_name=collection_name,
        query=[0.0] * dim,        # zero vector — returns any nearest neighbours
        limit=1,
        with_payload=True,
    )
    if results.points:
        sample = results.points[0]
        logger.info(
            "Sample point id=%s  payload_keys=%s",
            sample.id,
            list((sample.payload or {}).keys()),
        )
        text_preview = (sample.payload or {}).get("text", "")[:120]
        logger.info("Sample text preview: %r", text_preview)
    else:
        logger.warning("No points returned from spot-check query.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest enterprise intelligence documents into Qdrant."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent / "sample_documents",
        help="Directory containing .txt document files (default: ./sample_documents/)",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant server URL (default: {QDRANT_URL})",
    )
    args = parser.parse_args()

    docs_dir: Path = args.docs_dir
    collection: str = args.collection

    if not docs_dir.is_dir():
        logger.error("Documents directory not found: %s", docs_dir)
        sys.exit(1)

    # -- Client --------------------------------------------------------------
    logger.info("Connecting to Qdrant at %s …", args.qdrant_url)
    client = QdrantClient(
        url=args.qdrant_url,
        api_key=QDRANT_API_KEY,
        timeout=30.0,
    )
    try:
        client.get_collections()   # lightweight connectivity check
    except Exception as exc:
        logger.error("Cannot reach Qdrant at %s — %s", args.qdrant_url, exc)
        sys.exit(1)
    logger.info("Qdrant connection OK.")

    # -- Collection ----------------------------------------------------------
    ensure_collection(client, collection)

    # -- Load documents ------------------------------------------------------
    logger.info("Loading documents from %s …", docs_dir)
    docs = load_documents(docs_dir)
    logger.info("Loaded %d document(s) total.", len(docs))

    # -- Embedding model -----------------------------------------------------
    logger.info("Initialising FastEmbed model: %s …", FASTEMBED_MODEL)
    model = TextEmbedding(model_name=FASTEMBED_MODEL)
    logger.info("Model ready.")

    # -- Ingest --------------------------------------------------------------
    total_points = ingest(client, docs, model, collection)

    # -- Verify --------------------------------------------------------------
    verify(client, collection)
    logger.info("Ingestion complete. %d point(s) in collection '%s'.", total_points, collection)


if __name__ == "__main__":
    main()
