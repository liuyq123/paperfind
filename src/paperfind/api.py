"""FastAPI application for Paperfind."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Paperfind API", version="0.2.0")


# =============================================================================
# Pydantic Models
# =============================================================================


class PaperResult(BaseModel):
    """A paper recommendation or search result."""

    doi: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    score: float
    source: Optional[str] = None
    created_date: Optional[str] = None
    similar_to: Optional[str] = None


class RecommendResponse(BaseModel):
    """Response for recommend endpoint."""

    recommendations: List[PaperResult]
    count: int
    reranked: bool


class CollectionInfo(BaseModel):
    """A Zotero collection."""

    name: str
    key: str
    num_items: int


class CollectionsResponse(BaseModel):
    """Response for collections endpoint."""

    collections: List[CollectionInfo]
    count: int


class SearchResult(BaseModel):
    """A search result."""

    doi: Optional[str] = None
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    results: List[SearchResult]
    count: int
    query: str
    rag_answer: Optional[str] = None


class JobResponse(BaseModel):
    """Response when starting a background job."""

    job_id: str
    status: str
    job_type: str


class JobStatusResponse(BaseModel):
    """Response for job status endpoint."""

    job_id: str
    status: str
    job_type: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class PaperInfo(BaseModel):
    """A paper from the database."""

    doi: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    source: str
    created_date: Optional[str] = None


class PapersResponse(BaseModel):
    """Response for papers endpoint."""

    papers: List[PaperInfo]
    count: int
    total: int
    limit: int
    offset: int


# =============================================================================
# Job Store (in-memory)
# =============================================================================


class JobState:
    """State for a background job."""

    def __init__(self, job_id: str, job_type: str):
        self.job_id = job_id
        self.job_type = job_type
        self.status = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None


# In-memory job store
_jobs: Dict[str, JobState] = {}


def _create_job(job_type: str) -> JobState:
    """Create a new job and add it to the store."""
    job_id = str(uuid.uuid4())
    job = JobState(job_id, job_type)
    _jobs[job_id] = job
    return job


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    """Get the status of a background job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        job_type=job.job_type,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@app.get("/collections", response_model=CollectionsResponse)
def list_collections() -> CollectionsResponse:
    """List available Zotero collections."""
    from paperfind.config import ZOTERO_API_KEY, ZOTERO_LIBRARY_TYPE, ZOTERO_USER_ID
    from paperfind.fetchers.zotero.api import fetch_collections

    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        raise HTTPException(
            status_code=503,
            detail="Zotero credentials not configured. Set ZOTERO_API_KEY and ZOTERO_USER_ID.",
        )

    try:
        collections = fetch_collections(
            library_id=ZOTERO_USER_ID,
            api_key=ZOTERO_API_KEY,
            library_type=ZOTERO_LIBRARY_TYPE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch collections: {e}")

    results = [
        CollectionInfo(
            name=c["data"]["name"],
            key=c["data"]["key"],
            num_items=c["meta"].get("numItems", 0),
        )
        for c in collections
    ]

    return CollectionsResponse(collections=results, count=len(results))


@app.get("/search", response_model=SearchResponse)
def search_papers(
    query: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    source: str = Query(default="daily_papers", description="Data source: daily_papers or zotero"),
    scores: bool = Query(default=False, description="Include similarity scores"),
    rag: bool = Query(default=False, description="Use RAG to answer the query"),
) -> SearchResponse:
    """Semantic search across papers."""
    from paperfind.documents import extract_title_and_abstract
    from paperfind.search.search import rag_query, search, search_with_scores
    from paperfind.search.utils import check_vector_store

    if source not in ("daily_papers", "zotero"):
        raise HTTPException(status_code=400, detail="Source must be 'daily_papers' or 'zotero'")

    if not check_vector_store(source):
        raise HTTPException(
            status_code=503,
            detail=f"Vector store for '{source}' not found. Run appropriate sync/fetch first.",
        )

    rag_answer = None

    if rag:
        try:
            rag_answer = rag_query(query, k=k, source=source)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

    if scores:
        try:
            docs_with_scores = search_with_scores(query, k=k, source=source)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")

        results = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata or {}
            title, abstract = extract_title_and_abstract(doc)

            results.append(
                SearchResult(
                    doi=metadata.get("doi"),
                    title=title,
                    authors=metadata.get("authors"),
                    abstract=abstract,
                    source=metadata.get("source"),
                    score=score,
                )
            )
    else:
        try:
            docs = search(query, k=k, source=source)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")

        results = []
        for doc in docs:
            metadata = doc.metadata or {}
            title, abstract = extract_title_and_abstract(doc)

            results.append(
                SearchResult(
                    doi=metadata.get("doi"),
                    title=title,
                    authors=metadata.get("authors"),
                    abstract=abstract,
                    source=metadata.get("source"),
                    score=None,
                )
            )

    return SearchResponse(
        results=results,
        count=len(results),
        query=query,
        rag_answer=rag_answer,
    )


@app.get("/papers", response_model=PapersResponse)
def list_papers(
    limit: int = Query(default=50, ge=1, le=500, description="Number of papers to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    source: Optional[str] = Query(default=None, description="Filter by source"),
) -> PapersResponse:
    """List fetched papers with pagination."""
    from paperfind.db import DAILY_SCHEMA, get_conn, placeholder, qualify_table

    try:
        conn = get_conn(DAILY_SCHEMA)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")

    try:
        cursor = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "works")
        ph = placeholder()

        # Get total count
        count_sql = f"SELECT COUNT(*) AS total FROM {table}"
        params: List[Any] = []
        if source:
            count_sql += f" WHERE source = {ph}"
            params.append(source)

        cursor.execute(count_sql, params)
        total_row = cursor.fetchone()
        total = total_row["total"] if total_row is not None else 0

        # Get papers
        select_sql = (
            f"SELECT doi, title, authors, abstract, source, created_date "
            f"FROM {table}"
        )
        params = []  # type: List[Any]
        if source:
            select_sql += f" WHERE source = {ph}"
            params.append(source)

        select_sql += f" ORDER BY created_date DESC LIMIT {ph} OFFSET {ph}"
        params.extend([limit, offset])

        cursor.execute(select_sql, params)
        rows = cursor.fetchall()

        papers = [
            PaperInfo(
                doi=row["doi"],
                title=row["title"],
                authors=row["authors"],
                abstract=row["abstract"],
                source=row["source"],
                created_date=str(row["created_date"]) if row["created_date"] else None,
            )
            for row in rows
        ]

        return PapersResponse(
            papers=papers,
            count=len(papers),
            total=total,
            limit=limit,
            offset=offset,
        )
    finally:
        conn.close()


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    k: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    collection: Optional[str] = Query(default=None, description="Zotero collection name"),
    rerank: bool = Query(default=True, description="Enable cross-encoder reranking"),
    rerank_candidates: int = Query(default=50, ge=1, description="Candidate pool for reranking"),
) -> RecommendResponse:
    """Get paper recommendations based on your Zotero library."""
    from paperfind.documents import extract_title_and_abstract
    from paperfind.search.recommend import get_recommendations
    from paperfind.search.utils import check_vector_store

    if not check_vector_store():
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Run 'paperfind fetch --rebuild-vectors' first.",
        )

    recommendations, rerank_used = get_recommendations(
        k=k,
        collection=collection,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        return_rerank_used=True,
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found. Make sure you have synced your Zotero library.",
        )

    results = []
    for doi, (score, doc, zotero_title) in recommendations:
        metadata = doc.metadata or {}
        title, abstract = extract_title_and_abstract(doc)

        results.append(
            PaperResult(
                doi=doi,
                title=title,
                authors=metadata.get("authors"),
                abstract=abstract,
                score=score,
                source=metadata.get("source"),
                created_date=metadata.get("created_date"),
                similar_to=zotero_title,
            )
        )

    return RecommendResponse(
        recommendations=results,
        count=len(results),
        reranked=rerank_used,
    )


# =============================================================================
# Background Task Endpoints
# =============================================================================


def _run_sync(job: JobState) -> None:
    """Background task to run Zotero sync."""
    from paperfind.fetchers.zotero.sync import sync_library

    job.status = "running"
    try:
        num_items = sync_library()
        job.result = {
            "num_items": num_items,
            "message": f"Synced {num_items} items from Zotero library",
        }
        job.status = "completed"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
    finally:
        job.completed_at = datetime.now(timezone.utc).isoformat()


def _run_fetch(
    job: JobState,
    days: int,
    sources: Optional[List[str]],
    rebuild_vectors: bool,
) -> None:
    """Background task to run paper fetching."""
    from paperfind.fetchers.fetch_papers import fetch_all
    from paperfind.fetchers.vector import rebuild_vectors as do_rebuild_vectors

    job.status = "running"
    try:
        counts, dois = fetch_all(days=days, sources=sources)

        if rebuild_vectors:
            do_rebuild_vectors()

        job.result = {
            "counts": counts,
            "total": sum(counts.values()),
            "dois_fetched": len(dois),
            "vectors_rebuilt": rebuild_vectors,
        }
        job.status = "completed"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
    finally:
        job.completed_at = datetime.now(timezone.utc).isoformat()


def _run_embed(job: JobState, collection: str, force: bool) -> None:
    """Background task to embed a Zotero collection."""
    from paperfind.fetchers.zotero.sync import embed_collection

    job.status = "running"
    try:
        num_embedded = embed_collection(collection, force=force)
        job.result = {
            "collection": collection,
            "num_embedded": num_embedded,
            "force": force,
            "message": f"Embedded {num_embedded} items from collection '{collection}'",
        }
        job.status = "completed"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"
    finally:
        job.completed_at = datetime.now(timezone.utc).isoformat()


@app.post("/sync", response_model=JobResponse)
def sync(
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Trigger Zotero library sync as a background job.

    Syncs your entire Zotero library. To embed a specific collection
    for semantic search, use the embed command after syncing.
    """
    from paperfind.config import ZOTERO_API_KEY, ZOTERO_USER_ID

    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        raise HTTPException(
            status_code=503,
            detail="Zotero credentials not configured. Set ZOTERO_API_KEY and ZOTERO_USER_ID.",
        )

    job = _create_job("sync")
    background_tasks.add_task(_run_sync, job)

    return JobResponse(job_id=job.job_id, status=job.status, job_type=job.job_type)


@app.post("/embed", response_model=JobResponse)
def embed(
    background_tasks: BackgroundTasks,
    collection: str = Query(..., min_length=1, description="Collection name or key to embed"),
    force: bool = Query(default=False, description="Re-embed all items (ignore existing)"),
) -> JobResponse:
    """Embed a Zotero collection for semantic search.

    Creates vector embeddings for items in the specified collection.
    Items already embedded are skipped unless force=True.
    """
    job = _create_job("embed")
    background_tasks.add_task(_run_embed, job, collection, force)

    return JobResponse(job_id=job.job_id, status=job.status, job_type=job.job_type)


@app.post("/fetch", response_model=JobResponse)
def fetch(
    background_tasks: BackgroundTasks,
    days: int = Query(default=1, ge=1, le=30, description="Number of days to fetch"),
    sources: Optional[str] = Query(
        default=None, description="Comma-separated sources: crossref,biorxiv,medrxiv,arxiv"
    ),
    rebuild_vectors: bool = Query(default=False, description="Rebuild vectors after fetching"),
) -> JobResponse:
    """Trigger paper fetching as a background job."""
    source_list = None
    if sources:
        source_list = [s.strip() for s in sources.split(",")]
        valid_sources = {"crossref", "biorxiv", "medrxiv", "arxiv"}
        invalid = set(source_list) - valid_sources
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sources: {invalid}. Valid: {valid_sources}",
            )

    job = _create_job("fetch")
    background_tasks.add_task(_run_fetch, job, days, source_list, rebuild_vectors)

    return JobResponse(job_id=job.job_id, status=job.status, job_type=job.job_type)
