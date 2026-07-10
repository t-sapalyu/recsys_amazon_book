import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

import db_utility
import recommend
from db_utility import Interaction
from model_loader import load_artifacts
from schemas import (
    BecauseYouLikedResponse, HealthResponse, ItemOut, LatencyStats,
    RecommendationResponse, RecommendedItem, SameAuthorResponse, SimilarAuthorsResponse,
)

SAMPLE_MODE = os.getenv("SAMPLE_MODE", "true").lower() == "true"
APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
CHECKPOINT_ROOT = APP_ROOT / ("checkpoints_sample" if SAMPLE_MODE else "checkpoints")

# Rolling window for the average-response-time reporting required in the README.
_LATENCY_WINDOW = deque(maxlen=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts(CHECKPOINT_ROOT)

    db = next(db_utility.get_db())
    try:
        max_ts = db.query(func.max(Interaction.timestamp)).filter(Interaction.split == "train").scalar()
        app.state.train_max_ts = max_ts

        # Precomputed once at startup, not per-request: every demo user's
        # embedding (for "people like you also like") and every author's
        # centroid embedding (for "readers who liked this also read"). Both
        # are cheap at this catalog/user-pool size and would be wasteful to
        # recompute on every call.
        app.state.user_embeddings = recommend.compute_all_user_embeddings(db, app.state.artifacts)
        app.state.author_centroids, app.state.author_items = recommend.compute_author_centroids(db, app.state.artifacts)
        print(f"Precomputed {len(app.state.user_embeddings)} user embeddings, {len(app.state.author_centroids)} author centroids.")
    finally:
        db.close()

    yield


app = FastAPI(title="Amazon Books Recommender API", lifespan=lifespan)


@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if request.url.path.startswith(("/recommendations", "/items", "/users")):
        _LATENCY_WINDOW.append(elapsed_ms)
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


def _to_item_out(item) -> ItemOut:
    return ItemOut(
        parent_asin=item.parent_asin, title=item.title, author=item.author, category=item.category,
        image_url=item.image_url, average_rating=item.average_rating, price=item.price,
    )


def _to_recs(scored_items) -> list:
    return [RecommendedItem(item=_to_item_out(item), score=score) for item, score in scored_items]


# --- Health & ops ---

@app.get("/health", response_model=HealthResponse)
async def health(db: Session = Depends(db_utility.get_db)):
    return HealthResponse(
        status="healthy",
        model_loaded=hasattr(app.state, "artifacts"),
        db_connected=db_utility.test_db_connection(),
    )


@app.get("/metrics", response_model=LatencyStats)
async def metrics():
    if not _LATENCY_WINDOW:
        return LatencyStats(count=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0)
    arr = np.array(_LATENCY_WINDOW)
    return LatencyStats(
        count=len(arr), avg_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)), p95_ms=float(np.percentile(arr, 95)),
    )


# --- Users (demo login) ---

@app.get("/users")
async def list_users(limit: int = Query(300, ge=1, le=1000), db: Session = Depends(db_utility.get_db)):
    return {"users": db_utility.list_demo_users(db, limit=limit)}


# --- Items ---

@app.get("/items/{item_id}", response_model=ItemOut)
async def get_item(item_id: str, db: Session = Depends(db_utility.get_db)):
    item = db_utility.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    return _to_item_out(item)


@app.get("/items/{item_id}/similar", response_model=RecommendationResponse)
async def similar_items(
    item_id: str, k: int = Query(10, ge=1, le=50), temperature: float = Query(0.3, gt=0),
    db: Session = Depends(db_utility.get_db),
):
    item = db_utility.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    scored = recommend.similar_items(db, app.state.artifacts, item_id, k=k, temperature=temperature)
    return RecommendationResponse(source_model="Similar items (Two-tower cosine)", items=_to_recs(scored))


@app.get("/items/{item_id}/same-author", response_model=SameAuthorResponse)
async def same_author(
    item_id: str, k: int = Query(10, ge=1, le=50), db: Session = Depends(db_utility.get_db),
):
    item = db_utility.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    author, scored = recommend.same_author(db, item_id, k=k)
    if author is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' has no listed author")
    return SameAuthorResponse(source_model="More by this author", author=author, items=_to_recs(scored))


@app.get("/items/{item_id}/similar-authors", response_model=SimilarAuthorsResponse)
async def similar_authors(
    item_id: str, k: int = Query(10, ge=1, le=50), temperature: float = Query(0.4, gt=0),
    db: Session = Depends(db_utility.get_db),
):
    item = db_utility.get_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")
    if not item.author:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' has no listed author")
    scored, similar = recommend.similar_authors_books(
        db, app.state.author_centroids, app.state.author_items, item.author, k=k, temperature=temperature,
    )
    if not scored:
        raise HTTPException(status_code=404, detail=f"No similar authors found for '{item.author}'")
    return SimilarAuthorsResponse(
        source_model="Readers who liked this also read (author-centroid cosine)",
        author=item.author, similar_authors=similar, items=_to_recs(scored),
    )


# --- Recommendations ---

@app.get("/recommendations/popular", response_model=RecommendationResponse)
async def popular(k: int = Query(10, ge=1, le=50), temperature: float = Query(0.5, gt=0), db: Session = Depends(db_utility.get_db)):
    scored = recommend.popular_recommendations(db, k=k, temperature=temperature)
    return RecommendationResponse(source_model="Popular right now", items=_to_recs(scored))


@app.get("/recommendations/personalized", response_model=RecommendationResponse)
async def personalized(
    user_id: str = Query(None), k: int = Query(10, ge=1, le=50), temperature: float = Query(0.5, gt=0),
    db: Session = Depends(db_utility.get_db),
):
    """Homepage. Falls back to Popularity for unknown/missing user_id, per project.md."""
    if user_id:
        user = db.get(db_utility.User, user_id)
        if user is not None:
            scored = recommend.personalized_recommendations(
                db, app.state.artifacts, user_id, app.state.train_max_ts, k=k, temperature=temperature,
            )
            if scored:
                return RecommendationResponse(source_model="Two-tower + LightGBM", items=_to_recs(scored))

    scored = recommend.popular_recommendations(db, k=k, temperature=temperature)
    return RecommendationResponse(source_model="Popular right now (fallback)", items=_to_recs(scored))


@app.get("/users/{user_id}/because-you-liked", response_model=BecauseYouLikedResponse)
async def because_you_liked(
    user_id: str, k: int = Query(10, ge=1, le=50), temperature: float = Query(0.3, gt=0),
    db: Session = Depends(db_utility.get_db),
):
    because_item, scored = recommend.because_you_liked(db, app.state.artifacts, user_id, k=k, temperature=temperature)
    if because_item is None:
        raise HTTPException(status_code=404, detail=f"No interaction history found for user '{user_id}'")
    return BecauseYouLikedResponse(
        source_model="Because you liked... (Two-tower cosine)",
        because_of=_to_item_out(because_item),
        items=_to_recs(scored),
    )


@app.get("/users/{user_id}/similar-users-liked", response_model=RecommendationResponse)
async def similar_users_liked(
    user_id: str, k: int = Query(10, ge=1, le=50), temperature: float = Query(0.5, gt=0),
    db: Session = Depends(db_utility.get_db),
):
    """"People like you also like" — see recommend.similar_users_liked for the
    user-user similarity logic; uses the embeddings precomputed at startup."""
    if db.get(db_utility.User, user_id) is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    scored = recommend.similar_users_liked(db, app.state.user_embeddings, user_id, k=k, temperature=temperature)
    if not scored:
        raise HTTPException(status_code=404, detail=f"Not enough data to find similar users for '{user_id}'")
    return RecommendationResponse(source_model="People like you also like (user-user similarity)", items=_to_recs(scored))
