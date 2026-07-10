import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

import db_utility
import recommend
from db_utility import Interaction
from model_loader import load_artifacts
from schemas import (
    BecauseYouLikedResponse, CartActionRequest, CartItemOut, CartResponse, CategoryListResponse,
    CategoryPreferenceRequest, CheckoutRequest, CheckoutResponse, DiscoverResponse, EventLogItem,
    EventLogResponse, EventRequest, EventResponse, HealthResponse, ItemOut, LatencyStats, RatingRequest,
    RecommendationResponse, RecommendedItem, SameAuthorResponse, SignupRequest, SimilarAuthorsResponse,
    WishlistActionRequest, WishlistItemOut, WishlistResponse,
)

APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
CHECKPOINT_ROOT = APP_ROOT / "checkpoints"

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


# --- Categories & guest discovery ---

@app.get("/categories", response_model=CategoryListResponse)
async def list_categories(db: Session = Depends(db_utility.get_db)):
    return CategoryListResponse(categories=db_utility.list_categories(db))


@app.get("/recommendations/discover", response_model=DiscoverResponse)
async def discover(
    trending_k: int = Query(20, ge=1, le=50), random_k: int = Query(10, ge=1, le=50),
    temperature: float = Query(0.5, gt=0), db: Session = Depends(db_utility.get_db),
):
    """Guest homepage: trending books + a handful of random books from
    categories outside that trending set, from a single snapshot so the two
    sections are always consistent with each other (see `recommend.discover_for_guest`)."""
    trending, discover_items = recommend.discover_for_guest(db, trending_k=trending_k, random_k=random_k, temperature=temperature)
    return DiscoverResponse(
        trending_source_model="Popular right now",
        trending=_to_recs(trending),
        discover_source_model="Something different (other categories)",
        discover=_to_recs(discover_items),
    )


# --- Signup / onboarding ---

@app.post("/users/signup", status_code=201)
async def signup(payload: SignupRequest, db: Session = Depends(db_utility.get_db)):
    user = db_utility.create_user(db, payload.user_id)
    if user is None:
        raise HTTPException(status_code=409, detail=f"User '{payload.user_id}' already exists")
    return {"user_id": user.user_id}


@app.post("/users/{user_id}/preferences/categories")
async def set_category_preferences(user_id: str, payload: CategoryPreferenceRequest, db: Session = Depends(db_utility.get_db)):
    if db.get(db_utility.User, user_id) is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    if len(payload.categories) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 categories")
    db_utility.set_category_preferences(db, user_id, payload.categories)
    return {"status": "ok"}


@app.get("/onboarding-suggestions", response_model=RecommendationResponse)
async def onboarding_suggestions_by_categories(
    categories: List[str] = Query(...), k: int = Query(12, ge=1, le=50), db: Session = Depends(db_utility.get_db),
):
    """Signup step 2, before the account exists yet — takes the categories
    the user just picked directly, rather than reading persisted preferences
    for a user_id. Account creation (and persisting those preferences) is
    deferred to the Finish step, so an abandoned signup never leaves behind
    a half-created user."""
    scored = recommend.onboarding_suggestions(db, categories, k=k)
    return RecommendationResponse(source_model="Onboarding suggestions (category popularity)", items=_to_recs(scored))


@app.get("/users/{user_id}/onboarding-suggestions", response_model=RecommendationResponse)
async def onboarding_suggestions(user_id: str, k: int = Query(12, ge=1, le=50), db: Session = Depends(db_utility.get_db)):
    categories = db_utility.get_category_preferences(db, user_id)
    if not categories:
        raise HTTPException(status_code=404, detail=f"No category preferences set for user '{user_id}'")
    scored = recommend.onboarding_suggestions(db, categories, k=k)
    return RecommendationResponse(source_model="Onboarding suggestions (category popularity)", items=_to_recs(scored))


# --- Implicit events ---

@app.post("/events", response_model=EventResponse)
async def record_event(payload: EventRequest, db: Session = Depends(db_utility.get_db)):
    """Generic logger for every impression/view/positive/negative signal —
    see `db_utility.EVENT_SIGNALS` for the full catalog. Never trusts the
    client for `signal`; looks it up server-side from `event_type`."""
    if payload.event_type not in db_utility.EVENT_SIGNALS:
        raise HTTPException(status_code=400, detail=f"Unknown event_type '{payload.event_type}'")
    signal = db_utility.EVENT_SIGNALS[payload.event_type]
    db_utility.log_event(
        db, session_id=payload.session_id, user_id=payload.user_id, parent_asin=payload.parent_asin,
        event_type=payload.event_type, signal=signal, position=payload.position, value=payload.value,
    )

    derived = None
    if payload.event_type == "impression" and payload.parent_asin and payload.position is not None:
        derived = recommend.maybe_log_impression_no_click(
            db, session_id=payload.session_id, user_id=payload.user_id,
            parent_asin=payload.parent_asin, position=payload.position,
        )
    elif payload.event_type in ("checkout", "onboarding_like") and payload.user_id and payload.parent_asin:
        db_utility.backfill_interaction(db, payload.user_id, payload.parent_asin)

    return EventResponse(status="ok", derived_negative_event=derived)


# --- Wishlist ---

@app.get("/users/{user_id}/wishlist", response_model=WishlistResponse)
async def get_wishlist(user_id: str, session_id: str = Query(...), db: Session = Depends(db_utility.get_db)):
    rows = db_utility.bump_wishlist_views(db, user_id)
    for row in rows:
        recommend.maybe_log_repeated_wishlist_no_purchase(db, session_id=session_id, user_id=user_id, row=row)
    items_by_id = db_utility.get_items(db, [r.parent_asin for r in rows])
    return WishlistResponse(items=[
        WishlistItemOut(item=_to_item_out(items_by_id[r.parent_asin]), added_at=r.added_at, view_count=r.view_count)
        for r in rows if r.parent_asin in items_by_id
    ])


@app.post("/users/{user_id}/wishlist", status_code=201)
async def add_wishlist_item(user_id: str, payload: WishlistActionRequest, db: Session = Depends(db_utility.get_db)):
    if db.get(db_utility.User, user_id) is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    if db_utility.get_item(db, payload.parent_asin) is None:
        raise HTTPException(status_code=404, detail=f"Item '{payload.parent_asin}' not found")
    db_utility.add_to_wishlist(db, user_id, payload.parent_asin)
    db_utility.log_event(
        db, session_id=payload.session_id, user_id=user_id, parent_asin=payload.parent_asin,
        event_type="wishlist_add", signal="positive",
    )
    return {"status": "ok"}


@app.delete("/users/{user_id}/wishlist/{item_id}")
async def remove_wishlist_item(user_id: str, item_id: str, session_id: str = Query(...), db: Session = Depends(db_utility.get_db)):
    removed = db_utility.remove_from_wishlist(db, user_id, item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Item not in wishlist")
    db_utility.log_event(
        db, session_id=session_id, user_id=user_id, parent_asin=item_id,
        event_type="wishlist_remove", signal="negative",
    )
    return {"status": "ok"}


# --- Cart & checkout ---

@app.get("/users/{user_id}/cart", response_model=CartResponse)
async def get_cart(user_id: str, db: Session = Depends(db_utility.get_db)):
    rows = db_utility.get_cart(db, user_id)
    items_by_id = db_utility.get_items(db, [r.parent_asin for r in rows])
    out_items = [
        CartItemOut(item=_to_item_out(items_by_id[r.parent_asin]), added_at=r.added_at)
        for r in rows if r.parent_asin in items_by_id
    ]
    total_price = sum((items_by_id[r.parent_asin].price or 0.0) for r in rows if r.parent_asin in items_by_id)
    return CartResponse(items=out_items, total_price=total_price)


@app.post("/users/{user_id}/cart", status_code=201)
async def add_cart_item(user_id: str, payload: CartActionRequest, db: Session = Depends(db_utility.get_db)):
    if db.get(db_utility.User, user_id) is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
    if db_utility.get_item(db, payload.parent_asin) is None:
        raise HTTPException(status_code=404, detail=f"Item '{payload.parent_asin}' not found")
    db_utility.add_to_cart(db, user_id, payload.parent_asin)
    db_utility.log_event(
        db, session_id=payload.session_id, user_id=user_id, parent_asin=payload.parent_asin,
        event_type="cart_add", signal="positive",
    )
    return {"status": "ok"}


@app.delete("/users/{user_id}/cart/{item_id}")
async def remove_cart_item(user_id: str, item_id: str, session_id: str = Query(...), db: Session = Depends(db_utility.get_db)):
    removed = db_utility.remove_from_cart(db, user_id, item_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Item not in cart")
    db_utility.log_event(
        db, session_id=session_id, user_id=user_id, parent_asin=item_id,
        event_type="cart_remove", signal="negative",
    )
    return {"status": "ok"}


@app.post("/users/{user_id}/checkout", response_model=CheckoutResponse)
async def checkout(user_id: str, payload: CheckoutRequest, db: Session = Depends(db_utility.get_db)):
    """Simulated checkout, no real payment — logs a `checkout` event and
    backfills a live `Interaction` per cart item, then empties the cart."""
    rows = db_utility.get_cart(db, user_id)
    if not rows:
        raise HTTPException(status_code=400, detail="Cart is empty")
    items_by_id = db_utility.get_items(db, [r.parent_asin for r in rows])
    # Built before `clear_cart` — its bulk delete() expires these ORM
    # instances, so `row.parent_asin` isn't safe to read afterward.
    checked_out_items = [_to_item_out(items_by_id[r.parent_asin]) for r in rows if r.parent_asin in items_by_id]
    for row in rows:
        db_utility.backfill_interaction(db, user_id, row.parent_asin)
        db_utility.log_event(
            db, session_id=payload.session_id, user_id=user_id, parent_asin=row.parent_asin,
            event_type="checkout", signal="positive",
        )
    db_utility.clear_cart(db, user_id)
    return CheckoutResponse(status="ok", checked_out_items=checked_out_items)


# --- Activity log (viewing recorded positive/negative signals) ---

@app.get("/users/{user_id}/events", response_model=EventLogResponse)
async def get_user_events(user_id: str, limit: int = Query(200, ge=1, le=1000), db: Session = Depends(db_utility.get_db)):
    rows = db_utility.get_user_events(db, user_id, limit=limit)
    items_by_id = db_utility.get_items(db, [r.parent_asin for r in rows if r.parent_asin])
    return EventLogResponse(events=[
        EventLogItem(
            event_type=r.event_type, signal=r.signal, reason=db_utility.EVENT_DESCRIPTIONS.get(r.event_type),
            parent_asin=r.parent_asin,
            item_title=(items_by_id[r.parent_asin].title if r.parent_asin in items_by_id else None),
            position=r.position, value=r.value, created_at=r.created_at,
        )
        for r in rows
    ])


# --- Explicit ratings (double as a positive/negative implicit signal) ---

@app.post("/users/{user_id}/ratings")
async def submit_rating(user_id: str, payload: RatingRequest, db: Session = Depends(db_utility.get_db)):
    event_type = "high_rating" if payload.rating >= 4 else "low_rating"
    signal = db_utility.EVENT_SIGNALS[event_type]
    db_utility.log_event(
        db, session_id=payload.session_id, user_id=user_id, parent_asin=payload.parent_asin,
        event_type=event_type, signal=signal, value=payload.rating,
    )
    return {"status": "ok"}
