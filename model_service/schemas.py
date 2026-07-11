"""Pydantic response models for the recommendation API."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ItemOut(BaseModel):
    parent_asin: str
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    average_rating: Optional[float] = None
    price: Optional[float] = None


class RecommendedItem(BaseModel):
    item: ItemOut
    score: float


class RecommendationResponse(BaseModel):
    source_model: str  # e.g. "Popular right now", "Two-tower + LightGBM", "Similar items (Two-tower cosine)"
    items: List[RecommendedItem]


class BecauseYouLikedResponse(BaseModel):
    source_model: str
    because_of: ItemOut
    items: List[RecommendedItem]


class SameAuthorResponse(BaseModel):
    source_model: str
    author: str
    items: List[RecommendedItem]


class SimilarAuthorsResponse(BaseModel):
    source_model: str
    author: str
    similar_authors: List[str]
    items: List[RecommendedItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool


class LatencyStats(BaseModel):
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float


# --- Categories & guest discovery ---

class CategoryListResponse(BaseModel):
    categories: List[str]


class DiscoverResponse(BaseModel):
    trending_source_model: str
    trending: List[RecommendedItem]
    discover_source_model: str
    discover: List[RecommendedItem]


# --- Signup / onboarding ---

class SignupRequest(BaseModel):
    user_id: str


class CategoryPreferenceRequest(BaseModel):
    categories: List[str]


# --- Implicit events ---

class EventRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    parent_asin: Optional[str] = None
    event_type: str
    position: Optional[int] = None
    value: Optional[float] = None


class EventResponse(BaseModel):
    status: str
    derived_negative_event: Optional[str] = None


# --- Wishlist / cart / checkout / ratings ---

class WishlistActionRequest(BaseModel):
    parent_asin: str
    session_id: str


class WishlistItemOut(BaseModel):
    item: ItemOut
    added_at: datetime
    view_count: int


class WishlistResponse(BaseModel):
    items: List[WishlistItemOut]


class CartActionRequest(BaseModel):
    parent_asin: str
    session_id: str


class CartItemOut(BaseModel):
    item: ItemOut
    added_at: datetime


class CartResponse(BaseModel):
    items: List[CartItemOut]
    total_price: float


class CheckoutRequest(BaseModel):
    session_id: str


class CheckoutResponse(BaseModel):
    status: str
    checked_out_items: List[ItemOut]


class RatingRequest(BaseModel):
    parent_asin: str
    rating: float
    session_id: str


# --- Activity log (viewing recorded positive/negative signals) ---

class EventLogItem(BaseModel):
    event_type: str
    signal: str
    reason: Optional[str] = None
    parent_asin: Optional[str] = None
    item_title: Optional[str] = None
    position: Optional[int] = None
    value: Optional[float] = None
    created_at: datetime


class EventLogResponse(BaseModel):
    events: List[EventLogItem]


# --- Signal rules catalog (static reference, not per-user events) ---

class SignalRule(BaseModel):
    event_type: str
    signal: str
    kind: str
    reason: str


class SignalRulesResponse(BaseModel):
    rules: List[SignalRule]
