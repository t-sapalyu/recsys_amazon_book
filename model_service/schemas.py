"""Pydantic response models for the recommendation API."""

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
