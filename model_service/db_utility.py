"""Database schema and access helpers for the recommender webapp.

Schema: user profiles, interaction history, item metadata (title, image,
category) — per project.md's PostgreSQL requirement. Item-level ranking
features (item_popularity, item_age_days) are precomputed once at seed time
(prestart.py) and stored directly on `items`, since they're static; user-level
ranking features are recomputed per-request from `interactions` (Section
"Online" in project.md — the API fetches user history from PostgreSQL at
request time), since a user's history changes as new interactions accrue.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set!")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# --- Table definitions ---


class Item(Base):
    """Item metadata + precomputed static ranking features (see module docstring)."""
    __tablename__ = "items"

    parent_asin = Column(String, primary_key=True)
    title = Column(String)
    author = Column(String, index=True)  # may be null — ~5% of items have no listed author
    category = Column(String)
    image_url = Column(String)
    average_rating = Column(Float)  # raw value, for display — may be null
    price = Column(Float)           # raw value, for display — may be null

    # Precomputed at seed time from train — identical definitions to
    # notebook 05's ranking features, so the ranker sees the same features
    # at serving time it was trained on. Never null (fillna applied at seed time).
    item_popularity = Column(Float)        # log1p(# positive train interactions)
    item_avg_rating_feature = Column(Float)  # average_rating, catalog-mean-imputed
    item_price_norm = Column(Float)        # log1p(price), median-imputed
    item_age_days = Column(Integer)        # days since first train interaction, as of seed time


class User(Base):
    """Demo-able users — those with test-period interactions (see prestart.py).

    Not a real auth system: the webapp's "log in" is a user_id picker over
    this table, simulating a logged-in session for the demo per project.md's
    "Homepage (logged in)" requirement.
    """
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)


class Interaction(Base):
    """A user-item interaction. `split` marks whether it's train-period or
    test-period history — the demo computes a user's live embedding from
    their *test*-period rows specifically, simulating "recent activity the
    model never trained on" per project.md's deployment-simulation note."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    parent_asin = Column(String, ForeignKey("items.parent_asin"), index=True, nullable=False)
    rating = Column(Float, nullable=False)
    label = Column(Integer, nullable=False)  # rating >= 4, same threshold as notebook 01
    timestamp = Column(DateTime, nullable=False)
    split = Column(String, nullable=False)   # "train" | "test"


# --- Database operations ---


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_db_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception:
        return False


def get_item(db: Session, parent_asin: str) -> "Item | None":
    return db.get(Item, parent_asin)


def get_items(db: Session, parent_asins: list) -> dict:
    """Batch item lookup -> {parent_asin: Item}, for rendering a list of recommendations at once."""
    rows = db.query(Item).filter(Item.parent_asin.in_(parent_asins)).all()
    return {row.parent_asin: row for row in rows}


def list_demo_users(db: Session, limit: int = 300) -> list:
    return [row.user_id for row in db.query(User.user_id).order_by(User.user_id).limit(limit).all()]


def get_user_interactions(db: Session, user_id: str, splits=None) -> list:
    """All interactions for one user, optionally filtered to specific splits (e.g. ["train", "test"])."""
    query = db.query(Interaction).filter(Interaction.user_id == user_id)
    if splits is not None:
        query = query.filter(Interaction.split.in_(splits))
    return query.order_by(Interaction.timestamp).all()


def get_popular_items(db: Session, limit: int = 20):
    """Same definition as notebook 02's PopularityRecommender: ranked by
    item_popularity (log1p of positive train interactions), which is
    precomputed and stored on `items` at seed time."""
    return (
        db.query(Item)
        .filter(Item.item_popularity.isnot(None))
        .order_by(Item.item_popularity.desc())
        .limit(limit)
        .all()
    )


def get_items_by_author(db: Session, author: str, exclude_asin: str = None, limit: int = 20):
    query = db.query(Item).filter(Item.author == author)
    if exclude_asin:
        query = query.filter(Item.parent_asin != exclude_asin)
    return query.order_by(Item.item_popularity.desc()).limit(limit).all()


def get_all_authored_items(db: Session):
    """(parent_asin, author) for every item with a listed author — used to
    precompute per-author centroid embeddings at API startup."""
    return db.query(Item.parent_asin, Item.author).filter(Item.author.isnot(None)).all()


def list_all_users(db: Session) -> list:
    """Every seeded demo user id — used to precompute all users' embeddings
    at API startup for "people like you also like"."""
    return [row.user_id for row in db.query(User.user_id).all()]
