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
from datetime import datetime, timedelta

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine, func, text
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
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


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


class UserCategoryPreference(Base):
    """A category a user picked at signup — seeds their onboarding book
    suggestions. A preference signal, distinct from an actual interaction."""
    __tablename__ = "user_category_preferences"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    category = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Wishlist(Base):
    """A user's saved-for-later books. `view_count` counts how many times
    the wishlist page has been loaded while this row was still present — the
    counter behind the `repeated_wishlist_no_purchase` negative signal (see
    `EVENT_SIGNALS` below)."""
    __tablename__ = "wishlist"
    __table_args__ = (UniqueConstraint("user_id", "parent_asin", name="uq_wishlist_user_item"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    parent_asin = Column(String, ForeignKey("items.parent_asin"), index=True, nullable=False)
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    view_count = Column(Integer, nullable=False, default=0)


class CartItem(Base):
    """A book sitting in a user's shopping cart, pending checkout."""
    __tablename__ = "cart_items"
    __table_args__ = (UniqueConstraint("user_id", "parent_asin", name="uq_cart_user_item"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    parent_asin = Column(String, ForeignKey("items.parent_asin"), index=True, nullable=False)
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class UserEvent(Base):
    """Generic append-only implicit-signal log. Every impression, view,
    positive action (wishlist/cart/checkout/like/rating), and negative signal
    (not-interested, dismiss, bounce, ...) is one row here, distinguished by
    `event_type`/`signal` (see `EVENT_SIGNALS`). `user_id` is nullable and
    `session_id` always set so guest (pre-signup) activity is still
    trackable. `position` is the card's 0-indexed rank within whichever
    recommendation row it was shown in — needed to scope
    `impression_no_click` to top-row impressions only."""
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=True)
    session_id = Column(String, index=True, nullable=False)
    parent_asin = Column(String, ForeignKey("items.parent_asin"), index=True, nullable=True)
    event_type = Column(String, nullable=False)
    signal = Column(String, nullable=False)  # "positive" | "negative" | "neutral"
    position = Column(Integer, nullable=True)
    value = Column(Float, nullable=True)  # e.g. the star rating for a rating event
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# Every implicit event type this webapp records, and which bucket it belongs
# to — used by `POST /events` to validate incoming events and assign their
# `signal` server-side (never trusting the client for this).
EVENT_SIGNALS = {
    # neutral: raw substrate other signals are derived from
    "impression": "neutral",
    "view": "neutral",
    # positive (recorded ways to like/want a book)
    "wishlist_add": "positive",
    "cart_add": "positive",
    "checkout": "positive",
    "like": "positive",
    "onboarding_like": "positive",
    "high_rating": "positive",
    # negative (the ten ways to record disinterest)
    "impression_no_click": "negative",
    "viewed_no_action": "negative",
    "quick_bounce": "negative",
    "not_interested": "negative",
    "dismiss_recommendation": "negative",
    "wishlist_remove": "negative",
    "cart_remove": "negative",
    "repeated_wishlist_no_purchase": "negative",
    "low_rating": "negative",
    "onboarding_skip": "negative",
}


# Human-readable explanation of the condition each event type represents —
# shown on the webapp's Activity Log page so "negative"/"positive" isn't
# just a bare label with no indication of what actually triggered it.
EVENT_DESCRIPTIONS = {
    "impression": "Card was shown in a recommendation row.",
    "view": "User opened this item's detail page.",
    "wishlist_add": "User added this book to their wishlist.",
    "cart_add": "User added this book to their cart.",
    "checkout": "User completed checkout with this book in their cart.",
    "like": "User tapped \U0001F44D on this book.",
    "onboarding_like": "User marked this book as liked during signup.",
    "high_rating": "User submitted a rating of 4 or 5 stars.",
    "impression_no_click": "Shown ≥10 times in the top row of a list, never opened.",
    "viewed_no_action": "Item was opened, but no positive action followed before leaving (dwell ≥5s).",
    "quick_bounce": "Item was opened and left again within 5 seconds, no action taken.",
    "not_interested": "User tapped \U0001F44E \"Not interested\".",
    "dismiss_recommendation": "User dismissed (✕) this card without opening it.",
    "wishlist_remove": "User removed this book from their wishlist.",
    "cart_remove": "User removed this book from their cart before checkout.",
    "repeated_wishlist_no_purchase": "Sat in wishlist ≥182 days with ≥10 views, never checked out.",
    "low_rating": "User submitted a rating below 4 stars.",
    "onboarding_skip": "Shown as a signup suggestion but not marked as liked.",
}


# "explicit" = the interaction's entire purpose is stating a preference (a
# rating widget, a thumbs up/down, an onboarding like/skip pick) — the user
# did nothing else with that click. "implicit" = preference is inferred as
# a side effect of an action whose primary purpose is something else
# (browsing, shopping, list management) — the classic implicit-vs-explicit
# feedback distinction in recommender systems (e.g. Hu, Koren & Volinsky,
# "Collaborative Filtering for Implicit Feedback Datasets").
EVENT_KIND = {
    "impression": "implicit",
    "view": "implicit",
    "wishlist_add": "implicit",
    "cart_add": "implicit",
    "checkout": "implicit",
    "like": "explicit",
    "onboarding_like": "explicit",
    "high_rating": "explicit",
    "impression_no_click": "implicit",
    "viewed_no_action": "implicit",
    "quick_bounce": "implicit",
    "not_interested": "explicit",
    "dismiss_recommendation": "implicit",
    "wishlist_remove": "implicit",
    "cart_remove": "implicit",
    "repeated_wishlist_no_purchase": "implicit",
    "low_rating": "explicit",
    "onboarding_skip": "explicit",
}


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
    """Most-recently-created first — otherwise the 300 pre-seeded demo users
    (see prestart.py) already fill an alphabetical `ORDER BY user_id LIMIT`,
    silently pushing every newly signed-up user off the end of the list."""
    return [row.user_id for row in db.query(User.user_id).order_by(User.created_at.desc()).limit(limit).all()]


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


# --- Categories (guest discovery + signup) ---


def list_categories(db: Session) -> list:
    """Distinct non-empty item categories, for the signup picker and the
    discover-homepage's "other categories" exclusion set."""
    rows = db.query(Item.category).filter(Item.category.isnot(None)).distinct().all()
    return sorted({r[0] for r in rows if r[0]})


def get_items_by_categories(db: Session, categories: list, limit: int = 50):
    return (
        db.query(Item)
        .filter(Item.category.in_(categories))
        .order_by(Item.item_popularity.desc())
        .limit(limit)
        .all()
    )


def get_user_events(db: Session, user_id: str, limit: int = 200) -> list:
    """A user's logged implicit events, most recent first — backs the
    activity-log page that shows what's actually landing in `user_events`
    for a given (real or demo) user."""
    return (
        db.query(UserEvent)
        .filter(UserEvent.user_id == user_id)
        .order_by(UserEvent.created_at.desc())
        .limit(limit)
        .all()
    )


def get_random_items_excluding_categories(db: Session, exclude_categories: set, limit: int = 10):
    """True-random pick from items whose category is outside
    `exclude_categories` — the guest homepage's "something different"
    section. `func.random()` works on both Postgres and SQLite."""
    query = db.query(Item).filter(Item.category.isnot(None))
    if exclude_categories:
        query = query.filter(Item.category.notin_(exclude_categories))
    return query.order_by(func.random()).limit(limit).all()


def get_recently_added_items(db: Session, pool_size: int = 100, limit: int = 10):
    """"Recently added" proxy — this catalog is a static historical
    snapshot with no real "date added to store" column, so `item_age_days`
    (days between an item's first observed interaction and the dataset's
    cutoff — smaller means it entered the interaction data more recently)
    stands in for recency. Pool of the `pool_size` most recent by that
    measure, then a true-random pick of `limit` from that pool — changes on
    every call, same as `get_random_items_excluding_categories`."""
    recent_pool = (
        db.query(Item.parent_asin)
        .filter(Item.item_age_days.isnot(None))
        .order_by(Item.item_age_days.asc())
        .limit(pool_size)
        .subquery()
    )
    return (
        db.query(Item)
        .filter(Item.parent_asin.in_(db.query(recent_pool.c.parent_asin)))
        .order_by(func.random())
        .limit(limit)
        .all()
    )


# --- Signup ---


def create_user(db: Session, user_id: str) -> "User | None":
    """Create a new demo-able user for signup. Returns None if user_id is taken."""
    if db.get(User, user_id) is not None:
        return None
    user = User(user_id=user_id)
    db.add(user)
    db.commit()
    return user


def set_category_preferences(db: Session, user_id: str, categories: list):
    for category in categories:
        if db.get(UserCategoryPreference, (user_id, category)) is None:
            db.add(UserCategoryPreference(user_id=user_id, category=category))
    db.commit()


def get_category_preferences(db: Session, user_id: str) -> list:
    rows = db.query(UserCategoryPreference).filter(UserCategoryPreference.user_id == user_id).all()
    return [row.category for row in rows]


def backfill_interaction(db: Session, user_id: str, parent_asin: str, rating: float = 5.0):
    """Insert a `test`-split interaction if one doesn't already exist for
    this (user, item) pair. Called on `checkout`/`onboarding_like` so a
    brand-new signed-up user's "Recommended for you" can compute a live
    two-tower embedding immediately, instead of only ever falling back to
    popularity (see `personalized_recommendations` in recommend.py)."""
    exists = (
        db.query(Interaction)
        .filter(Interaction.user_id == user_id, Interaction.parent_asin == parent_asin, Interaction.split == "test")
        .first()
    )
    if exists is not None:
        return
    db.add(Interaction(
        user_id=user_id, parent_asin=parent_asin, rating=rating,
        label=1 if rating >= 4 else 0, timestamp=datetime.utcnow(), split="test",
    ))
    db.commit()


# --- Implicit events ---


def log_event(
    db: Session, session_id: str, event_type: str, signal: str,
    user_id: str = None, parent_asin: str = None, position: int = None, value: float = None,
) -> UserEvent:
    event = UserEvent(
        user_id=user_id, session_id=session_id, parent_asin=parent_asin,
        event_type=event_type, signal=signal, position=position, value=value,
    )
    db.add(event)
    db.commit()
    return event


def has_event(db: Session, session_id: str, parent_asin: str, event_type: str) -> bool:
    return (
        db.query(UserEvent)
        .filter(UserEvent.session_id == session_id, UserEvent.parent_asin == parent_asin, UserEvent.event_type == event_type)
        .first()
        is not None
    )


def count_top_row_impressions(db: Session, session_id: str, parent_asin: str, max_position: int) -> int:
    return (
        db.query(UserEvent)
        .filter(
            UserEvent.session_id == session_id, UserEvent.parent_asin == parent_asin,
            UserEvent.event_type == "impression", UserEvent.position < max_position,
        )
        .count()
    )


def has_user_event(db: Session, user_id: str, parent_asin: str, event_type: str) -> bool:
    return (
        db.query(UserEvent)
        .filter(UserEvent.user_id == user_id, UserEvent.parent_asin == parent_asin, UserEvent.event_type == event_type)
        .first()
        is not None
    )


def has_checked_out(db: Session, user_id: str, parent_asin: str) -> bool:
    return has_user_event(db, user_id, parent_asin, "checkout")


# --- Wishlist ---


def get_wishlist(db: Session, user_id: str) -> list:
    return db.query(Wishlist).filter(Wishlist.user_id == user_id).order_by(Wishlist.added_at.desc()).all()


def add_to_wishlist(db: Session, user_id: str, parent_asin: str) -> Wishlist:
    existing = db.query(Wishlist).filter(Wishlist.user_id == user_id, Wishlist.parent_asin == parent_asin).first()
    if existing is not None:
        return existing
    row = Wishlist(user_id=user_id, parent_asin=parent_asin)
    db.add(row)
    db.commit()
    return row


def remove_from_wishlist(db: Session, user_id: str, parent_asin: str) -> bool:
    row = db.query(Wishlist).filter(Wishlist.user_id == user_id, Wishlist.parent_asin == parent_asin).first()
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def bump_wishlist_views(db: Session, user_id: str) -> list:
    """Increment `view_count` on every row in a user's wishlist — called
    each time the wishlist page loads. Backing counter for
    `repeated_wishlist_no_purchase`."""
    rows = get_wishlist(db, user_id)
    for row in rows:
        row.view_count += 1
    db.commit()
    return rows


# --- Cart ---


def get_cart(db: Session, user_id: str) -> list:
    return db.query(CartItem).filter(CartItem.user_id == user_id).order_by(CartItem.added_at.desc()).all()


def add_to_cart(db: Session, user_id: str, parent_asin: str) -> CartItem:
    existing = db.query(CartItem).filter(CartItem.user_id == user_id, CartItem.parent_asin == parent_asin).first()
    if existing is not None:
        return existing
    row = CartItem(user_id=user_id, parent_asin=parent_asin)
    db.add(row)
    db.commit()
    return row


def remove_from_cart(db: Session, user_id: str, parent_asin: str) -> bool:
    row = db.query(CartItem).filter(CartItem.user_id == user_id, CartItem.parent_asin == parent_asin).first()
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def clear_cart(db: Session, user_id: str):
    db.query(CartItem).filter(CartItem.user_id == user_id).delete()
    db.commit()
