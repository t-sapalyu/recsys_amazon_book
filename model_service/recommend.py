"""Recommendation logic: the four sections project.md's webapp spec requires
(Popularity, Two-tower + LightGBM, item-item "Similar items", "Because you
liked X"), plus three additional sections built on the same learned
embeddings — "People like you also like" (user-user similarity), "More by
this author", and "Readers who liked this also read" (author-centroid
similarity).
"""

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import db_utility
from model_loader import Artifacts


def _temperature_sample(scored_items: List[Tuple[object, float]], k: int, temperature: float, rng: np.random.Generator):
    """Sample up to k items with probability proportional to exp(score/temperature).

    Satisfies project.md's "recommendations should vary on reload" — a small
    temperature stays close to deterministic top-k, a larger one adds variety
    without ignoring relevance entirely (unlike uniform random sampling).
    """
    if len(scored_items) <= k:
        return scored_items
    scores = np.array([s for _, s in scored_items], dtype=np.float64)
    scores = scores - scores.max()  # numerical stability, doesn't change softmax output
    probs = np.exp(scores / max(temperature, 1e-6))
    probs = probs / probs.sum()
    idx = rng.choice(len(scored_items), size=k, replace=False, p=probs)
    return [scored_items[i] for i in idx]


def popular_recommendations(db, k: int = 10, pool_size: int = 30, temperature: float = 0.5, rng: np.random.Generator = None):
    """Homepage, not logged in. Same definition as notebook 02's PopularityRecommender."""
    rng = rng or np.random.default_rng()
    pool = db_utility.get_popular_items(db, limit=pool_size)
    scored = [(item, item.item_popularity or 0.0) for item in pool]
    return _temperature_sample(scored, k, temperature, rng)


def similar_items(
    db, artifacts: Artifacts, item_id: str, k: int = 10, pool_size: int = 30,
    temperature: float = 0.3, rng: np.random.Generator = None, exclude: set = None,
):
    """Item page "Similar items" — cosine (inner product on L2-normalized
    two-tower embeddings is equivalent to cosine; our embeddings aren't
    explicitly normalized, so this is inner-product similarity, consistent
    with how the retriever itself was trained and evaluated in notebook 04)."""
    rng = rng or np.random.default_rng()
    if item_id not in artifacts.tt_item_to_idx:
        return []
    idx = artifacts.tt_item_to_idx[item_id]
    query = artifacts.tt_item_emb[idx: idx + 1].astype("float32")
    exclude_set = set(exclude or set()) | {item_id}

    search_k = min(pool_size + len(exclude_set) + 1, artifacts.faiss_index.ntotal)
    scores, neighbor_idx = artifacts.faiss_index.search(np.ascontiguousarray(query), search_k)

    candidate_ids, candidate_scores = [], []
    for j, sc in zip(neighbor_idx[0], scores[0]):
        if j == -1:
            continue
        cid = artifacts.tt_idx_to_item[j]
        if cid in exclude_set:
            continue
        candidate_ids.append(cid)
        candidate_scores.append(float(sc))
        if len(candidate_ids) >= pool_size:
            break

    items_by_id = db_utility.get_items(db, candidate_ids)
    scored = [(items_by_id[cid], sc) for cid, sc in zip(candidate_ids, candidate_scores) if cid in items_by_id]
    return _temperature_sample(scored, k, temperature, rng)


def because_you_liked(db, artifacts: Artifacts, user_id: str, k: int = 10, temperature: float = 0.3, rng: np.random.Generator = None):
    """Homepage row: pick one item from the user's history, show items similar to it."""
    rng = rng or np.random.default_rng()
    history = db_utility.get_user_interactions(db, user_id)
    if not history:
        return None, []

    positives = [h for h in history if h.label == 1]
    pool = positives or history
    chosen = max(pool, key=lambda h: h.timestamp)  # most recent liked item (or most recent overall)
    because_item = db_utility.get_item(db, chosen.parent_asin)
    seen = {h.parent_asin for h in history}
    sims = similar_items(db, artifacts, chosen.parent_asin, k=k, temperature=temperature, rng=rng, exclude=seen)
    return because_item, sims


def compute_user_embedding_for_serving(db, artifacts: Artifacts, user_id: str) -> Optional[np.ndarray]:
    """The user *embedding* is computed from **test-period** interactions
    specifically — per project.md, this is the whole point of the demo: the
    model produces a fresh embedding from activity it never trained on. Falls
    back to whatever history exists if a user has none in the test split.

    Shared by `personalized_recommendations` and the "people like you also
    like" precomputation (`compute_all_user_embeddings`) — both need exactly
    the same embedding for a given user, computed the same way.
    """
    all_history = db_utility.get_user_interactions(db, user_id)
    if not all_history:
        return None
    test_history = [h for h in all_history if h.split == "test"]
    embedding_source = test_history or all_history
    history_item_ids = [h.parent_asin for h in embedding_source]
    avg_rating_for_embedding = float(np.mean([h.rating for h in embedding_source]))
    return artifacts.user_embedding(history_item_ids, len(history_item_ids), avg_rating_for_embedding)


def compute_all_user_embeddings(db, artifacts: Artifacts) -> dict:
    """Precompute every seeded demo user's embedding once (at API startup) —
    reused by "people like you also like" instead of recomputing per request."""
    embeddings = {}
    for user_id in db_utility.list_all_users(db):
        emb = compute_user_embedding_for_serving(db, artifacts, user_id)
        if emb is not None:
            embeddings[user_id] = emb
    return embeddings


def similar_users_liked(
    db, user_embeddings: dict, user_id: str, k: int = 10, n_similar_users: int = 20,
    pool_size: int = 30, temperature: float = 0.5, rng: np.random.Generator = None,
):
    """"People like you also like" — find the users whose embeddings are
    closest to this user's (not items!), then aggregate what *those* users
    positively interacted with, weighted by how similar each is. A genuinely
    different signal from the personalized homepage: that's "items close to
    your own embedding," this is "what people whose taste resembles yours
    actually liked" — the classic collaborative-filtering framing, just built
    on the same learned embeddings instead of a separate co-occurrence model.
    """
    rng = rng or np.random.default_rng()
    if user_id not in user_embeddings:
        return []
    my_emb = user_embeddings[user_id]
    other_ids = [u for u in user_embeddings if u != user_id]
    if not other_ids:
        return []

    other_embs = np.stack([user_embeddings[u] for u in other_ids])
    scores = other_embs @ my_emb
    top_n = min(n_similar_users, len(other_ids))
    top_idx = np.argsort(-scores)[:top_n]
    similar_user_ids = [other_ids[i] for i in top_idx]
    similarity = {other_ids[i]: float(scores[i]) for i in top_idx}

    seen = {h.parent_asin for h in db_utility.get_user_interactions(db, user_id)}
    item_scores = {}
    for su in similar_user_ids:
        weight = max(similarity[su], 0.0) + 1e-6  # similarity-weighted vote, never zero/negative
        for h in db_utility.get_user_interactions(db, su):
            if h.label != 1 or h.parent_asin in seen:
                continue
            item_scores[h.parent_asin] = item_scores.get(h.parent_asin, 0.0) + weight

    if not item_scores:
        return []
    ranked_ids = sorted(item_scores, key=item_scores.get, reverse=True)[:pool_size]
    items_by_id = db_utility.get_items(db, ranked_ids)
    scored = [(items_by_id[cid], item_scores[cid]) for cid in ranked_ids if cid in items_by_id]
    return _temperature_sample(scored, k, temperature, rng)


def same_author(db, item_id: str, k: int = 10):
    """"More by this author" — a plain metadata lookup, no embeddings involved."""
    item = db_utility.get_item(db, item_id)
    if item is None or not item.author:
        return None, []
    others = db_utility.get_items_by_author(db, item.author, exclude_asin=item_id, limit=k)
    scored = [(it, it.item_popularity or 0.0) for it in others]
    return item.author, scored


def compute_author_centroids(db, artifacts: Artifacts):
    """Precompute each author's centroid embedding (mean of their books' two-
    tower embeddings) once at API startup — reused by `similar_authors_books`."""
    author_items = {}
    for parent_asin, author in db_utility.get_all_authored_items(db):
        author_items.setdefault(author, []).append(parent_asin)

    author_centroids = {}
    for author, item_ids in author_items.items():
        vecs = [artifacts.tt_item_emb[artifacts.tt_item_to_idx[i]] for i in item_ids if i in artifacts.tt_item_to_idx]
        if vecs:
            author_centroids[author] = np.mean(vecs, axis=0)
    return author_centroids, author_items


def similar_authors_books(
    db, author_centroids: dict, author_items: dict, author: str,
    k: int = 10, n_similar_authors: int = 5, temperature: float = 0.4, rng: np.random.Generator = None,
):
    """"Readers who liked this also read..." — nearest *authors* by centroid
    embedding similarity (not nearest books), then pool their books. Distinct
    from "Similar items": that finds books close to one specific book; this
    finds authors whose overall body of work sits close in embedding space,
    which surfaces a different kind of "you might also like" (a whole writer
    to explore, not just one similar title)."""
    rng = rng or np.random.default_rng()
    if author not in author_centroids:
        return [], []
    my_centroid = author_centroids[author]
    other_authors = [a for a in author_centroids if a != author]
    if not other_authors:
        return [], []

    other_centroids = np.stack([author_centroids[a] for a in other_authors])
    scores = other_centroids @ my_centroid
    top_idx = np.argsort(-scores)[: min(n_similar_authors, len(other_authors))]
    similar_authors = [other_authors[i] for i in top_idx]

    candidates = []
    for i in top_idx:
        a = other_authors[i]
        for item_id in author_items[a]:
            candidates.append((item_id, float(scores[i])))

    candidate_ids = [c[0] for c in candidates]
    items_by_id = db_utility.get_items(db, candidate_ids)
    scored = [(items_by_id[cid], sc) for cid, sc in candidates if cid in items_by_id]
    return _temperature_sample(scored, k, temperature, rng), similar_authors


def personalized_recommendations(
    db, artifacts: Artifacts, user_id: str, train_max_ts: datetime,
    k: int = 10, retrieve_k: int = 100, temperature: float = 0.5, rng: np.random.Generator = None,
) -> Optional[List[Tuple[object, float]]]:
    """Homepage, logged in: two-tower retrieval -> LightGBM re-rank.

    **Two history windows, deliberately different:**
    - The user *embedding* (see `compute_user_embedding_for_serving`) is
      computed from **test-period** interactions specifically — per
      project.md, this is the whole point of the demo: the model produces a
      fresh embedding from activity it never trained on.
    - The **ranking features** (activity count, avg rating given, recency,
      category affinity) are computed from their **train-period** interactions
      — matching exactly how the LightGBM ranker's training data was built in
      notebook 05. Recomputing them from a handful of test-period rows instead
      would feed the ranker feature values from a different, much smaller
      population than what it learned from.
    Falls back to "whatever history exists" if a user has none in the
    expected split (keeps the demo working for any seeded user).
    """
    rng = rng or np.random.default_rng()
    all_history = db_utility.get_user_interactions(db, user_id)
    if not all_history:
        return None

    train_history_rows = [h for h in all_history if h.split == "train"]
    ref_rows = train_history_rows or all_history
    seen_items = {h.parent_asin for h in all_history}

    user_emb = compute_user_embedding_for_serving(db, artifacts, user_id)
    if user_emb is None:
        return None

    query = user_emb.reshape(1, -1).astype("float32")
    search_k = min(retrieve_k + len(seen_items) + 50, artifacts.faiss_index.ntotal)
    scores, neighbor_idx = artifacts.faiss_index.search(np.ascontiguousarray(query), search_k)

    candidates = []  # (item_id, score, rank)
    rank = 0
    for j, sc in zip(neighbor_idx[0], scores[0]):
        if j == -1:
            continue
        item_id = artifacts.tt_idx_to_item[j]
        if item_id in seen_items:
            continue
        rank += 1
        candidates.append((item_id, float(sc), rank))
        if rank >= retrieve_k:
            break
    if not candidates:
        return None

    activity_count = np.log1p(len(ref_rows))
    avg_rating_given = float(np.mean([h.rating for h in ref_rows]))
    last_ts = max(h.timestamp for h in ref_rows)
    days_since_last = (train_max_ts - last_ts).days
    last_hour = last_ts.hour

    candidate_ids = [c[0] for c in candidates]
    items_by_id = db_utility.get_items(db, candidate_ids)

    cat_counts = {}
    for h in ref_rows:
        item = items_by_id.get(h.parent_asin) or db_utility.get_item(db, h.parent_asin)
        if item and item.category:
            cat_counts[item.category] = cat_counts.get(item.category, 0) + 1
    total_ref = len(ref_rows)

    rows = []
    for item_id, score, rank in candidates:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        rows.append({
            "parent_asin": item_id,
            "user_activity_count": activity_count,
            "user_avg_rating_given": avg_rating_given,
            "item_popularity": item.item_popularity or 0.0,
            "item_avg_rating": item.item_avg_rating_feature or 0.0,
            "item_price_norm": item.item_price_norm or 0.0,
            "item_category": item.category or "Unknown",
            "user_category_affinity": cat_counts.get(item.category, 0) / total_ref,
            "retrieval_score": score,
            "retrieval_rank": rank,
            "user_days_since_last": days_since_last,
            "user_last_interaction_hour": last_hour,
            "item_age_days": item.item_age_days or 0,
        })
    if not rows:
        return None

    feat_df = pd.DataFrame(rows)
    feat_df["item_category"] = pd.Categorical(feat_df["item_category"], categories=artifacts.ranker_categories)
    lgb_scores = artifacts.booster.predict(feat_df[artifacts.feature_columns])

    scored = [(items_by_id[row["parent_asin"]], float(s)) for row, s in zip(rows, lgb_scores)]
    return _temperature_sample(scored, k, temperature, rng)
