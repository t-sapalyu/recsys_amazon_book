"""Seed PostgreSQL from the offline artifacts notebooks 01-05 produced.

Populates:
- `items`  — full catalog from items_meta.parquet, plus item-level ranking
  features precomputed here with the *exact* definitions notebook 05 used
  (log1p popularity, log1p price, mean/median imputation, days since first
  train interaction) so the ranker sees identical features at serving time.
- `users`  — a capped set of demo-able users: those with test-period
  interactions, since project.md's "logged in" demo specifically needs a
  user's test-period activity to compute a live embedding from.
- `interactions` — train + test rows for exactly those demo users (not the
  full ~21k train users — the DB only needs to serve users the demo can
  actually log in as).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

import db_utility
from db_utility import Interaction, Item, User

APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
DATA_DIR = APP_ROOT / "data" / "processed"
N_DEMO_USERS = int(os.getenv("N_DEMO_USERS", "300"))


def build_item_rows(items_meta: pd.DataFrame, train: pd.DataFrame) -> list:
    """Mirrors notebook 05's Section 6 feature definitions exactly."""
    items_meta = items_meta.set_index("parent_asin")
    train_max_ts = train["timestamp"].max()

    item_popularity = np.log1p(train[train["label"] == 1]["parent_asin"].value_counts())
    item_first_seen = train.groupby("parent_asin")["timestamp"].min()

    avg_rating_fill = items_meta["average_rating"].mean()
    price_fill = items_meta["price"].median()

    rows = []
    for parent_asin, row in items_meta.iterrows():
        price = row["price"] if pd.notna(row["price"]) else price_fill
        avg_rating = row["average_rating"] if pd.notna(row["average_rating"]) else avg_rating_fill
        first_seen = item_first_seen.get(parent_asin, train_max_ts)
        rows.append({
            "parent_asin": parent_asin,
            "title": row.get("title"),
            "author": row.get("author") if pd.notna(row.get("author")) else None,
            "category": row.get("categories"),
            "image_url": row.get("image_url"),
            "average_rating": None if pd.isna(row["average_rating"]) else float(row["average_rating"]),
            "price": None if pd.isna(row["price"]) else float(row["price"]),
            "item_popularity": float(item_popularity.get(parent_asin, 0.0)),
            "item_avg_rating_feature": float(avg_rating),
            "item_price_norm": float(np.log1p(price)),
            "item_age_days": int((train_max_ts - first_seen).days),
        })
    return rows


def select_demo_users(test: pd.DataFrame, n: int) -> list:
    """Cap the demo-able user pool for a clean, fast-loading Streamlit picker
    — every seeded user still has real test-period history to demo with."""
    counts = test["user_id"].value_counts()
    return counts.head(n).index.tolist()


def build_interaction_rows(train: pd.DataFrame, test: pd.DataFrame, demo_users: set) -> list:
    rows = []
    for split_name, df in (("train", train), ("test", test)):
        sub = df[df["user_id"].isin(demo_users)]
        for r in sub.itertuples(index=False):
            rows.append({
                "user_id": r.user_id,
                "parent_asin": r.parent_asin,
                "rating": float(r.rating),
                "label": int(r.label),
                "timestamp": r.timestamp,
                "split": split_name,
            })
    return rows


def _insert_stmt(model):
    """SQLite (used for local/CI testing) and Postgres (production) both
    support `ON CONFLICT DO NOTHING`, just via different dialect modules."""
    if db_utility.engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        return sqlite_insert(model)
    return insert(model)


def upsert_batch(session, model, rows: list, pk_cols: list, batch_size: int = 5000):
    """Insert, skipping rows whose PK already exists — correct for `users`
    and `interactions`, which are append-only historical facts."""
    if not rows:
        return
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        stmt = _insert_stmt(model).values(batch)
        stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
        session.execute(stmt)
    session.commit()


def upsert_update_batch(session, model, rows: list, pk_cols: list, batch_size: int = 5000):
    """Insert-or-update — correct for `items`, which is a *current snapshot*
    of items_meta.parquet, not historical fact. Plain `DO NOTHING` would leave
    existing rows permanently stale after a schema/data change (e.g. adding
    the `author` column, or re-running against updated metadata) since the
    parent_asin PK already exists on a rerun."""
    if not rows:
        return
    update_cols = [c for c in rows[0].keys() if c not in pk_cols]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        stmt = _insert_stmt(model).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=pk_cols,
            set_={col: getattr(stmt.excluded, col) for col in update_cols},
        )
        session.execute(stmt)
    session.commit()


def migrate_schema():
    """`create_all` only creates missing *tables*, never alters existing ones
    — tables from before a column was added need an explicit ALTER. Checked
    defensively (only runs if the column is actually missing) so this stays
    a no-op on a fresh database or a rerun."""
    from sqlalchemy import text
    with db_utility.engine.connect() as conn:
        if db_utility.engine.dialect.name == "sqlite":
            existing_item_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(items)"))}
            if "author" not in existing_item_cols:
                conn.execute(text("ALTER TABLE items ADD COLUMN author VARCHAR"))
            existing_user_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(users)"))}
            if "created_at" not in existing_user_cols:
                # Backfilled with the current timestamp (not NULL) — a NULL
                # would sort inconsistently against real timestamps in the
                # `ORDER BY created_at DESC` used to surface recent signups.
                conn.execute(text("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            conn.commit()
        else:
            conn.execute(text("ALTER TABLE items ADD COLUMN IF NOT EXISTS author VARCHAR"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT now()"))
            conn.commit()


def main():
    print(f"Seeding from {DATA_DIR}")

    print("Creating tables...")
    db_utility.Base.metadata.create_all(bind=db_utility.engine, checkfirst=True)
    migrate_schema()

    items_meta = pd.read_parquet(DATA_DIR / "items_meta.parquet")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    session = db_utility.SessionLocal()
    try:
        print(f"Seeding {len(items_meta):,} items...")
        item_rows = build_item_rows(items_meta, train)
        upsert_update_batch(session, Item, item_rows, ["parent_asin"])

        demo_user_ids = select_demo_users(test, N_DEMO_USERS)
        print(f"Seeding {len(demo_user_ids):,} demo users...")
        upsert_batch(session, User, [{"user_id": u} for u in demo_user_ids], ["user_id"])

        interaction_rows = build_interaction_rows(train, test, set(demo_user_ids))
        print(f"Seeding {len(interaction_rows):,} interactions (train+test, demo users only)...")
        # Interactions have a synthetic serial PK, not a natural one — dedupe
        # on rerun via (user_id, parent_asin, split) instead of on_conflict.
        existing = {
            (r.user_id, r.parent_asin, r.split)
            for r in session.query(Interaction.user_id, Interaction.parent_asin, Interaction.split)
        }
        new_rows = [r for r in interaction_rows if (r["user_id"], r["parent_asin"], r["split"]) not in existing]
        if new_rows:
            session.bulk_insert_mappings(Interaction, new_rows)
            session.commit()
        print(f"Inserted {len(new_rows):,} new interaction rows ({len(interaction_rows) - len(new_rows):,} already present).")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print("Seeding complete.")


if __name__ == "__main__":
    main()
