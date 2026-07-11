# Amazon Books Recommender 
# Analysis

---

## 1. Ablation: retrieval-only vs +ranking

These numbers all come from the same setup: each retriever first shortlists
its own top 100–400 candidates, then NDCG/Recall/Coverage are measured
*within that shortlist*. This is what the LightGBM ranker and the live
webapp actually work with. (Section 6a further down measures retrieval a
different, harder way - ranking the *entire* catalog. Therefore, those numbers are
lower. Don't compare across the two directly.)

Same LightGBM LambdaRank ranker, trained once per retriever on that
retriever's own candidates (Presentation slides p.9–10).

| Retriever | Split | Stage | NDCG@10 | Recall@20 | Recall@50 | Catalog cov. |
|---|---|---|---:|---:|---:|---:|
| Popularity | val | retrieval-only | 0.0662 | 0.2225 | 0.4653 | 0.07% |
| Popularity | val | **+LightGBM** | **0.1325** | **0.3504** | **0.6169** | **0.27%** |
| Popularity | test | retrieval-only | 0.0505 | 0.2105 | 0.5010 | 0.07% |
| Popularity | test | **+LightGBM** | **0.1238** | **0.2740** | **0.5679** | **0.26%** |
| MF-BPR | val | retrieval-only | 0.0945 | 0.3082 | 0.6280 | 24.2% |
| MF-BPR | val | **+LightGBM** | **0.1705** | **0.4541** | **0.6732** | **32.9%** |
| MF-BPR | test | retrieval-only | 0.0958 | 0.3277 | 0.6196 | 21.1% |
| MF-BPR | test | **+LightGBM** | **0.1361** | **0.3827** | **0.6677** | **28.7%** |
| Two-tower | val | retrieval-only | 0.1029 | 0.3498 | 0.6470 | 58.9% |
| Two-tower | val | **+LightGBM** | **0.2022** | **0.5190** | **0.7352** | **58.3%** |
| Two-tower | test | retrieval-only | 0.0874 | 0.3107 | 0.6280 | 51.2% |
| Two-tower | test | **+LightGBM** | **0.1639** | **0.4416** | **0.6911** | **50.4%** |

**Ranking lifts every retriever, on every split, on every metric**. This is the
core ablation result. On test, LightGBM roughly **doubles NDCG@10** for
Popularity (+145%) and Two-tower (+87%), and lifts MF-BPR by +42%. The gain
is entirely a re-ranking effect: `retrieval_score`/`retrieval_rank` are only
2 of 14 features (Section 3), so the ranker is mostly winning by exploiting
signal the retriever never saw (author affinity, category affinity, item
freshness) rather than just correcting the retriever's ordering.

Ranking-stage catalog coverage moves much less than accuracy (Two-tower:
51.2% → 50.4%, essentially flat) because coverage here is bounded by each
retriever's own top-100 candidate pool. LightGBM re-ranks *within* that
pool, it can't recommend an item the retriever never surfaced.

---

## 2. Cold-start analysis: performance by user activity level

Test users bucketed into thirds by **train-period interaction count**:

| Activity bucket | Test users | Avg. train interactions | NDCG@10 retrieval-only | NDCG@10 +LightGBM | Recall@20 retrieval-only | Recall@20 +LightGBM |
|---|---:|---:|---:|---:|---:|---:|
| Few (1–5) | 2,142 | 3.5 | 0.1024 | 0.1455 (**+42%**) | 0.3227 | 0.3888 (+20%) |
| Medium (6–14) | 2,023 | 9.0 | 0.1118 | 0.1256 (+12%) | 0.3579 | 0.3870 (+8%) |
| Many (15+) | 1,931 | 41.3 | 0.0740 | 0.1233 (**+67%**) | 0.3047 | 0.3950 (**+30%**) |

Two things stand out:

- **Cold users are not the worst-served group.** Retrieval-only NDCG@10 for
  the "few" bucket (0.1024) is close to "medium" (0.1118) and clearly beats
  "many" (0.0740). MF-BPR's popularity-weighted negative sampling and
  per-item bias term give sparse users a reasonable fallback toward
  generically well-liked books even with almost no history to embed.
- **Ranking helps heavy users the most, in relative terms.** The "many"
  bucket gets the largest NDCG@10 lift (+67%) and the largest Recall@20
  lift (+30%). A long, varied history averages out into a single blurry
  MF-BPR embedding, diluting the retrieval dot-product score but that
  same history is exactly what feeds `author_affinity` /
  `user_category_affinity`, so the ranker recovers precision the
  embedding alone loses. Ranking, not retrieval, is what actually rewards
  having a rich profile.

Net: LightGBM re-ranking is not just an average-case win, it's the
mechanism that keeps performance from degrading for both ends of the
activity spectrum. Cold users get pulled up from a fragile retrieval
baseline, heavy users get their richer signal actually used.

---

## 3. Feature importance

14 features across the 4 required groups (Presentation slides p.9), one LightGBM ranker
per retriever. Gains below are the Two-tower ranker's split gains (the
retriever actually wired into the live webapp).

| Rank | Feature | Group | Gain (relative to #2) |
|---|---|---|---:|
| 1 | `author_affinity` | Cross | 609,041 (**57x**) |
| 2 | `item_category` | Item | 10,727 |
| 3 | `retrieval_rank` | Retrieval | 9,644 |
| 4 | `item_popularity` | Item | 9,254 |
| 5 | `user_category_affinity` | Cross | 3,310 |
| 6 | `retrieval_score` | Retrieval | 3,279 |
| 7 | `item_price_norm` | Item | 1,881 |
| 8 | `item_age_days` | Item | 1,281 |
| 9 | `user_activity_count` | User | 1,205 |
| 10 | `category_overlap_affinity` | Cross | 855 |
| 11–14 | `item_avg_rating`, `user_last_interaction_hour`, `user_days_since_last`, `user_avg_rating_given` | Item/User | 0 (unused) |

**`author_affinity` dominates, and it's the one feature no other notebook in
the project uses** (Section 1 for all three rankers (Popularity, MF-BPR,
Two-tower alike, Presentation slides p.11/13)), at roughly 50–60x the runner-up regardless
of which retriever supplied the candidates. "Has this user read this author
before" is a stronger signal for books than anything the retrievers learn
from embeddings. The four zero-gain features are all *pure* user features
(no cross-term with the candidate item). It is consistent with a median user
history that's short enough (Section 2: bottom-third users average 3.5 train
interactions) that "this user's average rating" or "what hour they last
browsed" carries essentially no discriminative power once `author_affinity`
and `item_category` are available.

---

## 4. Latency breakdown

Measured against the live `docker compose up` stack (Postgres + FastAPI +
the full ~592K-row `data/processed/train.parquet`), by instrumenting
`recommend.personalized_recommendations`'s four stages directly (60 sampled
users, not through HTTP. It isolates in-process compute from network/routing
overhead) plus 98 live HTTP calls to `/recommendations/personalized` for the
end-to-end number the API's own `/metrics` endpoint reports.

| Component | Avg (ms) | p50 (ms) | p95 (ms) | Share of in-process time |
|---|---:|---:|---:|---:|
| DB - fetch user history | 1.20 | 1.03 | 2.02 | 5.9% |
| DB - batch-fetch candidate items | 2.64 | 2.34 | 4.60 | 12.9% |
| DB - category-affinity loop (per-history-row fallback) | 12.58 | 8.16 | 35.73 | **61.6%** |
| User tower forward pass | 0.44 | 0.37 | 0.77 | 2.2% |
| FAISS top-100 search | 0.10 | 0.09 | 0.20 | 0.5% |
| LightGBM predict (100 candidates) | 2.96 | 2.51 | 6.77 | 14.5% |
| Feature-table assembly (pandas) | 0.18 | 0.16 | 0.36 | 0.9% |
| **In-process total** | **20.43** | **16.70** | **46.43** | 100% |
| **End-to-end HTTP** (`GET /metrics`, live traffic) | **37.1** | **25.5** | **98.5** | — |

The model itself is cheap: user tower + FAISS + LightGBM together are under
3.5ms, **~17% of in-process time**. The dominant cost is a **hidden N+1
query**: `personalized_recommendations` re-derives each candidate's category
via `items_by_id.get(...) or db_utility.get_item(db, ...)` per row of the
user's *reference* history (train-period rows) to build
`user_category_affinity` live, and most of those history items aren't
already in the candidate batch fetched two steps earlier. So, each one is a
separate round trip. That single loop is **62% of in-process latency**, more
than 4x the model inference cost combined. The gap between the in-process
total (20.4ms avg) and the end-to-end HTTP number (37.1ms avg) is FastAPI
routing, the SQLAlchemy session dependency, and response serialization. None of it model cost either.

Caveat: this was measured against the two-tower checkpoint actually on disk,
whose FAISS index only covers 479 items. The FAISS-search line would
grow if a full-scale (~43K-item) index were served, but `IndexFlatIP` is
still comfortably sub-millisecond at that size, so it wouldn't change which
component dominates.

---

## 5. Limitations

**Primary: sparsity and temporal drift** (the limitation raised at the
defense, deck p.15). After k-core filtering the median user still has very
few interactions. The "few" bucket in #2 averages 3.5 *train* interactions
per user, and the full pre-split dataset's median is 1 interaction before
the k-core floor of 5 kicks in. Two-tower is the retriever most exposed to
this: its own validation → test comparison (deck p.7–8) drops from
Recall@20 0.049 → 0.035 (−29% relative) across the time gap, while MF-BPR's
holds essentially flat (0.048 → 0.051). A user's mean-pooled MF-BPR history
embedding is a coarse summary that a handful of new test-period books can
shift more than a memorized MF-BPR user vector does.

*How to address it with more time:* a sequential retriever (SASRec or
GRU4Rec) that models interaction *order* rather than mean-pooling history
would target this directly. It degrades more gracefully as history grows
and is explicitly designed for the short, order-sensitive histories this
dataset has. Pair it with hard-negative mining for the two-tower's in-batch
negatives (currently uniform in-batch sampling, which under-penalizes
popular-but-irrelevant negatives) and MMR re-ranking to make the
relevance/diversity trade-off explicit rather than implicit in the ranker.

**Also observed while building this analysis: serving artifacts have drifted
from what's reported.** Not the deck's stated limitation, but concrete and
worth recording since it explains why the live demo's retrieval quality
can't be directly checked against the reported metrics:

- `checkpoints/two_tower/` on disk is a smaller run (21,035 users, 479
  items) than the full-scale run that produced the deck's numbers (48,199
  users, ~43K items). The live webapp's two-tower FAISS index can only
  ever surface 479 books, regardless of what's seeded in Postgres.
- The deployed `checkpoints/lightgbm_ranker/lightgbm_ranker.txt` (loaded by
  `model_service/model_loader.py`) only has **12** features. It's missing
  `author_affinity` and `category_overlap_affinity`, the two "cross"
  features `notebooks/recsys-ranking.ipynb` actually trains with (14
  features, #3). Given #3's finding that `author_affinity` is the single
  strongest feature by a 50–60x margin, the live ranker is running without
  its best signal.
- `notebooks/recsys-ranking.ipynb`'s own two-tower cell constructs the user
  tower with `numeric_dim=4` and per-call-recomputed z-scores; the weights
  actually saved in `checkpoints/two_tower/model.pt` only accept
  `numeric_dim=2` with the *frozen* training-time stats in
  `id_maps.json["user_numeric_stats"]` (matching
  `model_service/model_loader.py` exactly). Running that notebook cell
  against the current checkpoint throws a state-dict size mismatch.

*How to address it:* add a startup assertion in `model_loader.py` that the
FAISS index's item count matches the item count seeded in Postgres (fail
loudly instead of silently serving a stale, undersized catalog), promote
the notebook's 14-feature per-retriever ranker to be the one exported to
the flat `checkpoints/lightgbm_ranker/` path the API loads, and add a CI
smoke test that loads every checkpoint end-to-end against the current
`data/processed/` schema so this kind of drift fails a build instead of
surfacing during analysis.

---

## 6. Final comparison table - all methods, all metrics

### 6a. Baselines & standalone retrieval (Presentation slides p.5/8)

Here each model ranks the *entire* ~43K-book catalog for every user. A
harder task than #1/6b's shortlist-based numbers, so these are lower. Use
this table to compare retrievers to each other and to the Random/Popularity
floor, not to compare against #1/6b.

| Model | Split | Recall@20 | Recall@50 | NDCG@10 | Catalog coverage |
|---|---|---:|---:|---:|---:|
| Random | val | 0.0006 | — | 0.0001 | 99.4% |
| Random | test | 0.0006 | — | 0.0002 | 95.1% |
| Popularity | val | 0.0096 | — | 0.0032 | 0.08% |
| Popularity | test | 0.0080 | — | 0.0021 | 0.08% |
| MF-BPR | val | 0.0484 | 0.0968 | 0.0165 | 54.7% |
| **MF-BPR** | **test** | **0.0507** | **0.0954** | **0.0161** | **50.5%** |
| Two-tower | val | 0.0490 | 0.0886 | 0.0165 | 97.5% |
| Two-tower | test | 0.0349 | 0.0708 | 0.0111 | 94.5% |

MF-BPR is the most accurate single retriever on this harness and the only
one that holds up val→test; Random and Popularity are the required floor
(Popularity: 13x Random on Recall@20, but coverage collapses to ~0.08% —
everyone gets the same ~30 bestsellers).

### 6b. Retrieval → ranking pipeline (candidate-pool harness, incl. bonus)

Same table as #1, plus the bonus multi-source blend (deck p.12: 60
candidates from Two-tower + 30 from MF-BPR + 10 from Popularity, merged
before ranking; `multi_source_100` caps the blended pool at 100).

| Retriever | Split | Stage | NDCG@10 | Recall@20 | Recall@50 | Catalog cov. |
|---|---|---|---:|---:|---:|---:|
| Popularity | test | retrieval-only | 0.0505 | 0.2105 | 0.5010 | 0.07% |
| Popularity | test | +LightGBM | 0.1238 | 0.2740 | 0.5679 | 0.26% |
| MF-BPR | test | retrieval-only | 0.0958 | 0.3277 | 0.6196 | 21.1% |
| MF-BPR | test | +LightGBM | 0.1361 | 0.3827 | 0.6677 | 28.7% |
| Two-tower | test | retrieval-only | 0.0874 | 0.3107 | 0.6280 | 51.2% |
| **Two-tower** | **test** | **+LightGBM** | **0.1639** | **0.4416** | **0.6911** | **50.4%** |
| Multi-source (bonus) | test | retrieval-only | 0.0886 | 0.3018 | 0.6184 | 22.7% |
| Multi-source (bonus) | test | +LightGBM | 0.1523 | 0.4232 | 0.7145 | 47.4% |
| Multi-source, capped @100 (bonus) | test | retrieval-only | 0.1005 | 0.3447 | 0.6396 | 22.9% |
| Multi-source, capped @100 (bonus) | test | +LightGBM | 0.1533 | 0.4334 | 0.7125 | 41.0% |

**Two-tower + LightGBM is the best end-to-end pipeline** on this harness
(NDCG@10 0.164, Recall@20 0.442) despite MF-BPR winning on raw retrieval
accuracy (Section 6a) because Two-tower's much higher catalog coverage (51% vs.
21%) hands the ranker a richer, more diverse candidate pool to re-rank,
and re-ranking is where most of the accuracy comes from (Section 1). This is why
`model_service/recommend.py` serves the live webapp with Two-tower →
LightGBM rather than MF-BPR, even though MF-BPR retrieves more accurately
in isolation. The bonus multi-source blend does not beat plain Two-tower on
NDCG@10 (0.152–0.153 vs. 0.164) diluting Two-tower's strong 60-candidate
share with weaker Popularity/MF-BPR candidates costs more than the added
diversity buys back at rank time, though it modestly beats Two-tower on
Recall@50 (0.71–0.71 vs. 0.69).
