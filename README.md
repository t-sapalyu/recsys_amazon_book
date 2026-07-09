# Real-Time Recommendation System — Amazon Books

Recommendation system built on Amazon Reviews 2023 (Books category): retrieval
(MF-BPR, two-tower), ranking (LightGBM), and a live demo webapp
(FastAPI + Streamlit + PostgreSQL, Docker Compose).

See [project.md](project.md) for the full assignment spec.

## Repo structure

```
notebooks/      self-contained notebooks — every function inlined, no imports between them
scripts/        one-off local dev script for building the SAMPLE_MODE test data (not needed to run notebooks)
data/           raw/interim/processed data (gitignored — see setup below)
checkpoints/    saved model checkpoints (gitignored)
results/        metrics.csv ledger, appended to by every notebook's evaluation cells
model_service/  FastAPI backend — loads checkpoints + serves recommendations (see Webapp below)
webapp/         Streamlit frontend
docker-compose.yml, .env   3 services: postgres, fastapi, streamlit
```

Each notebook is fully self-contained: every function it uses is defined in
its own cells, not imported from a shared package. This means some code
(parsing helpers, the evaluation harness) is duplicated across notebooks —
deliberate, so any single notebook can be opened on Colab/Kaggle/locally and
run top to bottom with nothing to set up beyond `pip install`. No repo
clone, no `sys.path` hacking, nothing to upload as a Kaggle "input" except
data. (`scripts/make_sample_dataset.py` imports its own copy of the sampling
logic the same way — it's a standalone dev tool, not a shared library.)

## Notebooks

| # | Notebook | Covers |
|---|----------|--------|
| 01 | `01_data_prep_eda.ipynb` | Download, clean, EDA, time-based train/val/test split |
| 02 | `02_baselines.ipynb` | Evaluation harness (Recall@K, NDCG@10, Catalog Coverage), Random & Popularity baselines |
| 03 | `03_mf_bpr.ipynb` | MF-BPR (from scratch, PyTorch), FAISS indexing, val-tuned hyperparameters |
| 04 | `04_two_tower.ipynb` | Two-tower (from scratch, PyTorch): inductive history-pooled user tower, in-batch negatives, softmax cross-entropy |
| 05 | `05_ranking.ipynb` | 12-feature LightGBM LambdaRank ranker on top of the val-selected best retriever; ablation vs. retrieval-only |

**`modeling_pipeline.ipynb`** is 01-05 combined into one notebook, run top to
bottom in a single session. Same design decisions and interpretations as the
individual notebooks — this exists purely so the whole pipeline can be handed
over or graded as one file. Since everything shares one kernel here, some
work only has to happen once instead of per-notebook: the ID mapping (used by
MF-BPR, two-tower, and ranking) is built a single time, and Part 5's ranking
stage reuses whichever retriever's embeddings/FAISS index are already in
memory instead of reloading a checkpoint from disk and reconstructing a model.
The two versions are kept in sync — if you change one, check whether the same
fix applies to the other. Prefer the individual notebooks (01-05) if you want
a focused, standalone read of a single stage (e.g. to hand someone just the
two-tower work); prefer `modeling_pipeline.ipynb` for a single top-to-bottom run.

## Setup

```bash
pip install -r requirements.txt
```

Notebooks run locally, on Google Colab, or on Kaggle (they auto-detect the
environment and adjust paths accordingly — see the Setup cell in each
notebook). No repo clone needed on any platform; only the `pip install` in
the Setup cell.

Raw/processed data and checkpoints are gitignored — running notebook 01
downloads and regenerates everything under `data/`.

### Fast local iteration with SAMPLE_MODE

Every notebook has a `SAMPLE_MODE` flag near the top of its Setup cell.
`SAMPLE_MODE = True` runs the whole pipeline against a small local test set —
the 500 most-reviewed books (picked by review count, so they survive the
k-core filter) — instead of the full ~5M-interaction dataset. Sample and full
runs write to separate directories (`data/*_sample/` vs `data/*/`,
`results/metrics_sample.csv` vs `results/metrics.csv`) so they never mix.

To build the sample (one-time, run locally — needs the full raw files first):

```bash
# Place the full raw files at data/Books.jsonl.gz and data/meta_Books.jsonl.gz
# (from https://amazon-reviews-2023.github.io/), then:
python scripts/make_sample_dataset.py
```

Set `SAMPLE_MODE = False` in each notebook for the real run (used for actual
reported results / the Colab deliverable).

### Running on Kaggle instead of Colab

Notebooks auto-detect Kaggle (`KAGGLE_KERNEL_RUN_TYPE` env var) alongside
Colab — no code changes needed, just environment setup:

1. **Settings → Internet: On** (needed for `pip install`, and for downloading
   the full dataset if `SAMPLE_MODE = False`).
2. For `SAMPLE_MODE = True`: zip and upload `data/raw_sample/` (~222MB, data
   only — no code) as a private Kaggle Dataset, attach it to the notebook
   (**Settings → Add Input**), and fill in its slug in the `RAW_DIR` line in
   Setup. Kaggle has no Drive-style mount — attached Datasets
   (`/kaggle/input/...`, read-only) are the persistence mechanism, and
   `/kaggle/working/` is the writable scratch space (kept only if you commit
   a notebook version).
3. Running notebooks 01 and 02 as **separate** Kaggle sessions means 02 can't
   see 01's `/kaggle/working` output directly — either run both cells in one
   notebook, or commit 01's output and attach it as 02's input.
4. GPU (needed from notebook 03 onward): **Settings → Accelerator → GPU**.
   Session limits: 9h/session, ~30 GPU-hours/week on the free tier.

## Webapp

FastAPI (`model_service/`) + Streamlit (`webapp/`) + PostgreSQL, wired together
by `docker-compose.yml`. Only three services — matches project.md's spec, no
extra infra.

### Architecture

- **Offline** (the notebooks): train models, export item/user embeddings,
  the FAISS index, and the LightGBM model to `checkpoints[_sample]/`.
- **Online** (`model_service/prestart.py`, once at container startup):
  seeds Postgres from `data/processed[_sample]/{items_meta,train,test}.parquet`
  — full item catalog, plus a capped pool of demo users (those with
  test-period interactions) and their train+test interaction rows.
- **Serving** (`model_service/main.py` + `recommend.py`): at request time,
  fetches a user's history from Postgres, computes their embedding via a
  forward pass through the **two-tower user tower** (the only model here with
  an inductive user tower — see below), retrieves top-100 via FAISS, re-ranks
  with LightGBM, and returns the top-K.

**Why only two-tower is used online, never MF-BPR**, even though notebook 05
picks whichever retriever wins on validation: MF-BPR has a fixed per-user
lookup table, so it structurally cannot score a user from fresh history at
request time — only two-tower's inductive user tower can. This matches
project.md's own table ("Homepage (logged in) | Two-tower → ranked") and its
Online architecture description ("computes user embedding (forward pass
through user tower)"). MF-BPR's item embeddings are still loaded, but only as
a frozen feature source for the two-tower user tower's history-pooling
(exactly how notebook 04 built it — see `model_loader.py`).

If the LightGBM ranker happens to have been trained on MF-BPR's candidates
(whenever MF-BPR wins notebook 05's val-based selection), it's still applied
to two-tower's candidates in the webapp — a mild, documented feature-scale
mismatch on `retrieval_score` rather than a hard requirement to retrain.
Worth mentioning as a limitation in ANALYSIS.md if it comes up.

**Two history windows, deliberately different** (see `recommend.py`'s
docstring): the live user *embedding* is computed from **test-period**
interactions specifically (project.md: "use test-period interactions as user
history... simulates a real deployment where the model serves recommendations
based on recent activity it wasn't trained on"). The LightGBM *ranking
features* (activity count, avg rating given, recency, category affinity) are
computed from **train-period** interactions instead, matching exactly how the
ranker's training data was built in notebook 05 — recomputing them from a
handful of test-period rows would feed the ranker feature values from a
different, much smaller population than it learned from.

### Endpoints (Swagger UI at `/docs`)

**Required by project.md:**

| Endpoint | Section |
|---|---|
| `GET /recommendations/popular` | Homepage, not logged in |
| `GET /recommendations/personalized?user_id=` | Homepage, logged in (falls back to popular for unknown/missing user) |
| `GET /users/{user_id}/because-you-liked` | Homepage row |
| `GET /items/{item_id}/similar` | Item page — cosine on two-tower embeddings |

**Beyond spec** — the same learned embeddings applied two more ways, plus a
plain metadata lookup, since a single retriever/ranker pair naturally supports
more than one kind of "you might also like":

| Endpoint | Section | How it differs from the above |
|---|---|---|
| `GET /users/{user_id}/similar-users-liked` | Homepage row: "People like you also like" | User-**user** similarity (nearest *users* by two-tower embedding, then what *they* liked) — not user-item, which is what the personalized homepage already does. Classic collaborative filtering, built on the same embeddings instead of a separate co-occurrence model. |
| `GET /items/{item_id}/same-author` | Item page: "More by this author" | Plain `WHERE author = ...` lookup — no embeddings, no ML. Needed real data-pipeline work: project.md's metadata spec only asked for title/image/category, so `author` wasn't extracted until this feature — see notebook 01/04's `author` field and `db_utility.py`'s migration. |
| `GET /items/{item_id}/similar-authors` | Item page: "Readers who liked this also read" | Nearest **authors** by centroid embedding (mean of an author's books), not nearest books — surfaces "a whole writer to explore" rather than one similar title. Centroids for authors with only 1-2 books in the SAMPLE_MODE catalog are noisy; expect this to sharpen on the full dataset. |

**Supporting**: `GET /items/{item_id}`, `GET /users`, `GET /health`, `GET /metrics`.

Every recommendation response includes `source_model` (e.g. `"Two-tower +
LightGBM"`, `"Popular right now"`) so the frontend can show which model
produced it. Recommendations are temperature-sampled from a wider candidate
pool per request (`temperature` query param) — reload the homepage to see
them vary, per project.md. User embeddings (for "people like you") and author
centroids are precomputed once at API startup, not per request.

### Running it

```bash
docker compose up
```

Then open the Streamlit frontend (`http://localhost:8501`) and the API's
Swagger docs (`http://localhost:8000/docs`). `.env`'s `SAMPLE_MODE` must match
whichever mode you last ran the notebooks in — `true` reads
`checkpoints_sample/` + `data/processed_sample/`, `false` reads the full run.

### Average response time

Measured via `GET /metrics` against the real `docker compose up` stack
(Postgres, SAMPLE_MODE, 300-user demo pool, 37 requests mixing
`/recommendations/personalized`, `/because-you-liked`, and `/similar`):

| | avg | p50 | p95 |
|---|---|---|---|
| Response time | 11.3ms | 12.5ms | 15.5ms |

Re-measure against the full (non-SAMPLE_MODE) dataset once that run exists —
these numbers are for the ~500-item/300-user demo catalog, not the full
~5M-interaction one.

## Results

_Filled in as each stage lands — see `ANALYSIS.md` (not yet created) for the
full ablation, cold-start, and feature-importance analysis._

## Team

_Task split — fill in before submission._
