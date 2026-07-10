# Project: Build a Real-Time Recommendation System

**Groups of 4 • Defense during last session**

## Objective

Build a real-time recommendation system — from raw data to a serving webapp with live inference. You will implement retrieval and ranking models, serve them through an API, and build a frontend that showcases multiple recommendation strategies.

## Dataset

Choose one of the following:

| Dataset | Size | Type | Notes |
|---------|------|------|-------|
| [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) | 25M ratings, 62K movies | Explicit | Dense, clean — safe choice |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | Varies by category, 100K-50M | Explicit + implicit | Pick one category (Books, Electronics, Fashion, etc.). Rich metadata |
| [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) | 7M interactions, 32K games | Implicit (playtime) | Good implicit signal, rich game metadata |
| [H&M Purchases](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) | 31M transactions, 105K items | Implicit (purchases) | Requires Kaggle account. Seasonal — temporal drift is a challenge |
| [Yelp](https://www.yelp.com/dataset) | 7M reviews, 150K businesses | Explicit | Location-dependent — recommendations without geo-context are harder |

> **Note:** If your dataset exceeds 5M interactions, subsample to the most recent 5M. Document your subsampling strategy in the notebook.

> **Not allowed:** MovieLens 100K (used in labs).

## Requirements

### 1. Data Preparation & Exploration

- Load and clean the dataset
- Exploratory analysis: sparsity, rating/interaction distribution, long tail, temporal patterns
- **Time-based train/test split** (mandatory — random split = 0 on evaluation criterion)
- Use a validation split from training data for hyperparameter tuning. Do not tune on the test set.
- Define what counts as a positive interaction (threshold if needed)

### 2. Baselines

- **Random** recommender
- **Popularity** recommender (most popular in training set)
- Report: Recall@20, NDCG@10, Catalog Coverage (% of items recommended at least once)

### 3. Retrieval

Implement **both** from scratch (not library calls):

- **MF-BPR:** Matrix factorization with BPR pairwise loss, SGD training
- **Two-tower:** User and item towers in PyTorch, in-batch negatives, softmax cross-entropy loss

For both:
- Index item embeddings with **FAISS** (Flat or IVF)
- Retrieve top-100 candidates per user
- Report: Recall@20, Recall@50, NDCG@10, Catalog Coverage

Compare: Popularity vs MF-BPR vs Two-tower

### 4. Ranking

- Engineer **at least 10 distinct features** (user-level, item-level, cross-features, temporal)
- Train a **LightGBM** ranker (LambdaRank objective) on candidates from your best retriever. Training data: retrieve candidates for train users, label as positive (interacted in ground truth) or negative (not interacted)
- Report: NDCG@10, Catalog Coverage — compare with retrieval-only ranking

Feature ideas: user activity count, item popularity, genre/category match, days since last interaction, user average rating, item average rating, time-of-day, user-item category affinity, etc.

### 5. Demo Webapp

Build a web application that showcases your recommendation pipeline in context. Different pages/sections should use different models — just like a real product.

**Required pages/sections:**

| Section | Model | Behavior |
|---------|-------|----------|
| Homepage (not logged in) | Popularity | Show trending/popular items |
| Homepage (logged in) | Two-tower → ranked | Personalized recommendations |
| Item page ("Similar items") | Cosine on learned embeddings | Given current item, show nearest items |
| Homepage row ("Because you liked X") | Same embeddings | Pick one item from user history, show items similar to *that* item |

**Tech stack (required):**
- **PostgreSQL** — user profiles, interaction history, item metadata (title, image, category)
- **FastAPI** backend — loads models + FAISS index at startup, serves recommendations as a REST API with documented endpoints (Swagger UI at `/docs`)
- **Streamlit** frontend — calls the API and displays results
- **Docker Compose** — all services (postgres, api, frontend) run with `docker compose up`

This separation mirrors production architecture: database, model serving, and UI are independent services.

**Architecture:**
- **Offline (batch):** Train models in notebook, export item embeddings, FAISS index, LightGBM model. Populate PostgreSQL with user data and item metadata.
- **Online (real-time):** At request time, API fetches user history from PostgreSQL, computes user embedding (forward pass through user tower) → FAISS top-100 → LightGBM re-rank → serve top-10
- Use your time-based split: train models on training data, use test-period interactions as user history in the webapp. This simulates a real deployment where the model serves recommendations based on recent activity it wasn't trained on.

**Implementation:**
- Display item metadata: title, image (from dataset or public URLs), and category/genre
- Show which model produced each recommendation (e.g., "Two-tower + LightGBM" or "Popular right now")
- Recommendations should vary on reload (e.g., sample from top candidates using temperature scaling)
- For unknown users (not logged in), fall back to popularity
- Item-to-item similarities can be precomputed (they're static)
- Log and report average response time in the README
- **Demo live during defense** with `docker compose up`. Deployment to a public URL is optional bonus.

**The goal:** A portfolio-ready project you can demo in interviews and show on your resume.

### 6. Analysis

Produce an `ANALYSIS.md` file in your repo with:

- **Ablation table:** Retrieval-only vs +ranking (metrics side by side)
- **Cold start analysis:** Performance by user activity level (few vs many interactions)
- **Feature importance:** Top ranking features (table or plot)
- **Latency breakdown:** Average time per component (DB query, user tower, FAISS, LightGBM)
- **Limitations:** One limitation of your system and how you'd address it with more time
- **Final comparison table:** All methods, all metrics, side by side

No code — just results, tables, and discussion.

## Bonus (optional, for extra credit)

Pick one or more:

- **MMR re-ranking:** Implement Maximal Marginal Relevance, tune λ, plot relevance-diversity trade-off
- **Personalized search:** BM25/TF-IDF retrieval on item titles + LightGBM re-ranking with user features (personalized search results)
- **Deploy webapp** to a public URL (Streamlit Cloud, Render, or Vercel)
- Sequential model (SASRec or GRU4Rec) as additional retriever
- Hard negative mining or sampling bias correction for two-tower
- Negative sampling comparison (uniform vs popularity-based)
- Fairness analysis: performance across user groups (by activity level)
- Explainability: generate explanations for top recommendations
- Multi-source retrieval: blend candidates from MF-BPR + two-tower + popularity

## Constraints & Policies

- **Time-based split is mandatory.** Using a random split results in 0 for the Data preparation & baselines criterion.
- **Groups and dataset choice must be communicated to the instructor by end of the week following S2.**
- **Implement from scratch:** MF-BPR training loop and two-tower model must be your code (PyTorch). You may use libraries for FAISS, LightGBM, data loading, and standard utilities.
- **No full-pipeline libraries** (RecBole, Surprise, LensKit) for the core models. Use them only for additional baselines if you want.
- **Subsample** datasets larger than 5M interactions.

**GenAI:** You may use GenAI tools (ChatGPT, Copilot, etc.) for implementation. However:

- **Implement core models from scratch** (MF-BPR, two-tower) — even if GenAI helps you write the code, the goal is that you understand every line. Writing the code yourself is the best way to learn how these models work.
- **You must be able to explain your code during the defense.** If you cannot explain your own loss function, training loop, or design choices, it will be obvious and graded accordingly.
- GenAI cannot do the analysis for you — interpreting results, explaining failures, and making design decisions is where the learning happens.

## Deliverables

### Code Repository (GitHub)

- Add the instructor as collaborator
- **Notebook** must run end-to-end on Google Colab (training and evaluation)
- **Webapp** runs locally with `docker compose up` — demo live during defense
- Save checkpoints to Google Drive to handle Colab session timeouts

### Notebook (your main deliverable)

Your notebook **is** your report. It must be readable top-to-bottom by someone who hasn't seen your code before. You may split into multiple notebooks (e.g., one per stage) — each must be self-contained and narrated.

Requirements:
- **Markdown cells between code** — explain what you're doing and why (not just "train model" but "we use BPR because...")
- **Design decisions documented** — why this dataset, why this subsample strategy, why k=50 candidates, why these features
- **Results printed/plotted inline** — tables and charts, not raw numbers
- **Interpretation after each result** — a markdown cell explaining what you observe and why

A notebook with only code and no narrative is not acceptable.

### README

- GIF of the webapp in action
- Results table (all methods, all metrics — one glance shows if it works)
- Who did what (task split per team member)
- Average response time of the API
- Setup instructions (how to reproduce)

### ANALYSIS.md

Dedicated analysis file — see Section 6 for required content.

### Presentation

- 15 minutes + 5 minutes questions
- **Slides** (results, insights, architecture decisions) + **live webapp demo**
- **Every team member must be able to explain any part of the code and justify design decisions**

## Grading

| Criterion | Weight |
|-----------|--------|
| Data preparation & baselines (correct split, proper baselines) | 10% |
| Retrieval (MF-BPR + two-tower, both working) | 25% |
| Ranking (feature engineering + LightGBM) | 15% |
| Demo webapp (Docker, API, frontend, real-time inference) | 20% |
| Analysis & insight (ANALYSIS.md) | 15% |
| Notebook quality & documentation (readable, well-narrated, reproducible) | 5% |
| Presentation & defense | 10% |
| Bonus | Up to +10% |

## Planning

### Workload Estimate

| Task | Estimated hours |
|------|----------------|
| Data prep, EDA, train/test/val split | 5-6h |
| Baselines + evaluation pipeline | 5-6h |
| MF-BPR (implementation + tuning) | 8-10h |
| Two-tower (implementation + tuning) | 10-15h |
| FAISS indexing + retrieval evaluation | 2-3h |
| Feature engineering (10+ features) | 6-8h |
| LightGBM ranker | 4-5h |
| Demo webapp (API + frontend + Docker + DB) | 15-20h |
| Analysis (ablation, cold start, feature importance, latency, limitations) | 5-7h |
| Notebook narration + README + GIF | 4-5h |
| Presentation prep | 2-3h |
| **Total** | **~66-88h** |

Per person (group of 4): **~17-22h.**

Document who did what in the README.

### Timeline

Start early — many tasks don't require waiting for later sessions.

| After | What you can do |
|-------|-----------------|
| **S1** | Dataset selection, EDA, train/test split, baselines, evaluation pipeline |
| **S2** | MF-BPR, feature engineering, webapp skeleton (show popularity + MF-BPR recs) |
| **S3** | Two-tower, FAISS indexing, retrieval evaluation, item embeddings for webapp |
| **S4** | LightGBM ranker, final analysis, webapp finalization |
| **S5** | Polish, presentation prep, **defense** |

> **Start the webapp early.** After S2 you can already show popularity and MF-BPR recommendations. Add two-tower and ranking results as you build them.

### Compute

Free Google Colab (T4 GPU) is sufficient for all required components. Tips:
- Save model checkpoints and preprocessed data to Google Drive
- Use GPU runtime for PyTorch training (MF-BPR, two-tower)
- LightGBM and FAISS run fine on CPU
- If Colab disconnects, your Drive data persists — just reload and continue
