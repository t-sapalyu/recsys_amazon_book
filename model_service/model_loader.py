"""Load offline artifacts (notebooks 01-05) once at API startup.

Only the two-tower retriever is used online — it's the only model in this
project with an inductive user tower, i.e. the only one that can compute an
embedding for a user from their *current* history at request time (matching
project.md's "Homepage (logged in) | Two-tower -> ranked" requirement and the
Online architecture's "computes user embedding (forward pass through user
tower)" description). MF-BPR's item embeddings are still loaded, but purely
as a frozen feature source for the two-tower user tower's history pooling —
exactly how notebook 04 built them; MF-BPR's own FAISS index and per-user
lookup table are not used for serving at all, since MF-BPR has no mechanism
to score a user outside its fixed training-time vocabulary.
"""

import json
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn


class UserTower(nn.Module):
    """Identical architecture to notebook 04/05 — weights are loaded, not trained, here."""

    def __init__(self, history_dim: int, numeric_dim: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(history_dim + numeric_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, history_feats: torch.Tensor, numeric_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([history_feats, numeric_feats], dim=-1)
        return self.mlp(x)


class Artifacts:
    """Container for everything loaded at startup, attached to `app.state.artifacts`.

    Serving only ever reads from the DB (items/interactions) and these
    checkpoints — never the raw notebook parquet files, which are a training-
    time artifact, not a serving-time dependency.
    """

    def __init__(self, checkpoint_root: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- MF-BPR item embeddings: frozen feature source only (see module docstring) ---
        mf_dir = checkpoint_root / "mf_bpr"
        self.mf_item_emb = np.load(mf_dir / "item_embeddings.npy")
        with open(mf_dir / "id_maps.json") as f:
            self.mf_item_to_idx = json.load(f)["item_to_idx"]
        self.mf_global_mean = self.mf_item_emb.mean(axis=0)

        # --- Two-tower: the online retriever ---
        tt_dir = checkpoint_root / "two_tower"
        self.tt_item_emb = np.load(tt_dir / "item_embeddings.npy")
        with open(tt_dir / "id_maps.json") as f:
            tt_maps = json.load(f)
        self.tt_item_to_idx = tt_maps["item_to_idx"]
        self.tt_idx_to_item = np.array(sorted(self.tt_item_to_idx, key=self.tt_item_to_idx.get))
        self.user_numeric_stats = tt_maps["user_numeric_stats"]
        mf_bpr_dim = tt_maps["mf_bpr_embedding_dim"]

        self.user_tower = UserTower(mf_bpr_dim, numeric_dim=2, embedding_dim=tt_maps["best_config"]["embedding_dim"])
        # weights_only=True restricts unpickling to plain tensors — this
        # checkpoint is just {"user_tower": state_dict, "item_tower":
        # state_dict}, no custom classes, so this is a free hardening with
        # no behavior change (and silences the FutureWarning about the
        # upcoming default flip).
        state = torch.load(tt_dir / "model.pt", map_location=self.device, weights_only=True)
        self.user_tower.load_state_dict(state["user_tower"])
        self.user_tower.to(self.device).eval()

        self.faiss_index = faiss.IndexFlatIP(self.tt_item_emb.shape[1])
        self.faiss_index.add(np.ascontiguousarray(self.tt_item_emb.astype("float32")))

        # --- LightGBM ranker ---
        ranker_dir = checkpoint_root / "lightgbm_ranker"
        self.booster = lgb.Booster(model_file=str(ranker_dir / "lightgbm_ranker.txt"))
        with open(ranker_dir / "metadata.json") as f:
            ranker_meta = json.load(f)
        self.feature_columns = ranker_meta["feature_columns"]
        self.ranker_categories = ranker_meta["ranker_categories"]
        self.ranker_best_retriever = ranker_meta["best_retriever"]

    def user_embedding(self, history_item_ids, history_len: int, avg_rating_given: float) -> np.ndarray:
        """Forward pass through the user tower for one user's current history.

        `history_item_ids` should be whatever the caller considers "current"
        history (see db_utility/recommend.py — the demo uses test-period rows
        specifically, to show the model working on activity it never trained on).
        """
        vecs = [self.mf_item_emb[self.mf_item_to_idx[i]] for i in history_item_ids if i in self.mf_item_to_idx]
        history_feat = np.mean(vecs, axis=0) if vecs else self.mf_global_mean

        stats = self.user_numeric_stats
        history_len_norm = (np.log1p(history_len) - stats["history_len_log_mean"]) / (stats["history_len_log_std"] + 1e-8)
        avg_rating_norm = (avg_rating_given - stats["avg_rating_given_mean"]) / (stats["avg_rating_given_std"] + 1e-8)

        history_t = torch.tensor(history_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        numeric_t = torch.tensor([history_len_norm, avg_rating_norm], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.user_tower(history_t, numeric_t)
        return emb.squeeze(0).cpu().numpy()


def load_artifacts(checkpoint_root: Path) -> Artifacts:
    return Artifacts(checkpoint_root)
