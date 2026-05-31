"""
embeddings.py — OpenAI embedding client and geometry utilities.

Mirrors the LLMClient pattern in llm.py: lazy init, cost tracked through
the existing CostTracker. Only dependency beyond numpy is the openai SDK
already used elsewhere.

The geometry utilities are pure numpy and operate on lists that may contain
None entries (for nodes where embedding failed or wasn't attempted). All
metrics propagate None rather than crashing — geometry is observability-
only, never a load-bearing decision input.
"""

import os
import threading

from logger_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────

class EmbeddingClient:
    """OpenAI embedding client. Lazy init, thread-safe."""

    def __init__(self, cost_tracker=None):
        self.cost_tracker = cost_tracker
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is not None:
                return self._client
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not found")
            base_url = os.environ.get("OPENAI_BASE_URL") or None
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            return self._client

    def embed(self, text, model):
        """Embed text. Returns list[float]. Raises on failure (caller decides).

        Accepts plain model names (e.g. 'text-embedding-3-small') or the
        'openai:model' form for consistency with the LLM model strings used
        elsewhere — the prefix is stripped if present.
        """
        model_name = model.split(":", 1)[1] if ":" in model else model
        client = self._get_client()
        resp = client.embeddings.create(model=model_name, input=text)
        if self.cost_tracker is not None:
            # output_tokens=0 — embeddings have no output token cost.
            self.cost_tracker.record(resp.usage.total_tokens, 0, model_name)
        return list(resp.data[0].embedding)


# ──────────────────────────────────────────────
# Geometry utilities (pure numpy)
# ──────────────────────────────────────────────

def _cosine_distance(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def rolling_dispersion(vectors, window=5):
    """Mean pairwise cosine distance over a sliding window.

    Args:
        vectors: list of list[float] | None
        window:  sliding window size
    Returns:
        list[float | None] — None when fewer than 2 valid vectors in window
    """
    import numpy as np
    out = []
    for i in range(len(vectors)):
        lo, hi = max(0, i - window + 1), i + 1
        w = [v for v in vectors[lo:hi] if v is not None]
        if len(w) < 2:
            out.append(None)
            continue
        dists = []
        for a in range(len(w)):
            for b in range(a + 1, len(w)):
                dists.append(_cosine_distance(w[a], w[b]))
        out.append(float(np.mean(dists)))
    return out


def neighborhood_coherence(vectors, k=3):
    """For each vector, mean cosine *similarity* to its k nearest neighbours.

    Returns list[float | None]; None where the vector itself is None.
    """
    import numpy as np
    out = []
    for i in range(len(vectors)):
        if vectors[i] is None:
            out.append(None)
            continue
        sims = []
        for j in range(len(vectors)):
            if i == j or vectors[j] is None:
                continue
            sims.append(1.0 - _cosine_distance(vectors[i], vectors[j]))
        if not sims:
            out.append(None)
            continue
        top = sorted(sims, reverse=True)[:k]
        out.append(float(np.mean(top)))
    return out


def z_score(values):
    """Z-score normalise. None entries pass through; statistics over non-None."""
    import numpy as np
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return [0.0 if v is not None else None for v in values]
    mean, std = float(np.mean(valid)), float(np.std(valid))
    if std == 0:
        return [0.0 if v is not None else None for v in values]
    return [(v - mean) / std if v is not None else None for v in values]


def compute_lineage(vectors):
    """Build the conceptual lineage tree from a sequence of embedding vectors.

    For each vector at index i (i > 0), its conceptual parent is the prior
    vector with smallest cosine distance — the iteration whose tested_estimand
    text is most semantically similar to vector[i]'s. The first vector is
    the root (no parent).

    Returns three lists parallel to ``vectors``:
      - parents[i]    : index of i's parent, or None for the root / None vectors
      - distances[i]  : cosine distance to parent, or None for root / None vectors
      - depths[i]     : number of hops back to root, or None for unembedded vectors

    Deterministic given the input — no randomness, no clustering, no
    parameters to tune. Same vectors always produce the same tree.
    """
    n = len(vectors)
    parents = [None] * n
    distances = [None] * n
    depths = [None] * n

    # Find the first embedded vector — it's the root
    root = next((i for i, v in enumerate(vectors) if v is not None), None)
    if root is None:
        return parents, distances, depths
    depths[root] = 0

    for i in range(root + 1, n):
        if vectors[i] is None:
            continue
        # Closest prior vector that's not None
        best_j, best_d = None, float('inf')
        for j in range(i):
            if vectors[j] is None:
                continue
            d = _cosine_distance(vectors[i], vectors[j])
            if d < best_d:
                best_j, best_d = j, d
        if best_j is None:
            continue
        parents[i] = best_j
        distances[i] = best_d
        depths[i] = (depths[best_j] or 0) + 1

    return parents, distances, depths


def pca_2d(vectors):
    """SVD-based 2D PCA. None vectors get [0.0, 0.0].

    Defensive against degenerate inputs: non-finite embedding components,
    rank-deficient matrices, and SVD failures all return zeros rather than
    propagating NaN/inf into the chart rendering.
    """
    import numpy as np
    valid_idx = [i for i, v in enumerate(vectors) if v is not None]
    if len(valid_idx) < 2:
        return [[0.0, 0.0] for _ in vectors], 0.0

    try:
        X = np.array([vectors[i] for i in valid_idx], dtype=float)
        # Scrub any non-finite components from the embedding matrix before
        # the SVD — a single NaN cascades through every projection.
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = X - X.mean(axis=0)

        with np.errstate(all='ignore'):
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
            if Vt.shape[0] < 2 or not np.isfinite(S[:2]).all():
                return [[0.0, 0.0] for _ in vectors], 0.0
            proj = X @ Vt[:2].T

        if not np.isfinite(proj).all():
            proj = np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)

        total = float((S ** 2).sum())
        var_explained = float((S[:2] ** 2).sum() / total) if total > 0 else 0.0
    except Exception:
        return [[0.0, 0.0] for _ in vectors], 0.0

    out = [[0.0, 0.0] for _ in vectors]
    for i, idx in enumerate(valid_idx):
        out[idx] = [float(proj[i, 0]), float(proj[i, 1])]
    return out, var_explained