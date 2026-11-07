"""Stage 3–10 feature-level filters.

Every filter module in this package follows the same output contract:

    kept_features.parquet   — single-column: feature_id survivors
    metrics.parquet         — per-feature diagnostic metric(s)
    audit/<stage>.parquet   — one row per feature with action + reason

Downstream stages load the full matrix via join on ``kept_features``.
No matrix ever gets rewritten across stages.
"""

from __future__ import annotations
