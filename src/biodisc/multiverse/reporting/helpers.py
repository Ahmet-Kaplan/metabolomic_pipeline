"""Shared geometric / statistical helpers for the path-report figures.

Currently:
  * ``confidence_ellipse_2d`` — 95 % confidence ellipse for a 2D point cloud.
    Used by the PCA-with-class-ellipses plot (Kirwan 2013) and any score-plot
    style figure.
"""

from __future__ import annotations

import numpy as np

# 95% confidence ellipse for a bivariate normal: scaling factor for the
# semi-axes derived from the chi-squared distribution with 2 d.o.f.
# χ²₂,₀.₉₅ = 5.991, so semi-axis = sqrt(5.991 × eigenvalue) ≈ 2.4477 × sqrt(eigenvalue).
CHI2_95_2DOF: float = float(np.sqrt(5.991))


def confidence_ellipse_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence: float = 0.95,
    n_points: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (rx, ry) coordinates for a confidence ellipse around (x, y).

    Standard eigendecomposition-of-covariance approach. Works on any
    bivariate point cloud with at least 3 points; below that the ellipse
    is degenerate and we return empty arrays so callers can skip drawing.

    Parameters
    ----------
    x, y
        1-D arrays of equal length — the point cloud.
    confidence
        Confidence level. Currently only 0.95 is supported (chi-squared 2-df
        critical value precomputed). Other levels would require scipy.
    n_points
        Resolution of the ellipse polyline.

    Returns
    -------
    (rx, ry)
        Two arrays of length ``n_points`` (or empty if degenerate) tracing
        the ellipse perimeter.
    """
    if confidence != 0.95:
        raise ValueError("Only 0.95 confidence is supported (chi-squared 2-df).")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        return np.array([]), np.array([])

    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return np.array([]), np.array([])

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns in ascending order; sort descending so axis 0 = major
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    semi_major = CHI2_95_2DOF * float(np.sqrt(eigvals[0]))
    semi_minor = CHI2_95_2DOF * float(np.sqrt(eigvals[1]))

    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    ex = semi_major * np.cos(t)
    ey = semi_minor * np.sin(t)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = cos_a * ex - sin_a * ey + float(x.mean())
    ry = sin_a * ex + cos_a * ey + float(y.mean())
    return rx, ry


def slugify(name: str) -> str:
    """Convert a human title to a filesystem-safe slug for figure paths."""
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
