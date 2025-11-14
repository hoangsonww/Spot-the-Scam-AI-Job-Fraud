"""
Spot the Scam package initialization.

Provides utilities for reproducible training, evaluation, and inference of
uncertainty-aware fraud detection models for job postings.
"""

from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Return the package version, falling back to development."""
    try:
        return version("spot-the-scam")
    except PackageNotFoundError:
        return "0.0.0-dev"


__all__ = ["get_version"]
