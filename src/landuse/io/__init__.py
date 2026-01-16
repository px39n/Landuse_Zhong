"""
I/O abstraction layer for local and cloud storage
"""

from .gcs import GCSManager, open_gcs, upload_to_gcs, download_from_gcs
from .local import LocalManager

__all__ = [
    "GCSManager",
    "open_gcs",
    "upload_to_gcs",
    "download_from_gcs",
    "LocalManager",
]
