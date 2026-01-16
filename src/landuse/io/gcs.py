"""
Google Cloud Storage (GCS) I/O utilities
Provides unified interface for reading/writing data to GCS
"""

import os
from pathlib import Path
from typing import Union, Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed. GCS functionality disabled.")


class GCSManager:
    """
    Manages Google Cloud Storage operations
    
    Supports:
    - Upload/download files
    - Open remote files (CSV, Parquet, GeoTIFF, NetCDF)
    - Blob existence checks
    - Batch operations
    """
    
    def __init__(self, bucket_name: str, project: Optional[str] = None):
        """
        Initialize GCS manager
        
        Args:
            bucket_name: GCS bucket name
            project: GCP project ID (optional, uses default credentials)
        """
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for GCS operations. "
                "Install with: pip install google-cloud-storage"
            )
        
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket_name)
        
        logger.info(f"GCS Manager initialized for bucket: {bucket_name}")
    
    def upload(self, local_path: Union[str, Path], gcs_path: str) -> str:
        """
        Upload file to GCS
        
        Args:
            local_path: Local file path
            gcs_path: Destination path in GCS (without gs:// prefix)
        
        Returns:
            Full GCS URI (gs://bucket/path)
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Uploaded {local_path} -> {gcs_uri}")
        return gcs_uri
    
    def download(self, gcs_path: str, local_path: Union[str, Path]) -> Path:
        """
        Download file from GCS
        
        Args:
            gcs_path: Source path in GCS (without gs:// prefix)
            local_path: Destination local file path
        
        Returns:
            Path to downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        
        logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} -> {local_path}")
        return local_path
    
    def exists(self, gcs_path: str) -> bool:
        """
        Check if blob exists in GCS
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
        
        Returns:
            True if blob exists
        """
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def list_blobs(self, prefix: str = "") -> list[str]:
        """
        List blobs with given prefix
        
        Args:
            prefix: Path prefix to filter
        
        Returns:
            List of blob paths
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
    
    def open_gcs(self, gcs_path: str, mode: str = "rb") -> BinaryIO:
        """
        Open GCS blob as file-like object
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
            mode: File mode ('rb' or 'wb')
        
        Returns:
            File-like object
        """
        blob = self.bucket.blob(gcs_path)
        return blob.open(mode)
    
    def delete(self, gcs_path: str) -> None:
        """
        Delete blob from GCS
        
        Args:
            gcs_path: Path in GCS (without gs:// prefix)
        """
        blob = self.bucket.blob(gcs_path)
        blob.delete()
        logger.info(f"Deleted gs://{self.bucket_name}/{gcs_path}")


# Convenience functions

def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """
    Parse GCS URI into bucket and path
    
    Args:
        uri: Full GCS URI (gs://bucket/path)
    
    Returns:
        Tuple of (bucket_name, path)
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    return bucket, path


def open_gcs(uri: str, mode: str = "rb", project: Optional[str] = None) -> BinaryIO:
    """
    Open GCS file by URI
    
    Args:
        uri: Full GCS URI (gs://bucket/path)
        mode: File mode ('rb' or 'wb')
        project: GCP project ID (optional)
    
    Returns:
        File-like object
    """
    bucket, path = parse_gcs_uri(uri)
    manager = GCSManager(bucket, project=project)
    return manager.open_gcs(path, mode=mode)


def upload_to_gcs(local_path: Union[str, Path], gcs_uri: str, 
                  project: Optional[str] = None) -> str:
    """
    Upload file to GCS by URI
    
    Args:
        local_path: Local file path
        gcs_uri: Destination GCS URI (gs://bucket/path)
        project: GCP project ID (optional)
    
    Returns:
        Full GCS URI
    """
    bucket, path = parse_gcs_uri(gcs_uri)
    manager = GCSManager(bucket, project=project)
    return manager.upload(local_path, path)


def download_from_gcs(gcs_uri: str, local_path: Union[str, Path],
                      project: Optional[str] = None) -> Path:
    """
    Download file from GCS by URI
    
    Args:
        gcs_uri: Source GCS URI (gs://bucket/path)
        local_path: Destination local path
        project: GCP project ID (optional)
    
    Returns:
        Path to downloaded file
    """
    bucket, path = parse_gcs_uri(gcs_uri)
    manager = GCSManager(bucket, project=project)
    return manager.download(path, local_path)


# Integration with xarray and rasterio

def open_netcdf_gcs(gcs_uri: str, project: Optional[str] = None):
    """
    Open NetCDF file from GCS using xarray
    
    Args:
        gcs_uri: GCS URI to NetCDF file
        project: GCP project ID (optional)
    
    Returns:
        xarray.Dataset
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray required for NetCDF support")
    
    # xarray supports gcsfs directly
    return xr.open_dataset(gcs_uri, engine="netcdf4")


def open_raster_gcs(gcs_uri: str, project: Optional[str] = None):
    """
    Open raster file from GCS using rasterio
    
    Args:
        gcs_uri: GCS URI to raster file
        project: GCP project ID (optional)
    
    Returns:
        rasterio dataset
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required for raster support")
    
    # rasterio supports /vsigs/ virtual file system
    vsi_path = gcs_uri.replace("gs://", "/vsigs/")
    return rasterio.open(vsi_path)
