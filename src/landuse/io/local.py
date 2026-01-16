"""
Local file system I/O utilities
"""

import os
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


class LocalManager:
    """
    Manages local file system operations
    Mirrors GCSManager interface for consistency
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize local manager
        
        Args:
            base_dir: Base directory for all operations
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local Manager initialized at: {self.base_dir}")
    
    def get_path(self, rel_path: str) -> Path:
        """
        Get absolute path from relative path
        
        Args:
            rel_path: Relative path from base_dir
        
        Returns:
            Absolute path
        """
        return self.base_dir / rel_path
    
    def exists(self, rel_path: str) -> bool:
        """
        Check if file exists
        
        Args:
            rel_path: Relative path from base_dir
        
        Returns:
            True if file exists
        """
        return self.get_path(rel_path).exists()
    
    def list_files(self, pattern: str = "*") -> list[Path]:
        """
        List files matching pattern
        
        Args:
            pattern: Glob pattern
        
        Returns:
            List of matching file paths
        """
        return list(self.base_dir.glob(pattern))
    
    def ensure_dir(self, rel_path: str) -> Path:
        """
        Ensure directory exists
        
        Args:
            rel_path: Relative path from base_dir
        
        Returns:
            Path to directory
        """
        dir_path = self.get_path(rel_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
