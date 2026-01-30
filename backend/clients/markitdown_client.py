"""
MarkItDown Client for document extraction
Thread-safe singleton implementation
"""

import os
import tempfile
import threading
from typing import Optional
from pathlib import Path
from markitdown import MarkItDown
from app.logger import logger


class MarkItDownClient:
    """Thread-safe MarkItDown client for document extraction"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern with thread locking"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize MarkItDown client"""
        if not hasattr(self, '_initialized'):
            self.md = MarkItDown()
            self._extraction_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ MarkItDown client initialized")

    def extract_content(self, file_content: bytes, filename: str) -> str:
        """
        Extract content from file using MarkItDown with thread locking

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Extracted markdown text

        Raises:
            Exception: If extraction fails
        """
        with self._extraction_lock:
            try:
                extension = Path(filename).suffix.lower()

                # Write to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name

                try:
                    # Extract content
                    result = self.md.convert(tmp_file_path)
                    extracted_text = result.text_content

                    logger.info(
                        f"✅ MarkItDown extracted {len(extracted_text)} chars from {filename}"
                    )
                    return extracted_text

                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"❌ MarkItDown extraction failed for {filename}: {str(e)}")
                raise Exception(f"MarkItDown extraction failed: {str(e)}")

    @staticmethod
    def is_supported(extension: str) -> bool:
        """
        Check if file extension is supported by MarkItDown

        Args:
            extension: File extension with dot (e.g., '.txt')

        Returns:
            True if supported
        """
        supported_formats = [
            ".doc", ".docx", ".xlsx", ".xls", ".zip",
            ".md", ".markdown", ".txt", ".csv"
        ]
        return extension.lower() in supported_formats


# Singleton instance
_markitdown_client: Optional[MarkItDownClient] = None
_client_lock = threading.Lock()


def get_markitdown_client() -> MarkItDownClient:
    """
    Get or create thread-safe MarkItDownClient singleton instance

    Returns:
        MarkItDownClient: Singleton client instance
    """
    global _markitdown_client

    if _markitdown_client is None:
        with _client_lock:
            if _markitdown_client is None:
                _markitdown_client = MarkItDownClient()

    return _markitdown_client
