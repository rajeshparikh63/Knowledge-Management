"""
Unstructured API Client for complex document extraction
Thread-safe singleton implementation with fast strategy
"""

import os
import tempfile
import threading
from typing import Optional
from pathlib import Path
from unstructured_client import UnstructuredClient as UnstructuredAPIClient
from unstructured_client.models import shared
from app.logger import logger
from app.settings import settings


class UnstructuredClient:
    """Thread-safe Unstructured API client for document extraction"""

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
        """Initialize Unstructured API client"""
        if not hasattr(self, '_initialized'):
            self.api_key = settings.UNSTRUCTURED_API_KEY
            self.api_url = settings.UNSTRUCTURED_API_URL

            if not self.api_key:
                raise ValueError("UNSTRUCTURED_API_KEY not configured in settings")

            # Initialize client
            self.client = UnstructuredAPIClient(
                api_key_auth=self.api_key,
                server_url=self.api_url if self.api_url else None
            )

            self._extraction_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Unstructured API client initialized")

    def extract_content(self, file_content: bytes, filename: str) -> str:
        """
        Extract content from file using Unstructured API with fast strategy

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Extracted text

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
                    # Read file and partition with fast strategy
                    with open(tmp_file_path, "rb") as f:
                        # Correct API structure: dictionary with nested partition_parameters
                        req = {
                            "partition_parameters": {
                                "files": {
                                    "content": f.read(),
                                    "file_name": filename,
                                },
                                "strategy": shared.Strategy.FAST,  # Fast strategy for speed
                                "languages": ["eng"],
                            }
                        }

                    # Call Unstructured API
                    res = self.client.general.partition(request=req)

                    # Extract text from elements
                    extracted_text = "\n\n".join([
                        element.get("text", "")
                        for element in res.elements
                        if element.get("text")
                    ])

                    logger.info(
                        f"✅ Unstructured API (fast) extracted {len(extracted_text)} chars from {filename}"
                    )
                    return extracted_text

                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"❌ Unstructured API extraction failed for {filename}: {str(e)}")
                raise Exception(f"Unstructured extraction failed: {str(e)}")

    @staticmethod
    def is_supported(extension: str) -> bool:
        """
        Check if file extension is supported by Unstructured API

        Args:
            extension: File extension with dot (e.g., '.pdf')

        Returns:
            True if supported
        """
        # Unstructured API supported formats (excluding what MarkItDown handles and media files)
        supported_formats = [
            # Documents
            ".pdf", ".dot", ".docm", ".dotm", ".rtf", ".odt",
            # Presentations
            ".ppt", ".pptx", ".pptm", ".pot", ".potx", ".potm",
            # HTML/Web
            ".html", ".htm", ".xml",
            # E-books and other
            ".epub", ".rst", ".org",
            # Email
            ".eml", ".msg", ".p7s",
            # Specialized formats
            ".abw", ".zabw", ".cwk", ".mcw", ".mw", ".hwp",
            # Spreadsheets (non-Excel)
            ".et", ".fods", ".tsv", ".dbf",
            # Other
            ".dif", ".eth", ".pbd", ".sdp", ".sxg", ".prn",
            # Images (Unstructured can extract text from images)
            
        ]
        return extension.lower() in supported_formats


# Singleton instance
_unstructured_client: Optional[UnstructuredClient] = None
_client_lock = threading.Lock()


def get_unstructured_client() -> UnstructuredClient:
    """
    Get or create thread-safe UnstructuredClient singleton instance

    Returns:
        UnstructuredClient: Singleton client instance
    """
    global _unstructured_client

    if _unstructured_client is None:
        with _client_lock:
            if _unstructured_client is None:
                _unstructured_client = UnstructuredClient()

    return _unstructured_client
