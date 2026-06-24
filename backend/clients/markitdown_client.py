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

# Plain-text formats we handle ourselves — MarkItDown's PlainTextConverter
# falls back to the system default codec (which on macOS without LC_ALL set
# is ASCII), so any UTF-8 file with smart quotes / em-dashes / bullets
# (byte 0xe2 onward) blows up with UnicodeDecodeError. These formats ARE
# the text already; there's nothing for MarkItDown to convert.
_PLAIN_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".log", ".tsv"}

# Encoding waterfall — tried in order. cp1252 covers most Windows-authored
# Western text; latin-1 NEVER raises (1-byte mapping), so it's the always-works
# safety net before our final replace-bad-bytes fallback.
_TEXT_DECODE_ATTEMPTS = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")


def _decode_text_bytes(data: bytes, filename: str) -> str:
    """Best-effort text decode with a defined waterfall.

    Tries UTF-8 (with optional BOM) first, then UTF-16 (catches Windows
    Notepad's "Unicode" save), then cp1252 (Windows Western), then latin-1
    (which never raises). If somehow all fail, returns UTF-8 with
    `errors='replace'` so we never lose the document — bad bytes become `�`.
    """
    # Encodings considered "expected" — no log unless we fall past them.
    utf8_family = {"utf-8", "utf-8-sig", "utf-16"}
    for enc in _TEXT_DECODE_ATTEMPTS:
        try:
            text = data.decode(enc)
            if enc not in utf8_family:
                logger.info(
                    f"📝 Decoded {filename} as {enc} "
                    f"(UTF-8 failed; likely Windows-authored file)"
                )
            return text
        except UnicodeDecodeError:
            continue
    # Should be unreachable since latin-1 can't fail, but be defensive.
    logger.warning(
        f"⚠️ All encodings failed for {filename}; falling back to UTF-8 with "
        f"replacement characters"
    )
    return data.decode("utf-8", errors="replace")


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

                # Fast path: plain-text formats — decode bytes ourselves with
                # a robust encoding waterfall. MarkItDown's PlainTextConverter
                # uses the system default codec and fails on non-ASCII UTF-8
                # (smart quotes, em-dashes, bullets), which is most real text.
                if extension in _PLAIN_TEXT_EXTENSIONS:
                    extracted_text = _decode_text_bytes(file_content, filename)
                    logger.info(
                        f"✅ Plain-text decode extracted {len(extracted_text)} "
                        f"chars from {filename} ({extension})"
                    )
                    return extracted_text

                # Write to temp file (MarkItDown wants a path for non-text formats)
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
