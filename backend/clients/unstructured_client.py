"""
Unstructured API Client for complex document extraction
Fresh instance per task for Celery compatibility
"""

from pathlib import Path
from unstructured_client import UnstructuredClient as UnstructuredAPIClient
from unstructured_client.models import shared
from app.logger import logger
from app.settings import settings


class UnstructuredClient:
    """Unstructured API client for document extraction (no singleton for Celery)"""

    def __init__(self):
        """Initialize Unstructured API client"""
        self.api_key = settings.UNSTRUCTURED_API_KEY
        self.api_url = settings.UNSTRUCTURED_API_URL

        if not self.api_key:
            raise ValueError("UNSTRUCTURED_API_KEY not configured in settings")

        # Initialize client with default HTTP client (more stable for HI_RES processing)
        self.client = UnstructuredAPIClient(
            api_key_auth=self.api_key,
            server_url=self.api_url if self.api_url else None
        )

        logger.info("✅ Unstructured API client initialized")

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'client') and self.client:
                # Close the client's internal HTTP client if available
                if hasattr(self.client, 'sdk_configuration') and hasattr(self.client.sdk_configuration, 'client'):
                    self.client.sdk_configuration.client.close()
                logger.info("✅ Closed Unstructured client")
        except Exception as e:
            logger.warning(f"Error cleaning up Unstructured client: {str(e)}")

    def extract_content(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from a file.

        For PDFs we first try Unstructured's FAST strategy (reads the embedded
        text layer — instant and cheap for normal documents). If that comes back
        essentially empty, the PDF is image-based / scanned (e.g. a designed
        brochure where the text is baked into the artwork), so we fall back to
        Unstructured's native VLM strategy (a vision model reads the rendered
        pages).

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Extracted text

        Raises:
            Exception: If extraction fails
        """
        extension = Path(filename).suffix.lower()

        # 1. FAST text-layer extraction — works for normal PDFs and other formats.
        fast_text = ""
        try:
            fast_text = self._extract_fast(file_content, filename)
        except Exception as e:
            # PDFs have a VLM fallback below; anything else is a hard failure.
            if extension != ".pdf":
                logger.error(f"❌ Unstructured API extraction failed for {filename}: {str(e)}")
                raise Exception(f"Unstructured extraction failed: {str(e)}")
            logger.warning(f"⚠️ FAST extraction errored for {filename} ({str(e)}); will try VLM")

        # 2. Image/scanned PDF → FAST yields ~nothing → Unstructured VLM strategy.
        if extension == ".pdf" and len(fast_text.strip()) < 50:
            logger.info(
                f"🔎 FAST extracted only {len(fast_text.strip())} chars from {filename} "
                f"— treating as an image PDF and re-parsing with the VLM strategy"
            )
            try:
                vlm_text = self._extract_vlm(file_content, filename)
                if len(vlm_text.strip()) > len(fast_text.strip()):
                    return vlm_text
            except Exception as e:
                logger.error(f"❌ VLM extraction failed for {filename}: {str(e)}")

        return fast_text

    def _extract_vlm(self, file_content: bytes, filename: str) -> str:
        """Unstructured API VLM strategy — a vision model reads the rendered
        pages. Best for image-based / scanned PDFs with no text layer."""
        req = {
            "partition_parameters": {
                "files": {
                    "content": file_content,
                    "file_name": filename,
                },
                "strategy": shared.Strategy.VLM,
                "vlm_model_provider": settings.VLM_MODEL_PROVIDER,
                "vlm_model": settings.VLM_MODEL,
                "split_pdf_page": False,
                "split_pdf_allow_failed": False,
                "split_pdf_concurrency_level": 1,
            }
        }

        res = self.client.general.partition(request=req)

        extracted_text = "\n\n".join([
            element.get("text", "")
            for element in res.elements
            if element.get("text")
        ])

        logger.info(
            f"✅ Unstructured API (VLM:{settings.VLM_MODEL}) extracted "
            f"{len(extracted_text)} chars from {filename}"
        )

        return extracted_text

    def _extract_fast(self, file_content: bytes, filename: str) -> str:
        """Unstructured API FAST strategy — embedded text-layer extraction only."""
        req = {
            "partition_parameters": {
                "files": {
                    "content": file_content,
                    "file_name": filename,
                },
                "strategy": shared.Strategy.FAST,
                "split_pdf_page": False,
                "split_pdf_allow_failed": False,
                "split_pdf_concurrency_level": 1,
            }
        }

        res = self.client.general.partition(request=req)

        extracted_text = "\n\n".join([
            element.get("text", "")
            for element in res.elements
            if element.get("text")
        ])

        logger.info(
            f"✅ Unstructured API (fast) extracted {len(extracted_text)} chars from {filename}"
        )

        return extracted_text

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


def get_unstructured_client() -> UnstructuredClient:
    """
    Create a fresh UnstructuredClient instance (no caching for Celery)

    Returns:
        UnstructuredClient: Fresh client instance
    """
    return UnstructuredClient()
