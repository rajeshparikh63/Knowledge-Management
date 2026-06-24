"""
PDF Image Extractor using PyMuPDF
Extracts images from PDFs and analyzes them in parallel (max 5 at a time)
"""

import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from app.logger import logger
from clients.image_analysis_client import get_image_analysis_client


class PDFImageExtractor:
    """Extract and analyze images from PDF files"""

    def __init__(self):
        """Initialize PDF image extractor"""
        self.image_analyzer = get_image_analysis_client()

    def extract_and_analyze_images(self, pdf_content: bytes, filename: str) -> str:
        """
        Extract images from PDF and analyze them (5 at a time in parallel)

        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename

        Returns:
            Combined text with image analyses
        """
        try:
            # Open PDF and extract all images
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            images = []

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        image_bytes = pdf_document.extract_image(xref)["image"]
                        images.append({
                            "bytes": image_bytes,
                            "page": page_num + 1,
                            "num": len(images) + 1
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract image: {str(e)}")

            pdf_document.close()

            if not images:
                return ""

            logger.info(f"📸 Analyzing {len(images)} images from {filename}")

            # Analyze images in parallel (max 5 concurrent)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self._analyze_image, img["bytes"], img["num"], img["page"], filename)
                    for img in images
                ]
                results = [f.result() for f in futures]

            logger.info(f"✅ Analyzed {len(images)} images from {filename}")
            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"❌ PDF image extraction failed: {str(e)}")
            raise

    def extract_pages_via_vlm(self, pdf_content: bytes, filename: str, dpi: int = 170) -> str:
        """
        Transcribe a PDF by rendering each FULL page to an image and sending it
        to the vision LLM.

        Unlike extract_and_analyze_images (which only pulls embedded xobject
        images), this rasterizes the WHOLE page — so it captures vector text
        drawn over background artwork, which is exactly how image-based
        brochures and scanned PDFs are built. Use this when the normal
        text-layer extraction (Unstructured FAST) comes back empty.
        """
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        except Exception as e:
            logger.error(f"❌ VLM extract: cannot open PDF {filename}: {str(e)}")
            raise

        try:
            zoom = dpi / 72.0  # 72 = PDF's native points-per-inch
            matrix = fitz.Matrix(zoom, zoom)
            pages = []
            for page_num in range(len(pdf_document)):
                try:
                    pix = pdf_document[page_num].get_pixmap(matrix=matrix, alpha=False)
                    pages.append({"bytes": pix.tobytes("png"), "page": page_num + 1})
                except Exception as e:
                    logger.warning(
                        f"⚠️ VLM render failed for {filename} page {page_num + 1}: {str(e)}"
                    )
        finally:
            pdf_document.close()

        if not pages:
            return ""

        logger.info(f"🖼️ VLM transcribing {len(pages)} page(s) from {filename}")

        def _transcribe(p) -> str:
            try:
                text = self.image_analyzer.analyze_image(
                    p["bytes"], f"{filename}#p{p['page']}"
                )
                text = (text or "").strip()
                return f"[Page {p['page']}]\n{text}" if text else ""
            except Exception as e:
                logger.warning(
                    f"⚠️ VLM transcription failed for {filename} page {p['page']}: {str(e)}"
                )
                return ""

        # Pages transcribe in parallel; executor.map preserves page order.
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(_transcribe, pages))

        combined = "\n\n".join(r for r in results if r)
        logger.info(
            f"✅ VLM extracted {len(combined)} chars from {filename} "
            f"across {len(pages)} page(s)"
        )
        return combined

    def _analyze_image(self, image_bytes: bytes, num: int, page: int, filename: str) -> str:
        """Analyze a single image"""
        try:
            description = self.image_analyzer.analyze_image(image_bytes, f"{filename}_img{num}")
            return f"\n[IMAGE {num} - Page {page}]\n{description}\n[END IMAGE {num}]\n"
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze image {num}: {str(e)}")
            return f"\n[IMAGE {num} - Analysis Failed]\n"


def get_pdf_image_extractor() -> PDFImageExtractor:
    """
    Get PDF image extractor instance

    Returns:
        PDFImageExtractor: Extractor instance
    """
    return PDFImageExtractor()
