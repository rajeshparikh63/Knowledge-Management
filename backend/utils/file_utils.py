"""
File utilities for extracting raw content from various file types
Uses MarkItDown for simple formats and Unstructured API for complex documents
"""

from pathlib import Path
from typing import Tuple
from clients.markitdown_client import get_markitdown_client, MarkItDownClient
from clients.unstructured_client import get_unstructured_client, UnstructuredClient
from clients.image_analysis_client import get_image_analysis_client
from clients.groq_whisper_client import get_groq_whisper_client
from clients.video_processor_client import get_video_processor_client
from app.logger import logger


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename

    Args:
        filename: Name of the file

    Returns:
        File extension with dot (e.g., '.pdf', '.txt')
    """
    return Path(filename).suffix.lower()


def is_image_file(extension: str) -> bool:
    """
    Check if file is an image

    Args:
        extension: File extension with dot

    Returns:
        True if image file
    """
    image_formats = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",".heic", ".tiff"]
    return extension in image_formats


def is_video_file(extension: str) -> bool:
    """
    Check if file is a video

    Args:
        extension: File extension with dot

    Returns:
        True if video file
    """
    video_formats = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]
    return extension in video_formats


def is_audio_file(extension: str) -> bool:
    """
    Check if file is audio

    Args:
        extension: File extension with dot

    Returns:
        True if audio file
    """
    audio_formats = [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]
    return extension in audio_formats


def extract_raw_data(file_content: bytes, file_name: str, folder_name: str = "files") -> str:
    """
    Extract raw text content from file

    Routes to appropriate extraction method based on file type:
    - MarkItDown: .docx, .xlsx, .xls, .zip, .md, .markdown, .txt, .csv
    - Unstructured API: .pdf, .ppt, .pptx, .doc, .rtf and other complex docs
    - Skip: images and videos (not supported yet)

    Args:
        file_content: File content as bytes
        file_name: Original filename

    Returns:
        Extracted raw text content

    Raises:
        Exception: If extraction fails or file type not supported
    """
    logger.info(f"📄 Extracting raw data from: {file_name}")

    # Get file extension
    extension = get_file_extension(file_name)

    # Check if image file - use image analysis
    if is_image_file(extension):
        logger.info(f"🖼️ Processing image file with vision model: {file_name}")
        image_client = get_image_analysis_client()
        return image_client.analyze_image(file_content, file_name)

    # Check if video file - process with video processor
    if is_video_file(extension):
        logger.info(f"🎬 Processing video file: {file_name}")
        video_client = get_video_processor_client()
        result = video_client.process_video(file_content, file_name, folder_name)
        # Return special format: combined_text for MongoDB + chunks for Pinecone
        return {
            'type': 'video',
            'combined_text': result['combined_text'],
            'chunks': result['chunks']
        }

    # Check if audio file - use Groq Whisper for transcription
    if is_audio_file(extension):
        logger.info(f"🎵 Transcribing audio file with Groq Whisper: {file_name}")
        whisper_client = get_groq_whisper_client()
        return whisper_client.transcribe_audio(file_content, file_name)

    # Get clients
    markitdown_client = get_markitdown_client()
    unstructured_client = get_unstructured_client()

    # Use MarkItDown for simple formats
    if MarkItDownClient.is_supported(extension):
        logger.info(f"🔧 Using MarkItDown for {file_name}")
        return markitdown_client.extract_content(file_content, file_name)

    # Use Unstructured API for complex documents
    elif UnstructuredClient.is_supported(extension):
        logger.info(f"🔧 Using Unstructured API for {file_name}")
        return unstructured_client.extract_content(file_content, file_name)

    else:
        raise ValueError(f"Unsupported file type: {extension} for file: {file_name}")


def validate_extracted_content(raw_content: str, min_length: int = 10) -> bool:
    """
    Validate that extracted content is usable

    Args:
        raw_content: Extracted text content
        min_length: Minimum acceptable length

    Returns:
        True if valid, False otherwise
    """
    if not raw_content:
        return False

    if not isinstance(raw_content, str):
        return False

    if len(raw_content.strip()) < min_length:
        return False

    # Check if it's a skip message
    if raw_content.startswith("[") and "skipped" in raw_content.lower():
        return False

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    safe_name = filename

    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')

    return safe_name


def get_file_size_mb(file_content: bytes) -> float:
    """
    Get file size in megabytes

    Args:
        file_content: File content bytes

    Returns:
        File size in MB
    """
    return len(file_content) / (1024 * 1024)
