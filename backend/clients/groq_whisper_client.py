"""
Groq Whisper Client for audio transcription
Thread-safe implementation using Whisper Large V3
"""

import tempfile
import threading
from typing import Optional
from pathlib import Path
from groq import Groq
from app.logger import logger
from app.settings import settings


class GroqWhisperClient:
    """Thread-safe Groq Whisper client for audio transcription"""

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
        """Initialize Groq client for Whisper"""
        if not hasattr(self, '_initialized'):
            self.api_key = settings.GROQ_API_KEY

            if not self.api_key:
                raise ValueError("GROQ_API_KEY not configured in settings")

            # Initialize Groq client
            self.client = Groq(api_key=self.api_key)
            self._transcription_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Groq Whisper client initialized")

    def transcribe_audio(self, file_content: bytes, filename: str) -> str:
        """
        Transcribe audio file using Groq's Whisper Large V3

        Args:
            file_content: Audio file content as bytes
            filename: Original filename

        Returns:
            Transcribed text (plain text, no timestamps)

        Raises:
            Exception: If transcription fails
        """
        segments = self.transcribe_audio_with_timestamps(file_content, filename)
        # Combine all segment texts
        return " ".join([seg['text'] for seg in segments])

    def transcribe_audio_with_timestamps(self, file_content: bytes, filename: str) -> list:
        """
        Transcribe audio file with timestamps using Groq's Whisper Large V3

        Args:
            file_content: Audio file content as bytes
            filename: Original filename

        Returns:
            List of segments with timestamps:
            [{'start': float, 'end': float, 'text': str}, ...]

        Raises:
            Exception: If transcription fails
        """
        with self._transcription_lock:
            try:
                extension = Path(filename).suffix.lower()

                # Write to temp file (Groq API requires file-like object)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    # Transcribe using Groq Whisper Large V3 with verbose JSON for timestamps
                    with open(tmp_file_path, "rb") as audio_file:
                        transcription = self.client.audio.transcriptions.create(
                            file=(filename, audio_file.read()),
                            model="whisper-large-v3",
                            response_format="verbose_json",  # Get timestamps
                            language="en",  # Optional: specify language or let it auto-detect
                            temperature=0.0
                        )

                    # Extract segments with timestamps
                    segments = []
                    if hasattr(transcription, 'segments') and transcription.segments:
                        for seg in transcription.segments:
                            segments.append({
                                'start': seg.start,
                                'end': seg.end,
                                'text': seg.text.strip()
                            })
                    else:
                        # Fallback if no segments (shouldn't happen with verbose_json)
                        logger.warning(f"No segments returned for {filename}, using full text")
                        text = transcription.text if hasattr(transcription, 'text') else str(transcription)
                        segments = [{'start': 0.0, 'end': 0.0, 'text': text}]

                    logger.info(f"✅ Groq Whisper transcribed {len(segments)} segments from {filename}")
                    return segments

                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"❌ Groq Whisper transcription failed for {filename}: {str(e)}")
                raise Exception(f"Audio transcription failed: {str(e)}")

    @staticmethod
    def is_supported(extension: str) -> bool:
        """
        Check if file extension is supported for audio transcription

        Args:
            extension: File extension with dot (e.g., '.mp3')

        Returns:
            True if supported
        """
        # Groq Whisper supports common audio formats
        supported_formats = [
            ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
            ".wav", ".webm", ".flac", ".ogg", ".aac"
        ]
        return extension.lower() in supported_formats


# Singleton instance
_groq_whisper_client: Optional[GroqWhisperClient] = None
_client_lock = threading.Lock()


def get_groq_whisper_client() -> GroqWhisperClient:
    """
    Get or create thread-safe GroqWhisperClient singleton instance

    Returns:
        GroqWhisperClient: Singleton client instance
    """
    global _groq_whisper_client

    if _groq_whisper_client is None:
        with _client_lock:
            if _groq_whisper_client is None:
                _groq_whisper_client = GroqWhisperClient()

    return _groq_whisper_client
