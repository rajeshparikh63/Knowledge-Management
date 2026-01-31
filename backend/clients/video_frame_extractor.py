"""
Video Frame Extractor Client
Extracts frames from video files at target FPS using OpenCV
Thread-safe singleton implementation with memory optimization
"""

import cv2
import tempfile
import threading
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from app.logger import logger
from app.settings import settings


class VideoFrameExtractor:
    """Thread-safe video frame extractor using OpenCV"""

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
        """Initialize frame extractor"""
        if not hasattr(self, '_initialized'):
            self._extraction_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Video frame extractor initialized")

    def extract_frames(
        self,
        file_content: bytes,
        filename: str,
        target_fps: Optional[int] = None
    ) -> List[Dict]:
        """
        Extract frames from video at target FPS

        Args:
            file_content: Video file content as bytes
            filename: Original filename
            target_fps: Target frames per second (default: from settings)

        Returns:
            List of frame dictionaries with:
            - frame_number: int
            - timestamp: float (seconds)
            - gray: ndarray (grayscale image for SSIM)

        Raises:
            Exception: If extraction fails
        """
        with self._extraction_lock:
            try:
                target_fps = target_fps or settings.VIDEO_TARGET_FPS
                extension = Path(filename).suffix.lower()

                # Write to temp file (OpenCV needs file path)
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    # Open video file
                    cap = cv2.VideoCapture(tmp_file_path)

                    if not cap.isOpened():
                        raise Exception(f"Failed to open video file: {filename}")

                    # Get video properties
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / original_fps if original_fps > 0 else 0

                    logger.info(
                        f"📹 Video info: {filename} - "
                        f"FPS={original_fps:.2f}, "
                        f"Frames={total_frames}, "
                        f"Duration={duration:.2f}s"
                    )

                    # Calculate frame skip
                    frame_skip = int(original_fps / target_fps) if original_fps > target_fps else 1

                    # Extract frames
                    frames = []
                    frame_count = 0
                    extracted_count = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Extract frame at target FPS
                        if frame_count % frame_skip == 0:
                            # Convert to grayscale for SSIM comparison
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            timestamp = frame_count / original_fps if original_fps > 0 else 0

                            frames.append({
                                'frame_number': frame_count,
                                'timestamp': timestamp,
                                'gray': gray  # Grayscale for SSIM
                            })
                            extracted_count += 1

                            # Log progress every 10%
                            if extracted_count % max(1, int(total_frames / (frame_skip * 10))) == 0:
                                progress = (frame_count / total_frames) * 100
                                logger.info(f"⏳ Frame extraction progress: {progress:.1f}% ({extracted_count} frames)")

                        frame_count += 1

                    cap.release()

                    logger.info(
                        f"✅ Extracted {len(frames)} frames from {filename} "
                        f"(target FPS: {target_fps})"
                    )

                    return frames

                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(f"❌ Frame extraction failed for {filename}: {str(e)}")
                raise Exception(f"Frame extraction failed: {str(e)}")

    def extract_color_frame(
        self,
        file_content: bytes,
        filename: str,
        frame_number: int
    ) -> np.ndarray:
        """
        Extract a specific frame in full color

        Used to extract key frames after scene detection

        Args:
            file_content: Video file content as bytes
            filename: Original filename
            frame_number: Frame number to extract

        Returns:
            Color frame as BGR numpy array

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
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    # Open video and seek to frame
                    cap = cv2.VideoCapture(tmp_file_path)

                    if not cap.isOpened():
                        raise Exception(f"Failed to open video file: {filename}")

                    # Seek to specific frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                    ret, frame = cap.read()
                    if not ret:
                        raise Exception(f"Failed to read frame {frame_number}")

                    cap.release()

                    return frame  # Return full color BGR image

                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

            except Exception as e:
                logger.error(
                    f"❌ Color frame extraction failed for {filename} "
                    f"frame {frame_number}: {str(e)}"
                )
                raise Exception(f"Color frame extraction failed: {str(e)}")


# Singleton instance
_video_frame_extractor: Optional[VideoFrameExtractor] = None
_client_lock = threading.Lock()


def get_video_frame_extractor() -> VideoFrameExtractor:
    """
    Get or create thread-safe VideoFrameExtractor singleton instance

    Returns:
        VideoFrameExtractor: Singleton client instance
    """
    global _video_frame_extractor

    if _video_frame_extractor is None:
        with _client_lock:
            if _video_frame_extractor is None:
                _video_frame_extractor = VideoFrameExtractor()

    return _video_frame_extractor
