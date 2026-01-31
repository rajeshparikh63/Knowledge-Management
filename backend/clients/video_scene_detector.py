"""
Video Scene Detector Client
Detects scene boundaries using SSIM and selects key frames using Shannon entropy
Thread-safe singleton implementation
"""

import threading
from typing import List, Dict, Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim
from app.logger import logger
from app.settings import settings


class VideoSceneDetector:
    """Thread-safe video scene detector using SSIM and entropy"""

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
        """Initialize scene detector"""
        if not hasattr(self, '_initialized'):
            self._detection_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Video scene detector initialized")

    def detect_scenes(
        self,
        frames: List[Dict],
        ssim_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect scene boundaries using SSIM comparison

        Args:
            frames: List of frame dicts with 'gray' images
            ssim_threshold: Similarity threshold (default: from settings)

        Returns:
            List of scene dictionaries with:
            - scene_id: int
            - start_time: float
            - end_time: float
            - frames: list of frame dicts
            - num_frames: int

        Raises:
            Exception: If detection fails
        """
        with self._detection_lock:
            try:
                if not frames:
                    return []

                ssim_threshold = ssim_threshold or settings.VIDEO_SSIM_THRESHOLD

                logger.info(f"🔍 Detecting scenes in {len(frames)} frames (SSIM threshold: {ssim_threshold})")

                # Find scene boundaries
                scene_boundaries = [0]  # Start with first frame

                for i in range(1, len(frames)):
                    # Compare consecutive frames using SSIM
                    similarity = ssim(
                        frames[i-1]['gray'],
                        frames[i]['gray'],
                        data_range=255
                    )

                    # If similarity drops below threshold, mark scene boundary
                    if similarity < ssim_threshold:
                        scene_boundaries.append(i)
                        logger.debug(
                            f"Scene boundary detected at frame {i} "
                            f"(SSIM: {similarity:.3f})"
                        )

                # Add last frame as boundary
                scene_boundaries.append(len(frames))

                # Group frames into scenes
                scenes = []
                for scene_id in range(len(scene_boundaries) - 1):
                    start_idx = scene_boundaries[scene_id]
                    end_idx = scene_boundaries[scene_id + 1]

                    scene_frames = frames[start_idx:end_idx]

                    scenes.append({
                        'scene_id': scene_id,
                        'start_time': scene_frames[0]['timestamp'],
                        'end_time': scene_frames[-1]['timestamp'],
                        'frames': scene_frames,
                        'num_frames': len(scene_frames)
                    })

                logger.info(
                    f"✅ Detected {len(scenes)} scenes "
                    f"(avg {len(frames) / len(scenes):.1f} frames per scene)"
                )

                return scenes

            except Exception as e:
                logger.error(f"❌ Scene detection failed: {str(e)}")
                raise Exception(f"Scene detection failed: {str(e)}")

    def select_key_frames(self, scenes: List[Dict]) -> List[Dict]:
        """
        Select one key frame per scene using Shannon entropy

        Entropy measures information content - higher = more detail

        Args:
            scenes: List of scene dicts from detect_scenes()

        Returns:
            List of key frame dictionaries with:
            - frame_number: int
            - timestamp: float
            - scene_id: int
            - scene_start: float
            - scene_end: float
            - entropy: float

        Raises:
            Exception: If selection fails
        """
        with self._detection_lock:
            try:
                logger.info(f"🔑 Selecting key frames from {len(scenes)} scenes")

                key_frames = []

                for scene in scenes:
                    # Calculate entropy for each frame in scene
                    frame_entropies = []

                    for frame in scene['frames']:
                        entropy = self._calculate_entropy(frame['gray'])
                        frame_entropies.append({
                            'frame': frame,
                            'entropy': entropy
                        })

                    # Select frame with highest entropy
                    if frame_entropies:
                        best_frame_data = max(frame_entropies, key=lambda x: x['entropy'])
                        best_frame = best_frame_data['frame']

                        key_frames.append({
                            'frame_number': best_frame['frame_number'],
                            'timestamp': best_frame['timestamp'],
                            'scene_id': scene['scene_id'],
                            'scene_start': scene['start_time'],
                            'scene_end': scene['end_time'],
                            'entropy': best_frame_data['entropy']
                        })

                        logger.debug(
                            f"Key frame selected for scene {scene['scene_id']}: "
                            f"frame {best_frame['frame_number']} "
                            f"(entropy: {best_frame_data['entropy']:.2f})"
                        )

                logger.info(f"✅ Selected {len(key_frames)} key frames")

                return key_frames

            except Exception as e:
                logger.error(f"❌ Key frame selection failed: {str(e)}")
                raise Exception(f"Key frame selection failed: {str(e)}")

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate Shannon entropy of an image

        Entropy = -sum(p * log2(p)) where p is histogram probabilities

        Args:
            image: Grayscale image as numpy array

        Returns:
            Entropy value (higher = more information)
        """
        # Calculate histogram
        histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

        # Normalize to probabilities
        histogram = histogram / histogram.sum()

        # Remove zero probabilities (log(0) is undefined)
        histogram = histogram[histogram > 0]

        # Calculate entropy
        entropy = -np.sum(histogram * np.log2(histogram))

        return float(entropy)


# Singleton instance
_video_scene_detector: Optional[VideoSceneDetector] = None
_client_lock = threading.Lock()


def get_video_scene_detector() -> VideoSceneDetector:
    """
    Get or create thread-safe VideoSceneDetector singleton instance

    Returns:
        VideoSceneDetector: Singleton client instance
    """
    global _video_scene_detector

    if _video_scene_detector is None:
        with _client_lock:
            if _video_scene_detector is None:
                _video_scene_detector = VideoSceneDetector()

    return _video_scene_detector
