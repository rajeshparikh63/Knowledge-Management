"""
Video Aligner Client
Aligns transcript segments with visual scenes and blends multimodal content
Thread-safe singleton implementation
"""

import base64
import cv2
import threading
import asyncio
from typing import List, Dict, Optional
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from clients.ultimate_llm import get_llm
from app.logger import logger


class VideoAligner:
    """Thread-safe video aligner for multimodal content blending"""

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
        """Initialize video aligner"""
        if not hasattr(self, '_initialized'):
            self._alignment_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Video aligner initialized")

    def align_and_blend(
        self,
        transcript_segments: List[Dict],
        scenes: List[Dict],
        key_frames_data: List[Dict],
        color_frames: List[np.ndarray]
    ) -> List[Dict]:
        """
        Align transcript with scenes and blend with visual descriptions

        Args:
            transcript_segments: List of dicts with 'start', 'end', 'text'
            scenes: List of scene dicts from scene detector
            key_frames_data: List of key frame metadata
            color_frames: List of color frames (BGR) corresponding to key_frames_data

        Returns:
            List of aligned chunks with:
            - scene_id: int
            - clip_start: float
            - clip_end: float
            - transcript_text: str
            - visual_description: str
            - blended_text: str
            - key_frame_timestamp: float

        Raises:
            Exception: If alignment/blending fails
        """
        with self._alignment_lock:
            try:
                logger.info(
                    f"🔗 Aligning {len(transcript_segments)} transcript segments "
                    f"with {len(scenes)} scenes (parallel with max 5 concurrent)"
                )

                # Process all scenes in parallel with semaphore limit of 5
                aligned_chunks = self._process_scenes_parallel(
                    scenes,
                    transcript_segments,
                    key_frames_data,
                    color_frames
                )

                logger.info(f"✅ Created {len(aligned_chunks)} aligned chunks")

                return aligned_chunks

            except Exception as e:
                logger.error(f"❌ Alignment and blending failed: {str(e)}")
                raise Exception(f"Alignment and blending failed: {str(e)}")

    def _process_scenes_parallel(
        self,
        scenes: List[Dict],
        transcript_segments: List[Dict],
        key_frames_data: List[Dict],
        color_frames: List
    ) -> List[Dict]:
        """
        Process all scenes in parallel with semaphore limit using pure asyncio

        Uses pure asyncio with ainvoke for optimal I/O-bound API calls.
        More efficient than ThreadPoolExecutor - no thread overhead.

        Args:
            scenes: List of scene dicts
            transcript_segments: List of transcript segments
            key_frames_data: List of key frame metadata
            color_frames: List of color frames

        Returns:
            List of aligned chunks (same order as scenes)
        """
        async def process_single_scene(
            semaphore: asyncio.Semaphore,
            scene: Dict,
            i: int
        ) -> Dict:
            """Process one scene with semaphore limit"""
            async with semaphore:  # Max 5 concurrent
                # Find overlapping transcript segments
                scene_start = scene['start_time']
                scene_end = scene['end_time']

                overlapping_segments = self._find_overlapping_segments(
                    transcript_segments,
                    scene_start,
                    scene_end
                )

                # Combine transcript text
                transcript_text = " ".join([seg['text'] for seg in overlapping_segments])

                # Get visual description from VLM (pure async)
                key_frame_data = key_frames_data[i]
                color_frame = color_frames[i]

                visual_description = await self._describe_frame_with_vlm_async(
                    color_frame,
                    key_frame_data['timestamp']
                )

                # Blend transcript + visual (pure async)
                blended_text = await self._blend_multimodal_content_async(
                    transcript_text,
                    visual_description
                )

                logger.debug(
                    f"✓ Scene {scene['scene_id']}: "
                    f"{len(overlapping_segments)} segments, "
                    f"{len(transcript_text)} chars transcript"
                )

                return {
                    'scene_id': scene['scene_id'],
                    'clip_start': scene_start,
                    'clip_end': scene_end,
                    'transcript_text': transcript_text,
                    'visual_description': visual_description,
                    'blended_text': blended_text,
                    'key_frame_timestamp': key_frame_data['timestamp']
                }

        async def process_all_scenes():
            """Process all scenes concurrently with semaphore"""
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent API calls

            tasks = [
                process_single_scene(semaphore, scene, i)
                for i, scene in enumerate(scenes)
            ]

            results = await asyncio.gather(*tasks)
            return results

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            aligned_chunks = loop.run_until_complete(process_all_scenes())
            return aligned_chunks
        finally:
            loop.close()

    def _find_overlapping_segments(
        self,
        segments: List[Dict],
        scene_start: float,
        scene_end: float
    ) -> List[Dict]:
        """
        Find transcript segments that overlap with scene time range

        Overlap logic: !(seg_end < scene_start OR seg_start > scene_end)

        Args:
            segments: List of transcript segments
            scene_start: Scene start time in seconds
            scene_end: Scene end time in seconds

        Returns:
            List of overlapping segments
        """
        overlapping = []

        for seg in segments:
            seg_start = seg['start']
            seg_end = seg['end']

            # Check for overlap
            if not (seg_end < scene_start or seg_start > scene_end):
                overlapping.append(seg)

        return overlapping

    async def _describe_frame_with_vlm_async(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> str:
        """
        Describe frame using VLM (extracts text + visual content) - ASYNC VERSION

        Uses same prompt as image_analysis_client.py for consistency.
        Uses ainvoke for true async I/O without thread overhead.

        Args:
            frame: Color frame as BGR numpy array
            timestamp: Frame timestamp in seconds

        Returns:
            Visual description including extracted text

        Raises:
            Exception: If VLM call fails
        """
        try:
            # Encode frame to base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Get VLM (using OpenAI vision)
            llm = get_llm(model="gpt-5-nano", provider="openai")

            # Create prompt (same as image_analysis_client.py)
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Extract ALL information from this image with precise attention to structure and layout. "
                    "DO NOT include any opening statements, explanations, or closing remarks. "
                    "START AND END with the extracted content only. "
                    "\n\n"
                    "For text-based images:\n"
                    "- Maintain paragraph breaks, bullet points, and formatting\n"
                    "- For tables: preserve row/column structure using markdown table format\n"
                    "- For charts/diagrams: describe visual elements, explain relationships between components, identify trends, and extract all data points and labels\n"
                    "- For formulas/equations: reconstruct them accurately\n"
                    "- Always maintain the original spatial layout and reading order\n"
                    "- Identify headers, footers, page numbers, and other document elements\n"
                    "\n"
                    "For images with NO TEXT or minimal text:\n"
                    "- Provide an extremely detailed visual description of EVERYTHING you can see\n"
                    "- Describe objects, people, scenes, colors, composition, spatial relationships\n"
                    "- Include details about lighting, textures, backgrounds, foregrounds\n"
                    "- Describe any actions, emotions, or interactions visible\n"
                    "- Be thorough and comprehensive - leave nothing out\n"
                    "\n"
                    "This is a critical data extraction task - ensure ALL content (text or visual) is captured comprehensively."
                ),
                (
                    "user",
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                )
            ])

            # Execute chain with ainvoke for true async
            chain = prompt | llm
            response = await chain.ainvoke({})

            visual_description = response.content

            logger.debug(
                f"VLM described frame at {timestamp:.2f}s: "
                f"{len(visual_description)} chars"
            )

            return visual_description

        except Exception as e:
            logger.error(f"❌ VLM description failed for frame at {timestamp:.2f}s: {str(e)}")
            # Return fallback
            return f"[Visual description unavailable for frame at {timestamp:.2f}s]"

    async def _blend_multimodal_content_async(
        self,
        transcript_text: str,
        visual_description: str
    ) -> str:
        """
        Blend transcript and visual description using LLM - ASYNC VERSION

        Uses ainvoke for true async I/O without thread overhead.

        Args:
            transcript_text: Spoken words from transcript
            visual_description: Visual content from VLM

        Returns:
            Blended multimodal description

        Raises:
            Exception: If blending fails
        """
        try:
            # Handle edge cases
            if not transcript_text and not visual_description:
                return "[No content available for this scene]"

            if not transcript_text:
                return visual_description

            if not visual_description:
                return transcript_text

            # Use LLM to intelligently blend both modalities
            llm = get_llm(model="gpt-5-nano", provider="openai")

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are combining transcript (spoken words) with visual description (what was shown on screen) "
                    "from a video segment. Create a single coherent description that:\n"
                    "- Merges both modalities naturally\n"
                    "- Removes redundancy (don't repeat same info from both sources)\n"
                    "- Keeps all important details from both sources\n"
                    "- Maintains context and clarity\n"
                    "- Outputs ONLY the blended content, no preamble or explanation\n\n"
                    "Format the output as a natural paragraph or structured text, depending on content."
                ),
                (
                    "user",
                    "TRANSCRIPT: {transcript}\n\nVISUAL CONTENT: {visual}\n\n"
                    "Blend these into a single coherent description:"
                )
            ])

            chain = prompt | llm
            response = await chain.ainvoke({
                "transcript": transcript_text,
                "visual": visual_description
            })

            blended = response.content

            logger.debug(
                f"Blended content: {len(transcript_text)} + {len(visual_description)} "
                f"→ {len(blended)} chars"
            )

            return blended

        except Exception as e:
            logger.error(f"❌ Blending failed: {str(e)}")
            # Fallback to concatenation
            return f"{transcript_text}\n\n{visual_description}"


# Singleton instance
_video_aligner: Optional[VideoAligner] = None
_client_lock = threading.Lock()


def get_video_aligner() -> VideoAligner:
    """
    Get or create thread-safe VideoAligner singleton instance

    Returns:
        VideoAligner: Singleton client instance
    """
    global _video_aligner

    if _video_aligner is None:
        with _client_lock:
            if _video_aligner is None:
                _video_aligner = VideoAligner()

    return _video_aligner
