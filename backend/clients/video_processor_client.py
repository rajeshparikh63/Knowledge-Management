"""
Video Processor Client
Orchestrates complete video processing pipeline
Thread-safe singleton implementation
"""

import tempfile
import threading
import io
import cv2
from typing import Dict, List, Optional
from pathlib import Path
from moviepy import VideoFileClip
from clients.groq_whisper_client import get_groq_whisper_client
from clients.video_frame_extractor import get_video_frame_extractor
from clients.video_scene_detector import get_video_scene_detector
from clients.video_aligner import get_video_aligner
from clients.idrivee2_client import get_idrivee2_client
from app.logger import logger


class VideoProcessorClient:
    """Thread-safe video processor orchestrating full pipeline"""

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
        """Initialize video processor"""
        if not hasattr(self, '_initialized'):
            self._processing_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Video processor initialized")

    def process_video(
        self,
        file_content: bytes,
        filename: str,
        folder_name: str = "videos"
    ) -> Dict:
        """
        Process video through complete streaming pipeline

        MEMORY-EFFICIENT: Uses streaming for unlimited video length!
        Peak memory: ~50MB regardless of video duration

        Pipeline stages:
        1. Audio extraction → Whisper transcription (timestamped)
        2. Scene detection → SSIM streaming (pass 1, ~10MB memory)
        3. Key frame selection → Entropy streaming (pass 2, ~10MB memory)
        4. Color frame extraction → Key frames only (seeking)
        5. Upload thumbnails → iDrive E2 (file_keys)
        6. VLM description → Multimodal alignment & blending
        7. Format output → Combined text + scene chunks

        Args:
            file_content: Video file content as bytes
            filename: Original filename
            folder_name: Folder for organizing uploads (default: "videos")

        Returns:
            Dictionary with:
            - combined_text: str (all scenes for MongoDB)
            - chunks: List[Dict] (scene chunks for Pinecone)
                Each chunk has:
                - chunk_id, video_id, video_name
                - clip_start, clip_end, duration
                - key_frame_timestamp, keyframe_file_key
                - transcript_text, visual_description, blended_text

        Raises:
            Exception: If processing fails
        """
        with self._processing_lock:
            try:
                logger.info(f"🎬 Starting video processing pipeline for: {filename}")

                # Generate video ID (sanitized filename)
                video_id = self._sanitize_video_id(filename)
                video_name = filename

                # Stage 1: Extract audio and transcribe (timestamped)
                logger.info("📝 Stage 1/7: Audio transcription (timestamped)")
                transcript_segments = self._extract_and_transcribe_audio(
                    file_content,
                    filename
                )

                # Stage 2: Detect scenes using streaming (pass 1)
                logger.info("🎬 Stage 2/7: Scene detection (streaming - pass 1)")
                frame_extractor = get_video_frame_extractor()
                scene_detector = get_video_scene_detector()

                # Stream frames for scene detection (memory: ~10MB)
                frame_generator_1 = frame_extractor.extract_frames_streaming(file_content, filename)
                scenes = scene_detector.detect_scenes_streaming(frame_generator_1)

                if not scenes:
                    raise Exception("No scenes detected in video")

                logger.info(f"✅ Detected {len(scenes)} scenes (memory-efficient streaming)")

                # Stage 3: Select key frames using streaming (pass 2)
                logger.info("🔑 Stage 3/7: Key frame selection (streaming - pass 2)")

                # Stream frames again for entropy calculation (memory: ~10MB)
                frame_generator_2 = frame_extractor.extract_frames_streaming(file_content, filename)
                key_frames_data = scene_detector.select_key_frames_streaming(frame_generator_2, scenes)

                logger.info(f"✅ Selected {len(key_frames_data)} key frames (memory-efficient streaming)")

                # Stage 4: Extract color frames for key frames only
                logger.info("🖼️ Stage 4/7: Color frame extraction (key frames only)")
                color_frames = []
                for kf in key_frames_data:
                    color_frame = frame_extractor.extract_color_frame(
                        file_content,
                        filename,
                        kf['frame_number']
                    )
                    color_frames.append(color_frame)

                logger.info(f"✅ Extracted {len(color_frames)} color key frames")

                # Stage 5: Upload key frame thumbnails to iDrive E2
                logger.info("📤 Stage 5/7: Uploading key frame thumbnails")
                keyframe_file_keys = self._upload_keyframe_thumbnails(
                    color_frames,
                    key_frames_data,
                    video_id,
                    folder_name
                )

                # Stage 6: Align and blend (includes VLM description)
                logger.info("🔗 Stage 6/7: Alignment and multimodal blending")
                aligner = get_video_aligner()
                aligned_chunks = aligner.align_and_blend(
                    transcript_segments,
                    scenes,
                    key_frames_data,
                    color_frames
                )

                # Clean up color frames and scenes from memory
                del color_frames
                del scenes
                logger.debug("🧹 Cleaned up color frames and scenes from memory")

                # Stage 7: Format final output
                logger.info("📦 Stage 7/7: Formatting final output")
                final_chunks = []
                combined_text_parts = []

                for i, chunk in enumerate(aligned_chunks):
                    chunk_id = f"{video_id}_chunk_{chunk['scene_id']}"

                    final_chunks.append({
                        'chunk_id': chunk_id,
                        'video_id': video_id,
                        'video_name': video_name,
                        'clip_start': chunk['clip_start'],
                        'clip_end': chunk['clip_end'],
                        'duration': chunk['clip_end'] - chunk['clip_start'],
                        'key_frame_timestamp': chunk['key_frame_timestamp'],
                        'keyframe_file_key': keyframe_file_keys[i],  # iDrive E2 file key for thumbnail
                        'transcript_text': chunk['transcript_text'],
                        'visual_description': chunk['visual_description'],
                        'blended_text': chunk['blended_text']  # This gets embedded
                    })

                    # Add to combined text with scene markers
                    combined_text_parts.append(
                        f"[Scene {chunk['scene_id']} - {chunk['clip_start']:.1f}s to {chunk['clip_end']:.1f}s]\n"
                        f"{chunk['blended_text']}\n"
                    )

                # Combine all scene texts
                combined_text = "\n".join(combined_text_parts)

                logger.info(
                    f"✅ Video processing complete: {filename} → "
                    f"{len(final_chunks)} chunks created, "
                    f"{len(combined_text)} chars combined text"
                )

                return {
                    'combined_text': combined_text,  # For MongoDB
                    'chunks': final_chunks  # For Pinecone (skip semantic chunking)
                }

            except Exception as e:
                logger.error(f"❌ Video processing failed for {filename}: {str(e)}")
                raise Exception(f"Video processing failed: {str(e)}")

    def _extract_and_transcribe_audio(
        self,
        file_content: bytes,
        filename: str
    ) -> List[Dict]:
        """
        Extract audio from video and transcribe using Groq Whisper

        Args:
            file_content: Video file content as bytes
            filename: Original filename

        Returns:
            List of transcript segments with 'start', 'end', 'text'

        Raises:
            Exception: If extraction or transcription fails
        """
        audio_path = None
        video_path = None

        try:
            extension = Path(filename).suffix.lower()

            # Write video to temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as video_file:
                video_file.write(file_content)
                video_path = video_file.name

            # Extract audio using MoviePy
            logger.info("🎵 Extracting audio from video...")
            video_clip = VideoFileClip(video_path)

            # Check if video has audio track
            if video_clip.audio is None:
                video_clip.close()
                logger.warning("⚠️ Video has no audio track")
                logger.info("📝 Continuing with visual-only processing (no audio transcript)")
                return []

            # Create temp audio file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.mp3'
            ) as audio_file:
                audio_path = audio_file.name

            # Extract audio
            video_clip.audio.write_audiofile(
                audio_path,
                logger=None,  # Suppress MoviePy logs
                verbose=False
            )
            video_clip.close()

            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_content = f.read()

            # Transcribe with Groq Whisper (with timestamps!)
            logger.info("🎤 Transcribing audio with Groq Whisper...")
            whisper_client = get_groq_whisper_client()

            # Get timestamped segments
            transcript_segments = whisper_client.transcribe_audio_with_timestamps(
                audio_content,
                f"{Path(filename).stem}.mp3"
            )

            total_chars = sum(len(seg['text']) for seg in transcript_segments)
            logger.info(f"✅ Transcription complete: {len(transcript_segments)} segments, {total_chars} chars")

            return transcript_segments

        except Exception as e:
            logger.warning(f"⚠️ Audio extraction/transcription failed: {str(e)}")
            logger.info("📝 Continuing with visual-only processing (no audio transcript)")
            # Return empty segments as fallback (visual-only mode)
            return []

        finally:
            # Clean up temp files
            import os
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                    logger.debug(f"🧹 Cleaned up temp audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp audio file: {str(e)}")

            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    logger.debug(f"🧹 Cleaned up temp video file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp video file: {str(e)}")

    def _upload_keyframe_thumbnails(
        self,
        color_frames: List,
        key_frames_data: List[Dict],
        video_id: str,
        folder_name: str
    ) -> List[str]:
        """
        Upload key frame thumbnails to iDrive E2

        Args:
            color_frames: List of color frames (numpy arrays)
            key_frames_data: List of key frame metadata
            video_id: Sanitized video ID
            folder_name: Folder for organizing uploads

        Returns:
            List of file_keys for uploaded thumbnails
        """
        import cv2
        import asyncio

        idrive_client = get_idrivee2_client()
        file_keys = []

        async def upload_all_frames():
            """Helper to upload all frames asynchronously"""
            upload_tasks = []

            for i, (frame, kf_data) in enumerate(zip(color_frames, key_frames_data)):
                # Encode frame as JPEG
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                if not success:
                    logger.warning(f"Failed to encode keyframe {i}, skipping")
                    file_keys.append(None)
                    continue

                # Create file key
                file_key = f"{folder_name}/{video_id}/keyframe_{kf_data['scene_id']}.jpg"

                # Create upload task
                upload_tasks.append(idrive_client.upload_file(
                    file_obj=io.BytesIO(buffer.tobytes()),
                    object_name=file_key,
                    content_type='image/jpeg'
                ))
                file_keys.append(file_key)

            # Upload all frames in parallel
            try:
                await asyncio.gather(*upload_tasks)
                logger.info(f"✅ Uploaded {len(file_keys)} keyframe thumbnails")
            except Exception as e:
                logger.error(f"❌ Some keyframe uploads failed: {str(e)}")

            return file_keys

        try:
            # Run async uploads in new event loop (safe since we're in a thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(upload_all_frames())
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"❌ Failed to upload keyframes: {str(e)}")
            # Return None for all frames as fallback
            return [None] * len(color_frames)

    def _sanitize_video_id(self, filename: str) -> str:
        """
        Sanitize filename to create valid video ID

        Removes special characters, keeps alphanumeric and underscores

        Args:
            filename: Original filename

        Returns:
            Sanitized video ID
        """
        import re

        # Remove extension
        name = Path(filename).stem

        # Keep only alphanumeric and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        return sanitized or 'video'


# Singleton instance
_video_processor_client: Optional[VideoProcessorClient] = None
_client_lock = threading.Lock()


def get_video_processor_client() -> VideoProcessorClient:
    """
    Get or create thread-safe VideoProcessorClient singleton instance

    Returns:
        VideoProcessorClient: Singleton client instance
    """
    global _video_processor_client

    if _video_processor_client is None:
        with _client_lock:
            if _video_processor_client is None:
                _video_processor_client = VideoProcessorClient()

    return _video_processor_client
