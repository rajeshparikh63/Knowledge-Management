"""
Document Ingestion Service
Orchestrates the complete ingestion pipeline:
1. Upload files to iDrive E2
2. Extract raw content (PDF / video / YouTube / etc.)
3. Store raw content + app metadata in PostgreSQL (`documents` table)
4. Hand the content to Cognee, which internally chunks, embeds to pgvector,
   extracts entities/relationships, and writes the knowledge graph to FalkorDB
"""

import io
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile
from clients.idrivee2_client import get_idrivee2_client
from clients.postgres_client import get_postgres_client
# New two-layer KG engine (replaces graphrag_client). Same public surface
# (ingest_text / ingest_chunks / delete_document), so this is a drop-in swap.
# Rollback = restore `from clients.graphrag_client import get_graphrag_client`.
from clients.kg.pipeline import get_kg_pipeline
from clients.chunker_client import validate_content_for_embeddings
from utils.file_utils import (
    extract_raw_data,
    validate_extracted_content,
    sanitize_filename,
    get_file_size_mb,
    get_file_extension
)
from app.logger import logger
from app.settings import settings
import uuid


# ---------------------------------------------------------------------------
# Persistent worker event loop (background-thread variant)
#
# Celery (default prefork pool) runs tasks as sync function calls. We need
# to drive async code from inside those sync calls.
#
# Two problems we have to solve at once:
#   1. `asyncio.run` creates+destroys a fresh loop per call → module-level
#      asyncio.Lock objects get stranded → "bound to a different event loop".
#   2. A "persistent loop + run_until_complete" approach raises "This event
#      loop is already running" when something else (a Celery internal,
#      kombu, an async DB client) leaves the loop in a running state between
#      tasks.
#
# Robust fix: run ONE event loop forever in a dedicated background daemon
# thread. From the Celery task thread, submit coroutines to it via
# `run_coroutine_threadsafe`, which returns a concurrent.futures.Future we
# can `.result()` on to block until done.
#
# Why this works:
#   * The loop is always-running in its own thread → never "stopped between
#     calls" → no rebinding issues, no "already running" errors at our
#     submit site.
#   * Every module-level asyncio.Lock binds to this loop on first use and
#     stays valid for the worker's lifetime.
#   * Celery's main thread is untouched — whatever loops Celery/kombu run
#     in their own threads don't collide with ours.
# ---------------------------------------------------------------------------
import threading

_worker_loop_singleton: Optional[asyncio.AbstractEventLoop] = None
_worker_loop_thread: Optional[threading.Thread] = None
_worker_loop_lock = threading.Lock()
_worker_loop_ready = threading.Event()


def _ensure_worker_loop() -> asyncio.AbstractEventLoop:
    """Start the background loop+thread if not already running. Returns the loop."""
    global _worker_loop_singleton, _worker_loop_thread

    if (_worker_loop_singleton is not None
            and _worker_loop_thread is not None
            and _worker_loop_thread.is_alive()):
        return _worker_loop_singleton

    with _worker_loop_lock:
        # Double-checked after acquiring the lock
        if (_worker_loop_singleton is not None
                and _worker_loop_thread is not None
                and _worker_loop_thread.is_alive()):
            return _worker_loop_singleton

        _worker_loop_singleton = asyncio.new_event_loop()
        _worker_loop_ready.clear()

        def _runner(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            _worker_loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        _worker_loop_thread = threading.Thread(
            target=_runner,
            args=(_worker_loop_singleton,),
            name="ingestion-async-loop",
            daemon=True,
        )
        _worker_loop_thread.start()
        # Wait until the loop is set up in its thread before returning
        _worker_loop_ready.wait()
        logger.info("✅ Started persistent ingestion event loop (background thread)")
        return _worker_loop_singleton


def _run_in_worker_loop(coro):
    """Submit a coroutine to the worker's persistent loop and block until done.

    Use this from sync Celery task code instead of asyncio.run().
    """
    loop = _ensure_worker_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# Back-compat alias — code that imported _worker_loop() previously called
# .run_until_complete(...) on it. New code should call _run_in_worker_loop().
def _worker_loop():  # pragma: no cover - kept only to avoid import errors
    raise RuntimeError(
        "_worker_loop() is deprecated — use _run_in_worker_loop(coro) "
        "to submit work to the persistent loop instead."
    )


class IngestionService:
    """Service for handling document ingestion pipeline"""

    def __init__(self):
        """Initialize service with required clients."""
        from clients.unstructured_client import get_unstructured_client
        self.idrivee2_client = get_idrivee2_client()
        self.postgres_client = get_postgres_client()
        # Attribute name kept as `graphrag_client` for a minimal diff; it is now
        # the new two-layer KG pipeline (drop-in interface).
        self.graphrag_client = get_kg_pipeline()
        self.unstructured_client = get_unstructured_client()

    def cleanup(self):
        """Clean up all client resources and thread pools"""
        try:
            if hasattr(self, 'idrivee2_client') and hasattr(self.idrivee2_client, 'cleanup'):
                self.idrivee2_client.cleanup()
            if hasattr(self, 'unstructured_client') and hasattr(self.unstructured_client, 'cleanup'):
                self.unstructured_client.cleanup()
            logger.info("✅ Cleaned up IngestionService")
        except Exception as e:
            logger.warning(f"Error cleaning up IngestionService: {str(e)}")

    async def _process_single_document_async(
        self,
        document_id: str,
        file_key: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Async document processing - all database operations in single event loop

        Args:
            document_id: Document UUID
            file_key: iDrive E2 file path (already contains document_id)
            file_content: File bytes
            filename: Original filename (for display)
            content_type: MIME type
            folder_name: Folder name
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional metadata
        """
        logger.info(f"📄 Processing document async {document_id}: {filename}")

        file_size_mb = get_file_size_mb(file_content)

        try:
            # Step 1: Update status and set file_key
            logger.info(f"🚀 Starting processing: Upload to E2 + Content extraction for {filename}")
            await self.postgres_client.update_document(
                organization_id=organization_id,
                user_id=user_id,
                document_id=document_id,
                updates={
                    "file_key": file_key,  # Set the file_key now
                    "processing_stage": "uploading_extracting",
                    "processing_stage_description": "Uploading to storage and extracting content"
                }
            )

            # Step 2+3: Upload to E2 and extract content IN PARALLEL
            # These are independent — both only read file_content bytes
            async def _upload_to_e2():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.idrivee2_client.upload_file_sync(
                        file_obj=io.BytesIO(file_content),
                        object_name=file_key,
                        content_type=content_type
                    )
                )

            async def _extract_content():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: extract_raw_data(file_content, filename, folder_name, self.unstructured_client)
                )

            _, raw_content = await asyncio.gather(_upload_to_e2(), _extract_content())
            logger.info(f"✅ Upload and extraction complete (parallel) for {filename}")

            # Check if video
            is_video = isinstance(raw_content, dict) and raw_content.get('type') == 'video'

            if is_video:
                combined_text = raw_content['combined_text']
                video_chunks = raw_content['chunks']
                logger.info(f"🎬 Video extracted: {len(video_chunks)} scene chunks created")

                if not combined_text or not video_chunks:
                    raise ValueError(f"Video processing failed for: {filename}")

                content_for_mongodb = combined_text
            else:
                if not validate_extracted_content(raw_content):
                    raise ValueError(f"Extracted content is invalid or empty for: {filename}")

                validation_result = validate_content_for_embeddings(raw_content)
                logger.info(
                    f"📊 Content validation: {validation_result['token_count']} tokens, "
                    f"needs_chunking={validation_result['needs_chunking']}"
                )
                content_for_mongodb = raw_content

            # Step 4: Update PostgreSQL with content
            logger.info(f"💾 Updating document with extracted content: {filename}")

            # Sanitize content: Remove null bytes which PostgreSQL TEXT doesn't allow in UTF-8
            sanitized_content = content_for_mongodb.replace('\x00', '') if content_for_mongodb else ''

            await self.postgres_client.update_document(
                organization_id=organization_id,
                user_id=user_id,
                document_id=document_id,
                updates={
                    "raw_content": sanitized_content,
                    "processing_stage": "content_extracted",
                    "processing_stage_description": "Content extraction completed"
                }
            )

            # Step 5: Hand off to Cognee — it chunks, embeds (pgvector), and
            # builds the knowledge graph (FalkorDB) in one call.
            total_chunks = 0

            await self.postgres_client.update_document(
                organization_id=organization_id,
                user_id=user_id,
                document_id=document_id,
                updates={
                    "processing_stage": "cognifying",
                    "processing_stage_description": "Building knowledge graph"
                }
            )

            try:
                if is_video:
                    scene_texts = [c['blended_text'] for c in video_chunks if c.get('blended_text')]
                    await self.graphrag_client.ingest_chunks(
                        chunks=scene_texts,
                        organization_id=organization_id,
                        document_id=document_id,
                    )
                    total_chunks = len(scene_texts)
                else:
                    await self.graphrag_client.ingest_text(
                        text=sanitized_content,
                        organization_id=organization_id,
                        document_id=document_id,
                        filename=filename,
                    )
                    # Cognee does internal chunking; approximate for status display
                    total_chunks = max(1, len(sanitized_content) // 1000)

                logger.info(f"✅ GraphRAG ingest complete for {filename}")
            except Exception as graphrag_error:
                logger.error(f"❌ GraphRAG ingest failed (document will be marked failed): {graphrag_error}")
                raise

            # Mark as completed
            await self.postgres_client.update_document(
                organization_id=organization_id,
                user_id=user_id,
                document_id=document_id,
                updates={
                    "status": "completed",
                    "processing_stage": "completed",
                    "processing_stage_description": f"Successfully processed ~{total_chunks} chunks",
                }
            )
            logger.info(f"✅ Document marked as completed: {document_id}")

            result = {
                "document_id": document_id,
                "file_name": filename,
                "folder_name": folder_name,
                "file_key": file_key,
                "file_size_mb": file_size_mb,
                "total_chunks": total_chunks
            }

            if not is_video:
                result["token_count"] = validation_result['token_count']

            return result

        except Exception as e:
            logger.error(f"❌ Processing failed for {filename}: {str(e)}")
            await self.postgres_client.update_document(
                organization_id=organization_id,
                user_id=user_id,
                document_id=document_id,
                updates={
                    "status": "failed",
                    "processing_stage": "failed",
                    "processing_stage_description": f"Processing failed: {str(e)}",
                    "error": str(e)
                }
            )
            raise

    def process_single_document_sync(
        self,
        document_id: str,
        file_key: str,
        file_content: bytes,
        filename: str,
        content_type: str,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for Celery tasks - runs the async pipeline on a
        process-wide persistent event loop so module-level asyncio.Locks
        (e.g. _schema_locks, EntityVectorCache locks) stay bound to the same
        loop across multiple tasks handled by the same Celery worker.

        Args:
            document_id: Document UUID
            file_key: iDrive E2 file path
            file_content: File bytes
            filename: Original filename
            content_type: MIME type
            folder_name: Folder name
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional metadata
        """
        return _run_in_worker_loop(self._process_single_document_async(
            document_id=document_id,
            file_key=file_key,
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata=additional_metadata
        ))

    async def ingest_documents(
        self,
        files: List[UploadFile],
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents through the complete pipeline

        Args:
            files: List of uploaded files
            folder_name: Folder name for organization
            user_id: Optional user ID for tracking
            organization_id: Optional organization ID for tracking
            additional_metadata: Optional additional metadata to attach

        Returns:
            Dict with ingestion results and statistics
        """
        logger.info(f"🚀 Starting ingestion of {len(files)} files for folder: {folder_name}")

        results = {
            "folder_name": folder_name,
            "total_files": len(files),
            "successful_files": 0,
            "failed_files": 0,
            "documents": [],
            "errors": [],
            "statistics": {
                "total_chunks": 0,
                "total_size_mb": 0,
                "processing_time": None
            }
        }

        start_time = datetime.utcnow()

        # Process all files in parallel for faster uploads
        logger.info(f"🚀 Processing {len(files)} files in parallel")

        async def process_file_with_error_handling(file):
            """Wrapper to handle errors for each file independently"""
            try:
                result = await self._process_single_document(
                    file=file,
                    folder_name=folder_name,
                    user_id=user_id,
                    organization_id=organization_id,
                    additional_metadata=additional_metadata
                )
                logger.info(f"✅ Successfully processed: {file.filename}")
                return {"success": True, "result": result, "filename": file.filename}
            except Exception as e:
                logger.error(f"❌ Failed to process {file.filename}: {str(e)}")
                return {"success": False, "error": str(e), "filename": file.filename}

        # Process all files concurrently
        file_results = await asyncio.gather(*[process_file_with_error_handling(file) for file in files])

        # Aggregate results
        for file_result in file_results:
            if file_result["success"]:
                results["successful_files"] += 1
                results["documents"].append(file_result["result"])
                results["statistics"]["total_chunks"] += file_result["result"].get("total_chunks", 0)
                results["statistics"]["total_size_mb"] += file_result["result"].get("file_size_mb", 0)
            else:
                results["failed_files"] += 1
                results["errors"].append({
                    "file_name": file_result["filename"],
                    "error": file_result["error"]
                })

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        results["statistics"]["processing_time"] = processing_time

        logger.info(
            f"🎉 Ingestion completed: {results['successful_files']}/{results['total_files']} successful, "
            f"{results['statistics']['total_chunks']} chunks, "
            f"{processing_time:.2f}s"
        )

        return results

    async def _process_single_document(
        self,
        file: UploadFile,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline

        Args:
            file: Uploaded file
            folder_name: Folder name for organization
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional additional metadata

        Returns:
            Dict with processing results
        """
        logger.info(f"📄 Processing document: {file.filename}")

        # Read file content
        file_content = await file.read()
        file_size_mb = get_file_size_mb(file_content)

        # Prepare for parallel operations
        safe_filename = sanitize_filename(file.filename)
        # Include organization_id in file_key for multi-tenant isolation
        if organization_id:
            file_key = f"{organization_id}/{folder_name}/{safe_filename}"
        else:
            file_key = f"{folder_name}/{safe_filename}"  # Backwards compatibility

        # STEP 0: Create document record with status="processing" FIRST
        logger.info(f"📝 Creating document record with status='processing': {file.filename}")
        document_id = await self._create_document_with_status(
            file_name=file.filename,
            folder_name=folder_name,
            file_key=file_key,
            file_size_mb=file_size_mb,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata=additional_metadata
        )
        logger.info(f"✅ Document record created: {document_id}")

        try:
            # Step 1 & 2: Upload to iDrive E2 + Extract content (sequential, no threading)
            logger.info(f"🚀 Starting processing: Upload to E2 + Content extraction for {file.filename}")

            # Update status to indicate we're uploading + extracting
            await self._update_document_status(
                document_id=document_id,
                stage="uploading_extracting",
                stage_description="Uploading to storage and extracting content"
            )

            # Upload to E2 first
            await self.idrivee2_client.upload_file(
                file_obj=io.BytesIO(file_content),
                object_name=file_key,
                content_type=file.content_type
            )

            # Extract content (blocking call, no threading) - pass unstructured_client for proper cleanup
            raw_content = extract_raw_data(file_content, file.filename, folder_name, self.unstructured_client)

            logger.info(f"✅ Upload and extraction complete for {file.filename}")

            # Check if this is a video file (special handling)
            is_video = isinstance(raw_content, dict) and raw_content.get('type') == 'video'

            if is_video:
                # For videos: extract components
                combined_text = raw_content['combined_text']
                video_chunks = raw_content['chunks']
                logger.info(f"🎬 Video extracted: {len(video_chunks)} scene chunks created")

                # Validate video chunks exist
                if not combined_text or not video_chunks:
                    raise ValueError(f"Video processing failed for: {file.filename}")

                content_for_mongodb = combined_text
            else:
                # Normal files: validate and check token limits
                if not validate_extracted_content(raw_content):
                    raise ValueError(f"Extracted content is invalid or empty for: {file.filename}")

                # Validate content for embeddings (check token limits)
                validation_result = validate_content_for_embeddings(raw_content)
                logger.info(
                    f"📊 Content validation: {validation_result['token_count']} tokens, "
                    f"needs_chunking={validation_result['needs_chunking']}"
                )

                content_for_mongodb = raw_content

            # Step 3: Update MongoDB with extracted content
            logger.info(f"💾 Updating document with extracted content: {file.filename}")
            await self._update_document_content(
            document_id=document_id,
            raw_content=content_for_mongodb,
            stage="content_extracted",
            stage_description="Content extraction completed"
            )
            logger.info(f"✅ MongoDB content updated. Document ID: {document_id}")

            # Step 4 & 5: Hand off to Cognee — chunks + embeds + builds graph
            total_chunks = 0

            await self._update_document_status(
                document_id=document_id,
                stage="cognifying",
                stage_description="Building semantic memory (chunks + graph)"
            )

            # Forward GraphRAG sub-stages to the document row so the UI can
            # show a live pipeline (chunking → extracting N/M → resolving → writing).
            async def _graphrag_progress(
                stage: str,
                description: str,
                current: Optional[int] = None,
                total: Optional[int] = None,
            ) -> None:
                await self._update_document_status(
                    document_id=document_id,
                    stage=stage,
                    stage_description=description,
                    progress_current=current,
                    progress_total=total,
                )

            try:
                if is_video:
                    scene_texts = [c['blended_text'] for c in video_chunks if c.get('blended_text')]
                    await self.graphrag_client.ingest_chunks(
                        chunks=scene_texts,
                        organization_id=organization_id,
                        document_id=document_id,
                        progress_callback=_graphrag_progress,
                    )
                    total_chunks = len(scene_texts)
                else:
                    await self.graphrag_client.ingest_text(
                        text=raw_content,
                        organization_id=organization_id,
                        document_id=document_id,
                        filename=file.filename,
                        progress_callback=_graphrag_progress,
                    )
                    total_chunks = max(1, len(raw_content) // 1000)

                logger.info(f"✅ GraphRAG ingest complete for {file.filename}")
            except Exception as graphrag_error:
                logger.error(f"❌ GraphRAG ingest failed: {graphrag_error}")
                raise

            # Mark as completed
            await self._update_document_status(
                document_id=document_id,
                status="completed",
                stage="completed",
                stage_description=f"Successfully processed ~{total_chunks} chunks",
                completed_at=datetime.utcnow()
            )
            logger.info(f"✅ Document marked as completed: {document_id}")

            # Return results
            result = {
            "document_id": document_id,
            "file_name": file.filename,
            "folder_name": folder_name,
            "file_key": file_key,
            "file_size_mb": file_size_mb,
            "total_chunks": total_chunks
            }

            # Add token_count only for non-video files
            if not is_video:
                result["token_count"] = validation_result['token_count']

            return result

        except Exception as e:
            # Mark document as failed
            logger.error(f"❌ Processing failed for {file.filename}: {str(e)}")
            await self._update_document_status(
                document_id=document_id,
                status="failed",
                stage="failed",
                stage_description=f"Processing failed: {str(e)}",
                error=str(e),
                failed_at=datetime.utcnow()
            )
            raise  # Re-raise the exception

    async def _process_single_document_with_existing_id(
        self,
        document_id: str,
        file: UploadFile,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single document using an EXISTING document ID (document already created)

        This is used when documents are pre-created in the upload endpoint.

        Args:
            document_id: Existing document ID from MongoDB
            file: Uploaded file
            folder_name: Folder name for organization
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional additional metadata

        Returns:
            Dict with processing results
        """
        logger.info(f"📄 Processing document with existing ID {document_id}: {file.filename}")

        # Read file content
        file_content = await file.read()
        file_size_mb = get_file_size_mb(file_content)

        # Prepare for parallel operations
        safe_filename = sanitize_filename(file.filename)
        # Include organization_id in file_key for multi-tenant isolation
        if organization_id:
            file_key = f"{organization_id}/{folder_name}/{safe_filename}"
        else:
            file_key = f"{folder_name}/{safe_filename}"  # Backwards compatibility

        try:
            # Step 1 & 2: Upload to iDrive E2 + Extract content (sequential, no threading)
            logger.info(f"🚀 Starting processing: Upload to E2 + Content extraction for {file.filename}")

            # Update status to indicate we're uploading + extracting
            await self._update_document_status(
                document_id=document_id,
                stage="uploading_extracting",
                stage_description="Uploading to storage and extracting content"
            )

            # Upload to E2 first
            await self.idrivee2_client.upload_file(
                file_obj=io.BytesIO(file_content),
                object_name=file_key,
                content_type=file.content_type
            )

            # Extract content (blocking call, no threading) - pass unstructured_client for proper cleanup
            raw_content = extract_raw_data(file_content, file.filename, folder_name, self.unstructured_client)

            logger.info(f"✅ Upload and extraction complete for {file.filename}")

            # Check if this is a video file (special handling)
            is_video = isinstance(raw_content, dict) and raw_content.get('type') == 'video'

            if is_video:
                # For videos: extract components
                combined_text = raw_content['combined_text']
                video_chunks = raw_content['chunks']
                logger.info(f"🎬 Video extracted: {len(video_chunks)} scene chunks created")

                # Validate video chunks exist
                if not combined_text or not video_chunks:
                    raise ValueError(f"Video processing failed for: {file.filename}")

                content_for_mongodb = combined_text
            else:
                # Normal files: validate and check token limits
                if not validate_extracted_content(raw_content):
                    raise ValueError(f"Extracted content is invalid or empty for: {file.filename}")

                # Validate content for embeddings (check token limits)
                validation_result = validate_content_for_embeddings(raw_content)
                logger.info(
                    f"📊 Content validation: {validation_result['token_count']} tokens, "
                    f"needs_chunking={validation_result['needs_chunking']}"
                )

                content_for_mongodb = raw_content

            # Step 3: Update MongoDB with extracted content
            logger.info(f"💾 Updating document with extracted content: {file.filename}")
            await self._update_document_content(
                document_id=document_id,
                raw_content=content_for_mongodb,
                stage="content_extracted",
                stage_description="Content extraction completed"
            )
            logger.info(f"✅ MongoDB content updated. Document ID: {document_id}")

            # Step 4 & 5: Hand off to Cognee — chunks + embeds + builds graph
            total_chunks = 0

            await self._update_document_status(
                document_id=document_id,
                stage="cognifying",
                stage_description="Building semantic memory (chunks + graph)"
            )

            # Forward GraphRAG sub-stages to the document row so the UI can
            # show a live pipeline (chunking → extracting N/M → resolving → writing).
            async def _graphrag_progress(
                stage: str,
                description: str,
                current: Optional[int] = None,
                total: Optional[int] = None,
            ) -> None:
                await self._update_document_status(
                    document_id=document_id,
                    stage=stage,
                    stage_description=description,
                    progress_current=current,
                    progress_total=total,
                )

            try:
                if is_video:
                    scene_texts = [c['blended_text'] for c in video_chunks if c.get('blended_text')]
                    await self.graphrag_client.ingest_chunks(
                        chunks=scene_texts,
                        organization_id=organization_id,
                        document_id=document_id,
                        progress_callback=_graphrag_progress,
                    )
                    total_chunks = len(scene_texts)
                else:
                    await self.graphrag_client.ingest_text(
                        text=raw_content,
                        organization_id=organization_id,
                        document_id=document_id,
                        filename=file.filename,
                        progress_callback=_graphrag_progress,
                    )
                    total_chunks = max(1, len(raw_content) // 1000)

                logger.info(f"✅ GraphRAG ingest complete for {file.filename}")
            except Exception as graphrag_error:
                logger.error(f"❌ GraphRAG ingest failed: {graphrag_error}")
                raise

            # Mark as completed
            await self._update_document_status(
                document_id=document_id,
                status="completed",
                stage="completed",
                stage_description=f"Successfully processed ~{total_chunks} chunks",
                completed_at=datetime.utcnow()
            )
            logger.info(f"✅ Document marked as completed: {document_id}")

            # Return results
            result = {
                "document_id": document_id,
                "file_name": file.filename,
                "folder_name": folder_name,
                "file_key": file_key,
                "file_size_mb": file_size_mb,
                "total_chunks": total_chunks
            }

            # Add token_count only for non-video files
            if not is_video:
                result["token_count"] = validation_result['token_count']

            return result

        except Exception as e:
            # Mark document as failed
            logger.error(f"❌ Processing failed for {file.filename}: {str(e)}")
            await self._update_document_status(
                document_id=document_id,
                status="failed",
                stage="failed",
                stage_description=f"Processing failed: {str(e)}",
                error=str(e),
                failed_at=datetime.utcnow()
            )
            raise  # Re-raise the exception

    async def _create_document_with_status(
        self,
        file_name: str,
        folder_name: str,
        file_key: str,
        file_size_mb: float,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create document record with status="processing" in PostgreSQL

        Returns:
            Document ID (UUID string)
        """
        # Generate new UUID for document
        document_id = str(uuid.uuid4())

        # Prepare document data for PostgreSQL
        document_data = {
            "id": document_id,
            "file_name": file_name,
            "folder_name": folder_name,
            "raw_content": None,  # Will be updated later
            "file_key": file_key,
            "file_size_mb": file_size_mb,
            "file_extension": get_file_extension(file_name),
            "user_id": user_id,  # UUID string
            "organization_id": organization_id,  # UUID string
            "status": "processing",
            "processing_stage": "initializing",
            "processing_stage_description": "Starting ingestion",
            "processing_progress": None,
            "error": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": None,
            "failed_at": None
        }

        # Add additional metadata if provided. `id` is a top-level column;
        # everything else (source, drive_file_id, …) must go into the `metadata`
        # jsonb column — NOT spread at the top level, where insert_document
        # would silently drop it (which broke Drive dedup + the import banner).
        if additional_metadata:
            extra = dict(additional_metadata)
            if "id" in extra:
                document_data["id"] = extra.pop("id")
            if extra:
                document_data["metadata"] = extra

        await self.postgres_client.insert_document(document_data)

        return document_data["id"]

    async def _update_document_status(
        self,
        document_id: str,
        status: str = None,
        stage: str = None,
        stage_description: str = None,
        progress_current: int = None,
        progress_total: int = None,
        error: str = None,
        completed_at: datetime = None,
        failed_at: datetime = None,
        organization_id: str = None,
        user_id: str = None
    ):
        """
        Update document processing status in PostgreSQL

        Args:
            document_id: Document ID (UUID)
            status: Overall status (processing/completed/failed)
            stage: Current processing stage
            stage_description: Description of current stage
            progress_current: Current progress count
            progress_total: Total items to process
            error: Error message if failed
            completed_at: Completion timestamp
            failed_at: Failure timestamp
            organization_id: Organization ID for filtering (optional)
            user_id: User ID for filtering (optional)
        """
        update_fields = {
            "updated_at": datetime.utcnow()
        }

        if status is not None:
            update_fields["status"] = status

        if stage is not None:
            update_fields["processing_stage"] = stage

        if stage_description is not None:
            update_fields["processing_stage_description"] = stage_description

        if progress_current is not None or progress_total is not None:
            progress = {}
            if progress_current is not None:
                progress["current"] = progress_current
            if progress_total is not None:
                progress["total"] = progress_total
            if progress_current is not None and progress_total is not None and progress_total > 0:
                progress["percentage"] = round((progress_current / progress_total) * 100, 2)
            update_fields["processing_progress"] = progress

        if error is not None:
            update_fields["error"] = error

        if completed_at is not None:
            update_fields["completed_at"] = completed_at

        if failed_at is not None:
            update_fields["failed_at"] = failed_at

        await self.postgres_client.update_document(
            organization_id=organization_id,
            user_id=user_id,
            document_id=document_id,
            updates=update_fields
        )

    async def _update_document_content(
        self,
        document_id: str,
        raw_content: str,
        stage: str = None,
        stage_description: str = None,
        organization_id: str = None,
        user_id: str = None
    ):
        """
        Update document with extracted content and optionally update status in PostgreSQL

        Args:
            document_id: Document ID (UUID)
            raw_content: Extracted content
            stage: Current processing stage
            stage_description: Description of current stage
            organization_id: Organization ID for filtering (optional)
            user_id: User ID for filtering (optional)
        """
        update_fields = {
            "raw_content": raw_content,
            "updated_at": datetime.utcnow()
        }

        if stage is not None:
            update_fields["processing_stage"] = stage

        if stage_description is not None:
            update_fields["processing_stage_description"] = stage_description

        await self.postgres_client.update_document(
            organization_id=organization_id,
            user_id=user_id,
            document_id=document_id,
            updates=update_fields
        )

    async def _store_document_in_postgres(
        self,
        file_name: str,
        folder_name: str,
        raw_content: str,
        file_key: str,
        file_size_mb: float,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store document in PostgreSQL

        Args:
            file_name: Original file name
            folder_name: Folder name
            raw_content: Extracted raw content
            file_key: iDrive E2 object key/path (not URL, since bucket is private)
            file_size_mb: File size in MB
            user_id: Optional user ID (UUID string)
            organization_id: Optional organization ID (UUID string)
            additional_metadata: Optional additional metadata

        Returns:
            Document ID (UUID string)
        """
        # Generate new UUID for document
        document_id = str(uuid.uuid4())

        # Prepare document data for PostgreSQL
        document = {
            "id": document_id,
            "file_name": file_name,
            "folder_name": folder_name,
            "raw_content": raw_content,
            "file_key": file_key,
            "file_size_mb": file_size_mb,
            "file_extension": get_file_extension(file_name),
            "user_id": user_id,  # UUID string
            "organization_id": organization_id,  # UUID string
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Add additional metadata if provided
        if additional_metadata:
            document.update(additional_metadata)

        await self.postgres_client.insert_document(document)

        return document_id

    async def delete_document(
        self,
        document_id: str,
        organization_id: str,
        user_id: str = None,
        delete_from_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Delete document and its chunks from all systems (PostgreSQL, pgvector, Apache AGE, iDrive E2)

        Args:
            document_id: Document ID (UUID string)
            organization_id: Organization ID (UUID string)
            user_id: Optional user ID (UUID string) for filtering
            delete_from_storage: Whether to delete from iDrive E2

        Returns:
            Dict with deletion results
        """
        logger.info(f"🗑️ Deleting document: {document_id}")

        # Get document from PostgreSQL
        document = await self.postgres_client.find_document(
            organization_id=organization_id,
            user_id=user_id,
            document_id=document_id
        )

        if not document:
            raise ValueError(f"Document not found: {document_id}")

        # Use document's org_id and user_id if not provided
        doc_org_id = organization_id or document.get("organization_id")
        doc_user_id = user_id or document.get("user_id")

        # Delete from iDrive E2 if requested
        if delete_from_storage and document.get("file_key"):
            try:
                await self.idrivee2_client.delete_file(document["file_key"])
                logger.info(f"✅ Deleted from iDrive E2: {document['file_key']}")
            except Exception as e:
                logger.warning(f"Failed to delete from iDrive E2: {str(e)}")

        # Delete from GraphRAG (removes graph nodes tagged with this doc)
        try:
            await self.graphrag_client.delete_document(
                document_id=document_id,
                organization_id=doc_org_id,
            )
            logger.info(f"✅ Deleted GraphRAG data for document: {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete from GraphRAG: {str(e)}")

        # Delete from PostgreSQL
        await self.postgres_client.delete_document(
            organization_id=doc_org_id,
            user_id=doc_user_id,
            document_id=document_id
        )

        logger.info(f"✅ Document deleted: {document_id}")

        return {
            "document_id": document_id,
            "deleted": True
        }

    async def _convert_file_key_to_url(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert file_key to presigned URL in document (async)

        Args:
            document: Document dict with file_key

        Returns:
            Document dict with file_url instead of file_key
        """
        if document and document.get("file_key"):
            try:
                # Generate presigned URL (async, expiration from settings, default 7 days)
                file_url = await self.idrivee2_client.generate_presigned_url(
                    object_name=document["file_key"],
                    expiration=settings.PRESIGNED_URL_EXPIRATION
                )
                # Replace file_key with file_url
                document["file_url"] = file_url
                # Remove file_key from response
                document.pop("file_key", None)
            except Exception as e:
                logger.warning(f"Failed to generate presigned URL for {document.get('file_key')}: {str(e)}")
                # Keep file_key if URL generation fails
        return document

    async def get_document(
        self,
        document_id: str,
        organization_id: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Get document from PostgreSQL and convert file_key to presigned URL

        Args:
            document_id: Document ID (UUID string)
            organization_id: Organization ID (UUID string)
            user_id: Optional user ID (UUID string) for filtering

        Returns:
            Document dict with fresh presigned URL
        """
        # Get document from PostgreSQL
        document = await self.postgres_client.find_document(
            organization_id=organization_id,
            user_id=user_id,
            document_id=document_id
        )

        if document:
            # Convert file_key to presigned URL
            document = await self._convert_file_key_to_url(document)

        return document

    async def list_documents(
        self,
        folder_name: str = None,
        user_id: str = None,
        organization_id: str = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents from PostgreSQL with optional filters and convert file_keys to presigned URLs

        Args:
            folder_name: Optional folder name filter (string)
            user_id: Optional user ID filter (UUID string)
            organization_id: Optional organization ID filter (UUID string)
            limit: Maximum number of documents to return
            skip: Number of documents to skip

        Returns:
            List of documents with fresh presigned URLs
        """
        # Build filter query
        filters = {}
        if folder_name:
            filters["folder_name"] = folder_name
        if user_id:
            filters["user_id"] = user_id
        if organization_id:
            filters["organization_id"] = organization_id

        # Get documents from PostgreSQL
        documents = await self.postgres_client.find_documents(
            filters=filters,
            limit=limit,
            offset=skip
        )

        # NOTE: We deliberately do NOT mint presigned URLs here. This list is
        # re-fetched on every poll (every few seconds while anything is
        # processing) for potentially hundreds of documents — generating a URL
        # per doc per poll was wasteful and flooded the logs. The frontend asks
        # for a fresh URL on demand via GET /documents/{id} only when the user
        # actually clicks a filename. We keep `file_key` so the UI knows a
        # downloadable file exists.
        return documents

    async def list_folders(
        self,
        user_id: str = None,
        organization_id: str = None
    ) -> List[str]:
        """
        Get list of unique folder names from PostgreSQL documents table

        Args:
            user_id: Optional user ID filter (UUID string)
            organization_id: Optional organization ID filter (UUID string)

        Returns:
            List of unique folder names
        """
        # Build filter query
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if organization_id:
            filters["organization_id"] = organization_id

        # Get distinct folder names from PostgreSQL
        folders = await self.postgres_client.distinct_folders(
            filters=filters
        )

        # Filter out None/empty values and sort
        folders = [f for f in folders if f]
        folders.sort()

        logger.info(f"📁 Found {len(folders)} folders")
        return folders

    async def delete_folder(
        self,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        delete_from_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Delete all documents in a folder from ALL systems:
        - PostgreSQL (document metadata)
        - pgvector (vector chunks)
        - Apache AGE (graph entities/relationships)
        - iDrive E2 (file storage)

        Args:
            folder_name: Folder name to delete
            user_id: Optional user ID filter (UUID string)
            organization_id: Optional organization ID filter (UUID string)
            delete_from_storage: Whether to delete from iDrive E2

        Returns:
            Dict with deletion results
        """
        logger.info(f"🗑️ Deleting folder '{folder_name}' from ALL systems (PostgreSQL + Cognee + iDrive E2)")

        # Build filter query
        filters = {"folder_name": folder_name}
        if user_id:
            filters["user_id"] = user_id
        if organization_id:
            filters["organization_id"] = organization_id

        # Get all documents in folder from PostgreSQL
        documents = await self.postgres_client.find_documents(filters=filters)

        if not documents:
            logger.warning(f"No documents found in folder: {folder_name}")
            return {
                "folder_name": folder_name,
                "deleted_count": 0,
                "deleted_documents": []
            }

        deleted_count = 0
        deleted_docs = []
        errors = []

        # Delete each document (this handles PostgreSQL, Cognee, and iDrive E2)
        for doc in documents:
            try:
                # Convert UUID object to string if needed
                doc_id = str(doc["id"]) if doc["id"] else None
                await self.delete_document(
                    document_id=doc_id,
                    organization_id=organization_id,
                    user_id=user_id,
                    delete_from_storage=delete_from_storage
                )
                deleted_count += 1
                deleted_docs.append(doc_id)
            except Exception as e:
                logger.error(f"❌ Failed to delete document {doc.get('id')}: {str(e)}")
                errors.append({
                    "document_id": doc.get("id"),
                    "error": str(e)
                })

        logger.info(f"✅ Folder '{folder_name}' deleted: {deleted_count} documents removed from all systems")

        return {
            "folder_name": folder_name,
            "deleted_count": deleted_count,
            "deleted_documents": deleted_docs,
            "errors": errors if errors else None
        }

    async def rename_folder(
        self,
        old_folder_name: str,
        new_folder_name: str,
        user_id: str = None,
        organization_id: str = None
    ) -> Dict[str, Any]:
        """
        Rename a folder across systems:
        - PostgreSQL (document metadata) — authoritative folder_name
        - Cognee (no change needed — folder_name is not indexed inside Cognee)
        - iDrive E2 (no change needed — uses file_key, not folder_name)

        Args:
            old_folder_name: Current folder name
            new_folder_name: New folder name
            user_id: Optional user ID filter (UUID string)
            organization_id: Optional organization ID filter (UUID string)

        Returns:
            Dict with rename results
        """
        logger.info(f"📝 Renaming folder '{old_folder_name}' → '{new_folder_name}' across systems")

        # Build filter query
        filters = {"folder_name": old_folder_name}
        if user_id:
            filters["user_id"] = user_id
        if organization_id:
            filters["organization_id"] = organization_id

        # Get documents to verify
        documents = await self.postgres_client.find_documents(filters=filters)

        if not documents:
            logger.warning(f"No documents found in folder: {old_folder_name}")
            return {
                "old_folder_name": old_folder_name,
                "new_folder_name": new_folder_name,
                "updated_count": 0
            }

        # 1. Update PostgreSQL documents (authoritative folder_name)
        postgres_updated = await self.postgres_client.update_documents_bulk(
            filters=filters,
            updates={"folder_name": new_folder_name}
        )
        logger.info(f"  ✅ PostgreSQL: Updated {postgres_updated} documents")

        # 2. Cognee - No action needed (folder_name is not indexed inside Cognee)
        logger.info(f"  ✅ Cognee: No action needed (folder_name not in semantic memory)")

        # 3. iDrive E2 - No action needed (uses file_key, not folder_name)
        logger.info(f"  ✅ iDrive E2: No action needed (uses file_key)")

        logger.info(f"✅ Folder renamed successfully in PostgreSQL")

        return {
            "old_folder_name": old_folder_name,
            "new_folder_name": new_folder_name,
            "postgres_updated": postgres_updated,
            "graphrag_updated": "N/A (folder_name not in GraphRAG)",
            "idrive_updated": "N/A (uses file_key)"
        }


# Singleton instance
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    """
    Get or create IngestionService singleton instance

    Returns:
        IngestionService instance
    """
    global _ingestion_service

    if _ingestion_service is None:
        _ingestion_service = IngestionService()

    return _ingestion_service
