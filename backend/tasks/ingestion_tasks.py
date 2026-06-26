"""
Document Ingestion Celery Tasks
Processes documents using existing PostgreSQL document UUIDs
"""
import gc
import base64
from typing import Dict, Any, List, Optional
from app.worker import celery_app
from services.ingestion_service import IngestionService
from app.logger import logger


@celery_app.task
def process_document_ids_task(
    documents_data: List[Dict[str, Any]],
    folder_name: str,
    user_id: str = None,
    organization_id: str = None
) -> Dict[str, Any]:
    """
    Main Celery task - creates individual tasks for each document

    Args:
        documents_data: List of dicts with:
            - document_id: PostgreSQL document UUID (already created)
            - content_b64: Base64-encoded file content
            - filename: Original filename
            - content_type: MIME type
        folder_name: Folder name
        user_id: User ID (UUID)
        organization_id: Organization ID (UUID)

    Returns:
        Dict with task IDs
    """
    logger.info(f"📦 Main task: Distributing {len(documents_data)} documents to workers")

    task_info = []

    # Create individual Celery task for each document
    for doc_data in documents_data:
        try:
            # Launch individual worker task
            task = process_single_document_task.delay(
                document_id=doc_data["document_id"],
                file_key=doc_data["file_key"],
                content_b64=doc_data["content_b64"],
                filename=doc_data["filename"],
                content_type=doc_data["content_type"],
                folder_name=folder_name,
                user_id=user_id,
                organization_id=organization_id
            )

            task_info.append({
                "document_id": doc_data["document_id"],
                "filename": doc_data["filename"],
                "task_id": task.id,
                "status": "queued"
            })

            logger.info(f"✅ Queued task {task.id} for: {doc_data['filename']}")

        except Exception as e:
            logger.error(f"❌ Failed to queue {doc_data['filename']}: {str(e)}")
            task_info.append({
                "document_id": doc_data.get("document_id"),
                "filename": doc_data.get("filename"),
                "task_id": None,
                "status": "error",
                "error": str(e)
            })

    return {
        "status": "success",
        "total": len(documents_data),
        "tasks": task_info
    }


@celery_app.task(bind=True)
def process_single_document_task(
    self,
    document_id: str,
    file_key: str,
    content_b64: str,
    filename: str,
    content_type: str,
    folder_name: str,
    user_id: str = None,
    organization_id: str = None
) -> Dict[str, Any]:
    """
    Worker task - processes ONE document

    Args:
        self: Celery task instance
        document_id: PostgreSQL document UUID (already created with status="processing")
        file_key: iDrive E2 file path (organization_id/folder/document_id.ext)
        content_b64: Base64-encoded file content
        filename: Original filename
        content_type: MIME type
        folder_name: Folder name
        user_id: User ID (UUID)
        organization_id: Organization ID (UUID)

    Returns:
        Processing result
    """
    ingestion_service = None
    try:
        logger.info(f"🚀 Worker processing: {filename} (doc_id: {document_id})")

        # Decode base64 file content
        file_content = base64.b64decode(content_b64)

        # Create ingestion service
        ingestion_service = IngestionService()

        # Use fully synchronous method - no event loop needed
        result = ingestion_service.process_single_document_sync(
            document_id=document_id,
            file_key=file_key,
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata=None
        )

        logger.info(f"✅ Worker completed: {filename}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "result": result
        }

    except Exception as e:
        logger.error(f"❌ Worker failed {filename}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "document_id": document_id,
            "filename": filename,
            "error": str(e)
        }
    finally:
        # CRITICAL: Clean up all client resources and thread pools after EACH task
        if ingestion_service:
            try:
                ingestion_service.cleanup()
                logger.info(f"🧹 Cleaned up resources for: {filename}")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning for {filename}: {str(cleanup_error)}")

        # Clean up Unstructured client (singleton with httpx)
        try:
            from clients.unstructured_client import UnstructuredClient
            unstructured_client = UnstructuredClient()
            if hasattr(unstructured_client, 'cleanup'):
                unstructured_client.cleanup()
        except Exception as e:
            logger.warning(f"Unstructured cleanup warning: {str(e)}")

        # Force garbage collection to clean up any lingering thread pools
        gc.collect()
        logger.info(f"🗑️ Forced garbage collection after: {filename}")


@celery_app.task(bind=True)
def process_youtube_document_task(
    self,
    document_id: str,
    youtube_url: str,
    folder_name: str,
    user_id: str = None,
    organization_id: str = None
) -> Dict[str, Any]:
    """
    Worker task - downloads and processes YouTube video

    Args:
        self: Celery task instance
        document_id: PostgreSQL document UUID (already created with status="processing")
        youtube_url: YouTube video URL
        folder_name: Folder name
        user_id: User ID (UUID)
        organization_id: Organization ID (UUID)

    Returns:
        Processing result
    """
    from clients.youtube_downloader import YouTubeDownloader
    from clients.postgres_client import get_postgres_client
    from services.ingestion_service import _run_in_worker_loop
    from datetime import datetime

    ingestion_service = None
    temp_file_path = None

    try:
        logger.info(f"🚀 Worker processing YouTube: {youtube_url} (doc_id: {document_id})")

        # 1. Download video (returns bytes directly)
        downloader = YouTubeDownloader()
        logger.info(f"📥 Downloading YouTube video...")

        video_bytes, actual_filename, metadata = downloader.download_video(youtube_url)

        logger.info(f"✅ Downloaded: {actual_filename} ({len(video_bytes) / (1024*1024):.2f} MB)")

        file_size_mb = len(video_bytes) / (1024 * 1024)

        # 3. Build file_key using document_id and extension from downloaded filename
        from utils.file_utils import get_file_extension
        extension = get_file_extension(actual_filename)
        if organization_id:
            file_key = f"{organization_id}/{folder_name}/{document_id}{extension}"
        else:
            file_key = f"{folder_name}/{document_id}{extension}"

        # Update document with actual filename, file_key, and metadata
        postgres = get_postgres_client()

        # Submit async update to the worker's persistent background-thread
        # loop (NOT asyncio.run, which would create a fresh loop and orphan
        # any module-level asyncio.Locks).
        _run_in_worker_loop(postgres.update_document(
            organization_id=organization_id,
            user_id=user_id,
            document_id=document_id,
            updates={
                "file_name": actual_filename,
                "file_key": file_key,
                "file_size_mb": file_size_mb,
                "youtube_video_id": metadata.get("video_id"),
                "youtube_title": metadata.get("title"),
                "youtube_uploader": metadata.get("uploader"),
                "youtube_duration": metadata.get("duration"),
                "youtube_upload_date": metadata.get("upload_date"),
                "youtube_description": metadata.get("description"),
                "updated_at": datetime.utcnow()
            }
        ))

        logger.info(f"📝 Updated document with actual filename: {actual_filename}")

        # 4. Process the video using existing pipeline
        ingestion_service = IngestionService()

        result = ingestion_service.process_single_document_sync(
            document_id=document_id,
            file_key=file_key,
            file_content=video_bytes,
            filename=actual_filename,
            content_type="video/mp4",
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata=None  # Already updated above
        )

        logger.info(f"✅ Worker completed: {actual_filename}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": actual_filename,
            "result": result
        }

    except Exception as e:
        logger.error(f"❌ Worker failed for YouTube {youtube_url}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "document_id": document_id,
            "youtube_url": youtube_url,
            "error": str(e)
        }
    finally:
        # CRITICAL: Clean up all client resources
        if ingestion_service:
            try:
                ingestion_service.cleanup()
                logger.info(f"🧹 Cleaned up resources for YouTube video")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {str(cleanup_error)}")

        # Clean up Unstructured client
        try:
            from clients.unstructured_client import UnstructuredClient
            unstructured_client = UnstructuredClient()
            if hasattr(unstructured_client, 'cleanup'):
                unstructured_client.cleanup()
        except Exception as e:
            logger.warning(f"Unstructured cleanup warning: {str(e)}")

        # Force garbage collection
        gc.collect()
        logger.info(f"🗑️ Forced garbage collection after YouTube video")


@celery_app.task(bind=True)
def process_drive_file_task(
    self,
    document_id: str,
    drive_file_id: str,
    drive_mime_type: str,
    file_name: str,
    file_key: str,
    folder_name: str,
    user_id: str,
    organization_id: str,
) -> Dict[str, Any]:
    """
    Worker task — download ONE file from the user's connected Google Drive
    and run it through the same async ingestion pipeline as direct uploads.

    The document row was already created with status='processing' by the
    /api/google-drive/ingest endpoint, so we never block on creation here.

    Args:
        document_id: PostgreSQL UUID created by the router.
        drive_file_id: Google Drive file id from the Picker.
        drive_mime_type: MIME type Picker reported.
        file_name: Display name from Drive (used as the document file_name).
        file_key: Pre-built iDrive E2 object path.
        folder_name: KB folder the document belongs to.
        user_id, organization_id: From the request.
    """
    from services.ingestion_service import _run_in_worker_loop
    from clients.google_drive_client import GoogleDriveClient, GoogleDriveError

    ingestion_service = None
    try:
        logger.info(
            f"📥 Drive worker processing: {file_name} "
            f"(drive_id={drive_file_id[:12]}…, doc_id={document_id})"
        )

        async def _download_then_ingest() -> Dict[str, Any]:
            # 1. Get a DriveClient bound to this user's stored tokens
            client = await GoogleDriveClient.for_user(organization_id, user_id)
            if client is None:
                raise GoogleDriveError(
                    f"User {user_id} has no Google Drive connection; "
                    "the row was deleted between queue + execution"
                )

            # 2. Download the file's bytes (handles Google-native export)
            content, effective_mime = await client.download_file(
                drive_file_id, drive_mime_type
            )
            logger.info(
                f"✅ Drive download complete: {file_name} "
                f"({len(content) / 1024:.1f} KB, mime={effective_mime})"
            )

            # 3. Hand off to the same async pipeline direct uploads use.
            #    For Google-native exports, the file extension on file_key
            #    doesn't match the exported MIME — but the ingestion service
            #    only uses file_name's extension for routing, so we map the
            #    name to a sensible extension.
            export_ext = {
                "text/plain": ".txt",
                "text/csv":   ".csv",
            }
            effective_name = file_name
            if drive_mime_type.startswith("application/vnd.google-apps."):
                ext = export_ext.get(effective_mime, ".txt")
                # Append the export extension so MarkItDown / our text decoder
                # picks the right handler.
                if not effective_name.lower().endswith(ext):
                    effective_name = f"{effective_name}{ext}"

            nonlocal ingestion_service
            ingestion_service = IngestionService()
            return await ingestion_service._process_single_document_async(
                document_id=document_id,
                file_key=file_key,
                file_content=content,
                filename=effective_name,
                content_type=effective_mime,
                folder_name=folder_name,
                user_id=user_id,
                organization_id=organization_id,
                additional_metadata=None,  # router already wrote source metadata
            )

        result = _run_in_worker_loop(_download_then_ingest())

        logger.info(f"✅ Drive worker completed: {file_name}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": file_name,
            "drive_file_id": drive_file_id,
            "result": result,
        }

    except Exception as e:
        logger.error(
            f"❌ Drive worker failed for {file_name}: {e}", exc_info=True
        )
        return {
            "status": "error",
            "document_id": document_id,
            "filename": file_name,
            "drive_file_id": drive_file_id,
            "error": str(e),
        }
    finally:
        if ingestion_service:
            try:
                ingestion_service.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning for {file_name}: {cleanup_error}")
        gc.collect()


def _sanitize_kb_folder(name: str) -> str:
    """Make a Drive folder name safe to use as a KB folder / object-key segment."""
    cleaned = (name or "").strip().replace("/", "-")
    return cleaned or "Google Drive"


@celery_app.task(bind=True)
def discover_drive_files_task(
    self,
    organization_id: str,
    user_id: str,
    folder_name: str = "Google Drive",
    folder_ids: "Optional[List[str]]" = None,
    folders: "Optional[List[Dict[str, Any]]]" = None,
) -> Dict[str, Any]:
    """
    Worker task — enumerate the user's connected Drive, fan out per-file
    ingestion for every new (un-ingested) file.

    Scope (in priority order):
      - `folders` = [{"id", "name"}, …]: each folder's files (recursively)
        land in a KB folder NAMED AFTER that Drive folder. This is the normal
        picker path — "HR", "Services", etc. become separate sidebar folders.
      - `folder_ids` = [...]: legacy — all files into the single `folder_name`.
      - neither: the WHOLE drive into `folder_name` (legacy /sync).

    Dedup: any Drive file already represented in this user's documents
    (matched by metadata.drive_file_id) is skipped, so re-running is idempotent.
    """
    import uuid as _uuid

    from services.ingestion_service import _run_in_worker_loop
    from clients.google_drive_client import GoogleDriveClient, GoogleDriveError
    from clients.postgres_client import get_postgres_client
    from utils.file_utils import get_file_extension

    if folders:
        scope_desc = f"{len(folders)} folder(s), per-folder KB naming"
    elif folder_ids:
        scope_desc = f"{len(folder_ids)} folder(s) → '{folder_name}'"
    else:
        scope_desc = f"entire drive → '{folder_name}'"
    logger.info(f"🔍 Drive discovery starting: user={user_id[:8]}… scope={scope_desc}")

    async def _discover_and_queue() -> Dict[str, Any]:
        client = await GoogleDriveClient.for_user(organization_id, user_id)
        if client is None:
            raise GoogleDriveError(
                f"User {user_id} has no Google Drive connection — "
                "did the callback finish?"
            )

        pg = get_postgres_client()
        already_ingested = await pg.list_ingested_drive_file_ids(
            organization_id, user_id
        )
        logger.info(
            f"📋 Dedup: {len(already_ingested)} Drive files already ingested "
            f"for this user — they'll be skipped"
        )

        ingestion_service = IngestionService()
        counters = {"discovered": 0, "queued": 0, "skipped": 0}

        async def _ingest_file(f, kb_folder: str) -> None:
            """Create a doc row + queue ingestion for one Drive file."""
            counters["discovered"] += 1
            if f.id in already_ingested:
                counters["skipped"] += 1
                return
            already_ingested.add(f.id)  # avoid double-queue within this run

            document_id = str(_uuid.uuid4())
            ext = get_file_extension(f.name) or ""
            if f.mime_type.startswith("application/vnd.google-apps.document"):
                ext = ".txt"
            elif f.mime_type.startswith("application/vnd.google-apps.spreadsheet"):
                ext = ".csv"
            elif f.mime_type.startswith("application/vnd.google-apps.presentation"):
                ext = ".txt"
            file_key = f"{organization_id}/{kb_folder}/{document_id}{ext}"

            await ingestion_service._create_document_with_status(
                file_name=f.name,
                folder_name=kb_folder,
                file_key=file_key,
                file_size_mb=(f.size or 0) / (1024 * 1024),
                user_id=user_id,
                organization_id=organization_id,
                additional_metadata={
                    "id": document_id,
                    "source": "google_drive",
                    "drive_file_id": f.id,
                    "drive_mime_type": f.mime_type,
                    "drive_file_name": f.name,
                    "drive_web_url": f.web_url,
                },
            )
            process_drive_file_task.delay(
                document_id=document_id,
                drive_file_id=f.id,
                drive_mime_type=f.mime_type,
                file_name=f.name,
                file_key=file_key,
                folder_name=kb_folder,
                user_id=user_id,
                organization_id=organization_id,
            )
            counters["queued"] += 1
            if counters["discovered"] % 50 == 0:
                logger.info(
                    f"🔍 Discovery progress: {counters['discovered']} examined, "
                    f"{counters['queued']} queued, {counters['skipped']} skipped"
                )

        try:
            if folders:
                # Per-folder KB naming: each selected folder → its own KB folder
                for spec in folders:
                    kb = _sanitize_kb_folder(spec.get("name", ""))
                    fid = spec.get("id")
                    if not fid:
                        continue
                    logger.info(f"📂 Ingesting Drive folder '{kb}' (id={fid[:12]}…)")
                    async for f in client.list_files_in_folders([fid]):
                        await _ingest_file(f, kb)
            else:
                kb = _sanitize_kb_folder(folder_name)
                file_iter = (
                    client.list_files_in_folders(folder_ids)
                    if folder_ids
                    else client.list_files()
                )
                async for f in file_iter:
                    await _ingest_file(f, kb)
        finally:
            try:
                ingestion_service.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Discovery cleanup warning: {cleanup_error}")

        return dict(counters)

    try:
        result = _run_in_worker_loop(_discover_and_queue())
        logger.info(
            f"✅ Drive discovery complete for user={user_id[:8]}…: "
            f"{result['discovered']} discovered, {result['queued']} queued, "
            f"{result['skipped']} skipped (already ingested)"
        )
        return {"status": "success", **result}

    except Exception as e:
        logger.error(
            f"❌ Drive discovery failed for user={user_id[:8]}…: {e}",
            exc_info=True,
        )
        return {"status": "error", "error": str(e)}
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# Startup reconciliation — self-heal orphaned Drive docs on worker boot
# ---------------------------------------------------------------------------
from celery.signals import worker_ready  # noqa: E402


@worker_ready.connect
def _reconcile_stuck_drive_docs(**_kwargs):
    """When the worker boots, re-queue any Google Drive docs left stuck at
    'initializing' (their per-file task was orphaned by a previous worker
    restart). Makes restarts self-healing instead of leaving docs stuck on
    'processing' forever.
    """
    import json as _json

    from services.ingestion_service import _run_in_worker_loop
    from clients.postgres_client import get_postgres_client

    async def _run() -> int:
        pg = get_postgres_client()
        pool = await pg.get_pool()
        async with pool.acquire() as c:
            rows = await c.fetch(
                """
                SELECT id, filename, file_key, folder_name, user_id,
                       organization_id, metadata
                FROM documents
                WHERE status = 'processing'
                  AND processing_stage = 'initializing'
                  AND metadata->>'source' = 'google_drive'
                """
            )
        n = 0
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try:
                    md = _json.loads(md)
                except (ValueError, TypeError):
                    md = {}
            md = md or {}
            drive_file_id = md.get("drive_file_id")
            if not drive_file_id:
                continue  # can't re-queue without the Drive id
            process_drive_file_task.delay(
                document_id=str(r["id"]),
                drive_file_id=drive_file_id,
                drive_mime_type=md.get("drive_mime_type"),
                file_name=md.get("drive_file_name") or r["filename"],
                file_key=r["file_key"],
                folder_name=r["folder_name"],
                user_id=str(r["user_id"]),
                organization_id=str(r["organization_id"]),
            )
            n += 1
        return n

    try:
        count = _run_in_worker_loop(_run())
        if count:
            logger.info(f"♻️  Reconciled {count} stuck Drive doc(s) on worker boot — re-queued")
        else:
            logger.info("♻️  No stuck Drive docs to reconcile on boot")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Stuck-doc reconciliation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# SharePoint (via Composio) — discovery + per-file ingestion
# ---------------------------------------------------------------------------

@celery_app.task(bind=True)
def discover_sharepoint_files_task(
    self,
    organization_id: str,
    user_id: str,
    libraries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Walk each selected SharePoint document library (drive) and fan out
    per-file ingestion for every new (un-ingested) supported file. Each
    library's files land in a KB folder named after the library.

    Dedup: any SharePoint item already represented in this user's documents
    (matched by metadata.sharepoint_item_id) is skipped, so re-running is
    idempotent.
    """
    import asyncio as _asyncio
    import uuid as _uuid

    from services.ingestion_service import _run_in_worker_loop
    from clients.sharepoint_client import get_sharepoint_client
    from clients.postgres_client import get_postgres_client
    from utils.file_utils import get_file_extension

    logger.info(
        f"🔍 SharePoint discovery starting: user={str(user_id)[:8]}… "
        f"{len(libraries)} library(ies)"
    )

    async def _discover_and_queue() -> Dict[str, Any]:
        pg = get_postgres_client()
        already = await pg.list_ingested_sharepoint_item_ids(organization_id, user_id)
        logger.info(f"📋 Dedup: {len(already)} SharePoint items already ingested — skipping")

        ingestion_service = IngestionService()
        counters = {"discovered": 0, "queued": 0, "skipped": 0}
        sp = get_sharepoint_client(user_id)

        try:
            for lib in libraries:
                drive_id = lib.get("id")
                kb = _sanitize_kb_folder(lib.get("name", ""))
                if not drive_id:
                    continue
                logger.info(f"📂 Walking SharePoint library '{kb}' (drive={drive_id[:10]}…)")
                # walk_drive is sync (Composio SDK) + slow (many API calls) — run
                # off the event loop so concurrent file-ingests aren't blocked.
                files = await _asyncio.to_thread(sp.walk_drive, drive_id)

                for f in files:
                    counters["discovered"] += 1
                    item_id = f.get("id") or f.get("Id")
                    name = f.get("name") or f.get("Name") or "file"
                    web_url = f.get("webUrl")
                    if not item_id or item_id in already:
                        counters["skipped"] += 1
                        continue
                    already.add(item_id)

                    document_id = str(_uuid.uuid4())
                    ext = get_file_extension(name) or ""
                    file_key = f"{organization_id}/{kb}/{document_id}{ext}"

                    await ingestion_service._create_document_with_status(
                        file_name=name,
                        folder_name=kb,
                        file_key=file_key,
                        file_size_mb=(f.get("size") or 0) / (1024 * 1024),
                        user_id=user_id,
                        organization_id=organization_id,
                        additional_metadata={
                            "id": document_id,
                            "source": "sharepoint",
                            "sharepoint_item_id": item_id,
                            "sharepoint_web_url": web_url,
                            "sharepoint_file_name": name,
                            "sharepoint_drive_id": drive_id,
                        },
                    )
                    process_sharepoint_file_task.delay(
                        document_id=document_id,
                        web_url=web_url,
                        file_name=name,
                        file_key=file_key,
                        folder_name=kb,
                        user_id=str(user_id),
                        organization_id=str(organization_id),
                    )
                    counters["queued"] += 1
                    if counters["discovered"] % 50 == 0:
                        logger.info(
                            f"🔍 SP discovery progress: {counters['discovered']} examined, "
                            f"{counters['queued']} queued, {counters['skipped']} skipped"
                        )
        finally:
            try:
                ingestion_service.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"SharePoint discovery cleanup warning: {cleanup_error}")

        return dict(counters)

    try:
        result = _run_in_worker_loop(_discover_and_queue())
        logger.info(
            f"✅ SharePoint discovery complete for user={str(user_id)[:8]}…: "
            f"{result['discovered']} discovered, {result['queued']} queued, "
            f"{result['skipped']} skipped (already ingested)"
        )
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"❌ SharePoint discovery failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        gc.collect()


@celery_app.task(bind=True)
def process_sharepoint_file_task(
    self,
    document_id: str,
    web_url: str,
    file_name: str,
    file_key: str,
    folder_name: str,
    user_id: str,
    organization_id: str,
) -> Dict[str, Any]:
    """Download ONE SharePoint file via Composio, then run the same async
    ingestion pipeline as direct uploads + Drive."""
    from services.ingestion_service import _run_in_worker_loop
    from clients.sharepoint_client import get_sharepoint_client

    ingestion_service = None
    try:
        logger.info(f"📥 SharePoint worker processing: {file_name} (doc_id={document_id})")

        sp = get_sharepoint_client(user_id)
        content, effective_mime, effective_name = sp.download_file(web_url, file_name)
        logger.info(
            f"✅ SharePoint download complete: {file_name} "
            f"({len(content) / 1024:.1f} KB, mime={effective_mime})"
        )

        async def _ingest() -> Dict[str, Any]:
            nonlocal ingestion_service
            ingestion_service = IngestionService()
            return await ingestion_service._process_single_document_async(
                document_id=document_id,
                file_key=file_key,
                file_content=content,
                filename=effective_name,
                content_type=effective_mime,
                folder_name=folder_name,
                user_id=user_id,
                organization_id=organization_id,
                additional_metadata=None,  # router/discovery already wrote source metadata
            )

        result = _run_in_worker_loop(_ingest())
        logger.info(f"✅ SharePoint worker completed: {file_name}")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": file_name,
            "result": result,
        }
    except Exception as e:
        logger.error(f"❌ SharePoint worker failed for {file_name}: {e}", exc_info=True)
        return {
            "status": "error",
            "document_id": document_id,
            "filename": file_name,
            "error": str(e),
        }
    finally:
        if ingestion_service:
            try:
                ingestion_service.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning for {file_name}: {cleanup_error}")
        gc.collect()
