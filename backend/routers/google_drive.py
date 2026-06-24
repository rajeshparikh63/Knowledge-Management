"""
Google Drive connector — OAuth + file ingestion endpoints.

Endpoints
---------
GET  /api/google-drive/connect       Start OAuth: returns the Google consent URL.
GET  /api/google-drive/callback      OAuth callback: stores tokens, redirects to UI.
GET  /api/google-drive/status        Is this user connected? Returns {connected, email}.
POST /api/google-drive/ingest        Ingest the files the user picked via Google Picker.
DELETE /api/google-drive/disconnect  Wipe stored tokens.

The frontend opens the Google Picker, which returns a list of selected file
ids + mime types. The frontend POSTs that list to /ingest; we queue one
Celery task per file. Each task downloads from Drive (using the stored
refresh token to mint an access token if needed) then runs the existing
ingestion pipeline. Documents show up in the sidebar with the same
processing-stage indicator as direct uploads.
"""
from __future__ import annotations

import base64
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.logger import logger
from app.settings import settings
from auth.keycloak_auth import get_current_user_keycloak
from clients.google_drive_client import (
    GoogleDriveClient,
    GoogleDriveError,
    build_authorize_url,
    exchange_code_for_tokens,
    fetch_userinfo,
)
from clients.postgres_client import get_postgres_client
from services.ingestion_service import get_ingestion_service
from tasks.ingestion_tasks import discover_drive_files_task, process_drive_file_task
from utils.file_utils import get_file_extension


router = APIRouter(prefix="/google-drive", tags=["google-drive"])


# Single-use state tokens (CSRF protection for OAuth round-trip).
# In-memory is fine — they expire in 10 minutes and we have one worker.
# For multi-worker prod, move this to Redis.
_state_store: Dict[str, Dict[str, Any]] = {}
_STATE_TTL = timedelta(minutes=10)


def _new_state(organization_id: str, user_id: str, folder_name: str) -> str:
    """Generate a CSRF state token + remember which user it belongs to."""
    state = secrets.token_urlsafe(32)
    _state_store[state] = {
        "organization_id": organization_id,
        "user_id": user_id,
        "folder_name": folder_name,
        "created_at": datetime.now(timezone.utc),
    }
    # Prune expired states opportunistically
    now = datetime.now(timezone.utc)
    expired = [s for s, v in _state_store.items() if now - v["created_at"] > _STATE_TTL]
    for s in expired:
        _state_store.pop(s, None)
    return state


def _consume_state(state: str) -> Optional[Dict[str, Any]]:
    entry = _state_store.pop(state, None)
    if entry is None:
        return None
    if datetime.now(timezone.utc) - entry["created_at"] > _STATE_TTL:
        return None
    return entry


# ---------------------------------------------------------------------------
# 1. Start OAuth — return the URL the frontend opens in a popup
# ---------------------------------------------------------------------------

@router.get("/connect")
async def connect(
    folder_name: str = "Google Drive",
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, str]:
    """Return the Google consent URL the user should be redirected to.

    Args:
        folder_name: The folder under which ingested Drive files will land.
            Defaults to "Google Drive". The frontend can override (e.g. user
            wants to file them under "Research") by passing ?folder_name=...

    Returns:
        {"auth_url": "https://accounts.google.com/o/oauth2/..."}
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    if not user_id or not organization_id:
        raise HTTPException(status_code=400, detail="User missing id or organization_id")

    try:
        state = _new_state(organization_id, user_id, folder_name.strip() or "Google Drive")
        auth_url = build_authorize_url(state)
    except GoogleDriveError as e:
        raise HTTPException(status_code=503, detail=str(e))

    logger.info(f"🔗 Drive connect URL issued for user={user_id[:8]}…")
    return {"auth_url": auth_url}


# ---------------------------------------------------------------------------
# 2. OAuth callback — Google redirects here with ?code=&state=
# ---------------------------------------------------------------------------

@router.get("/callback")
async def callback(request: Request) -> HTMLResponse:
    """Handle the OAuth callback: exchange code for tokens, store, notify FE.

    Returns a tiny HTML page that postMessages the result to the opener window
    (the dashboard) and closes itself. If there's no opener (popup was blocked
    and this opened as a full page), it redirects back to the dashboard with
    the result in the query string. NOTE: we no longer auto-ingest the whole
    Drive here — the user picks which folders to ingest after connecting.
    """
    qp = request.query_params

    # User clicked "deny" or something else went wrong on Google's side
    if "error" in qp:
        return _oauth_result_response(
            success=False, message=qp.get("error_description", qp["error"])
        )

    code = qp.get("code")
    state = qp.get("state")
    if not code or not state:
        return _oauth_result_response(success=False, message="missing code or state")

    entry = _consume_state(state)
    if entry is None:
        return _oauth_result_response(
            success=False, message="invalid or expired state — please try connecting again"
        )

    organization_id = entry["organization_id"]
    user_id = entry["user_id"]

    try:
        tokens = await exchange_code_for_tokens(code)
    except GoogleDriveError as e:
        return _oauth_result_response(success=False, message=f"token exchange failed: {e}")

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = int(tokens.get("expires_in", 3600))
    if not access_token or not refresh_token:
        # No refresh_token typically means the user already consented before
        # without prompt=consent. Our build_authorize_url forces consent, so
        # this should be rare — but tell the user how to fix if it happens.
        return _oauth_result_response(
            success=False,
            message="No refresh token returned. Revoke this app in your Google account settings, then reconnect.",
        )

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    # Fetch identity for display ("connected as alice@gmail.com")
    try:
        me = await fetch_userinfo(access_token)
    except GoogleDriveError:
        me = {}

    pg = get_postgres_client()
    await pg.upsert_google_drive_connection(
        organization_id=organization_id,
        user_id=user_id,
        email=me.get("email"),
        display_name=me.get("name"),
        access_token=access_token,
        refresh_token=refresh_token,
        access_token_expires_at=expires_at,
    )
    # A fresh connect clears any prior "needs reconnect" flag.
    try:
        await pg.set_google_drive_needs_reconnect(organization_id, user_id, False)
    except Exception as e:  # pragma: no cover - defensive (pre-migration)
        logger.warning(f"Could not clear needs_reconnect on connect: {e}")

    logger.info(
        f"✅ Google Drive connected: user={user_id[:8]}… email={me.get('email')} "
        f"(no auto-ingest — user will pick folders)"
    )

    return _oauth_result_response(success=True, email=me.get("email") or "")


def _oauth_result_response(*, success: bool, email: str = "", message: str = "") -> HTMLResponse:
    """Return an HTML page that reports the OAuth outcome to the opener window.

    Primary path: postMessage to window.opener (the dashboard) then close.
    Fallback (no opener — popup blocked, opened full-page): redirect to the
    dashboard with the result encoded in the query string.
    """
    from urllib.parse import quote

    frontend = settings.FRONTEND_URL.rstrip("/")
    payload = {
        "type": "drive-oauth-result",
        "success": success,
        "email": email,
        "message": message,
    }

    parts = [f"drive_connected={'1' if success else '0'}"]
    if email:
        parts.append(f"drive_email={quote(email)}")
    if message:
        parts.append(f"drive_message={quote(message)}")
    redirect_url = f"{frontend}/dashboard?{'&'.join(parts)}"

    body_text = (
        "Connected. You can close this window."
        if success
        else "Connection failed. You can close this window."
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Google Drive</title></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0a0a0a;color:#e4e4e7;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
  <p style="font-size:14px;">{body_text}</p>
  <script>
    (function() {{
      var payload = {json.dumps(payload)};
      try {{
        if (window.opener && !window.opener.closed) {{
          window.opener.postMessage(payload, {json.dumps(frontend)});
          window.close();
          return;
        }}
      }} catch (e) {{}}
      // No opener (popup blocked / full-page) → redirect back to the app
      window.location.replace({json.dumps(redirect_url)});
    }})();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# 3. Status — is this user connected?
# ---------------------------------------------------------------------------

@router.get("/status")
async def status_endpoint(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    pg = get_postgres_client()
    row = await pg.get_google_drive_connection(organization_id, user_id)
    if row is None:
        return {
            "connected": False,
            "email": None,
            "display_name": None,
            "needs_reconnect": False,
        }
    return {
        "connected": True,
        "email": row.get("email"),
        "display_name": row.get("display_name"),
        "connected_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        # `or False` covers the brief pre-migration window before the column exists
        "needs_reconnect": bool(row.get("needs_reconnect") or False),
    }


# ---------------------------------------------------------------------------
# 4. Ingest — frontend hands us the file list Picker returned
# ---------------------------------------------------------------------------

class DrivePickedFile(BaseModel):
    """One file as returned by Google Picker."""
    id: str = Field(..., description="Google Drive file id")
    name: str = Field(..., description="File name as shown in Drive")
    mime_type: str = Field(..., description="MIME type Picker reported")
    size: Optional[int] = Field(default=None, description="Size in bytes if known")


class DriveIngestRequest(BaseModel):
    folder_name: str = Field(default="Google Drive", description="Folder to file ingested docs under")
    files: List[DrivePickedFile] = Field(..., min_length=1)


@router.post("/ingest")
async def ingest(
    body: DriveIngestRequest,
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Queue one Celery task per picked file. Returns the doc ids it created.

    Mirrors the regular /upload flow: we create document rows in PostgreSQL
    with status='processing' first, then fan out per-file Celery tasks. The
    UI's existing IngestionPipeline component picks up the stages as the
    documents move through the pipeline.
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    if not user_id or not organization_id:
        raise HTTPException(status_code=400, detail="User missing id or organization_id")

    # Ensure the user actually connected Drive at some point
    pg = get_postgres_client()
    connection = await pg.get_google_drive_connection(organization_id, user_id)
    if connection is None:
        raise HTTPException(
            status_code=400,
            detail="Google Drive is not connected. Hit /api/google-drive/connect first.",
        )

    folder_name = body.folder_name.strip() or "Google Drive"
    ingestion_service = get_ingestion_service()

    document_ids: List[str] = []
    for f in body.files:
        document_id = str(uuid.uuid4())
        extension = get_file_extension(f.name)
        file_key = f"{organization_id}/{folder_name}/{document_id}{extension}"

        await ingestion_service._create_document_with_status(
            file_name=f.name,
            folder_name=folder_name,
            file_key=file_key,
            file_size_mb=(f.size or 0) / (1024 * 1024) if f.size else 0.0,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata={
                "id": document_id,
                "source": "google_drive",
                "drive_file_id": f.id,
                "drive_mime_type": f.mime_type,
                "drive_file_name": f.name,
            },
        )

        process_drive_file_task.delay(
            document_id=document_id,
            drive_file_id=f.id,
            drive_mime_type=f.mime_type,
            file_name=f.name,
            file_key=file_key,
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id,
        )
        document_ids.append(document_id)

    logger.info(
        f"📥 Drive ingest queued: user={user_id[:8]}… {len(document_ids)} files → folder={folder_name}"
    )
    return {
        "success": True,
        "document_ids": document_ids,
        "folder_name": folder_name,
        "queued_count": len(document_ids),
    }


# ---------------------------------------------------------------------------
# 5. List files — paginated, backs our own custom file picker modal
# ---------------------------------------------------------------------------

@router.get("/files")
async def list_files_paged(
    page_token: Optional[str] = None,
    page_size: int = 50,
    search: Optional[str] = None,
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Paginated file list for the in-app file picker.

    Returns one page at a time so the picker stays responsive on big drives.
    `page_token` from a previous response → next page. `search` does a
    substring name match server-side via Drive's `name contains '...'`.
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    client = await GoogleDriveClient.for_user(organization_id, user_id)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="Google Drive is not connected. Hit /connect first.",
        )
    try:
        return await client.list_files_page(
            page_token=page_token,
            page_size=page_size,
            search=search,
        )
    except GoogleDriveError as e:
        raise HTTPException(status_code=502, detail=f"Drive API error: {e}")


# ---------------------------------------------------------------------------
# 5b. List folders + ingest-by-folder — the post-connect folder picker
# ---------------------------------------------------------------------------

@router.get("/folders")
async def list_folders(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Return every folder in the user's Drive (with computed paths).

    Backs the folder-picker UI shown right after connecting, so the user
    chooses which folders to ingest instead of slurping the whole drive.
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    client = await GoogleDriveClient.for_user(organization_id, user_id)
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="Google Drive is not connected. Hit /connect first.",
        )
    try:
        folders = await client.list_folders()
    except GoogleDriveError as e:
        raise HTTPException(status_code=502, detail=f"Drive API error: {e}")
    return {"folders": folders}


class IngestFolderSpec(BaseModel):
    id: str = Field(..., description="Drive folder (or shared-drive) id")
    name: str = Field(..., description="Folder name — used as the KB folder")


class IngestFoldersRequest(BaseModel):
    folders: List[IngestFolderSpec] = Field(..., min_length=1)


@router.post("/ingest-folders")
async def ingest_folders(
    body: IngestFoldersRequest,
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Ingest every supported file under the selected Drive folders.

    Each selected folder's files (recursively, including subfolders) land in a
    KB folder named after that Drive folder — so "HR" / "Services" show up as
    separate folders in the sidebar instead of one big "Google Drive" bucket.
    Files already ingested are skipped via dedup.
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    pg = get_postgres_client()
    if await pg.get_google_drive_connection(organization_id, user_id) is None:
        raise HTTPException(
            status_code=400,
            detail="Google Drive is not connected. Hit /connect first.",
        )
    folders = [{"id": f.id, "name": f.name} for f in body.folders]
    discover_drive_files_task.delay(
        organization_id=organization_id,
        user_id=user_id,
        folders=folders,
    )
    logger.info(
        f"📂 Drive folder ingest queued: user={user_id[:8]}… "
        f"{len(folders)} folder(s)"
    )
    return {
        "success": True,
        "message": f"Ingesting {len(folders)} folder(s)",
        "folder_count": len(folders),
    }


# ---------------------------------------------------------------------------
# 6. Sync — manually re-discover (e.g. user added new files to Drive)
# ---------------------------------------------------------------------------

@router.post("/sync")
async def sync(
    folder_name: str = "Google Drive",
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Re-run discovery for an already-connected user.

    Idempotent: files that were already ingested are skipped by
    metadata.drive_file_id dedup; only genuinely new files get queued.
    """
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    pg = get_postgres_client()
    if await pg.get_google_drive_connection(organization_id, user_id) is None:
        raise HTTPException(
            status_code=400,
            detail="Google Drive is not connected. Hit /connect first.",
        )
    discover_drive_files_task.delay(
        organization_id=organization_id,
        user_id=user_id,
        folder_name=folder_name.strip() or "Google Drive",
    )
    return {"success": True, "message": "Discovery queued"}


# ---------------------------------------------------------------------------
# 7. Disconnect — wipe stored tokens
# ---------------------------------------------------------------------------

@router.delete("/disconnect")
async def disconnect(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    pg = get_postgres_client()
    removed = await pg.delete_google_drive_connection(organization_id, user_id)
    return {"success": True, "removed": removed}
