"""
SharePoint connector — via Composio (no token management on our side).

Endpoints
---------
POST   /api/sharepoint/connect           Start OAuth: body {subdomain} -> {auth_url}.
GET    /api/sharepoint/status            Is this user connected? {connected, status}.
GET    /api/sharepoint/libraries         Document libraries (drives) the user can ingest.
POST   /api/sharepoint/ingest-libraries  Ingest all supported files in the picked libraries.
DELETE /api/sharepoint/disconnect        Remove the Composio connection(s).

Composio owns the Microsoft OAuth flow + token refresh. We identify each user to
Composio by their user_id. The share_point auth config needs a per-user
`subdomain` (tenant name), collected by the connect UI. After the user visits
the returned auth_url and authorizes, Composio stores the connection; the
frontend polls /status until it flips to connected.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.logger import logger
from auth.keycloak_auth import get_current_user_keycloak
from clients.sharepoint_client import SharePointClient, SharePointError, get_sharepoint_client
from tasks.ingestion_tasks import discover_sharepoint_files_task


router = APIRouter(prefix="/sharepoint", tags=["sharepoint"])


def _client(current_user: dict) -> SharePointClient:
    user_id = current_user.get("id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User missing id")
    try:
        return get_sharepoint_client(user_id)
    except SharePointError as e:
        # COMPOSIO_API_KEY not configured, etc.
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# 1. Connect — returns the Composio-hosted OAuth URL
# ---------------------------------------------------------------------------

class ConnectRequest(BaseModel):
    # Where Composio redirects the browser after the user finishes consent.
    # The frontend passes `${origin}/oauth-callback`. The tenant subdomain is
    # collected by Composio's own hosted connect page — not by us.
    callback_url: Optional[str] = Field(
        default=None,
        description="URL Composio redirects to after the hosted OAuth flow completes.",
    )


@router.post("/connect")
async def connect(
    body: ConnectRequest,
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    client = _client(current_user)
    try:
        result = await asyncio.to_thread(client.initiate_connection, body.callback_url)
    except SharePointError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - Composio surface
        logger.error(f"SharePoint connect failed: {e}")
        raise HTTPException(status_code=502, detail=f"Composio error: {e}")

    if not result.get("auth_url"):
        raise HTTPException(status_code=502, detail="Composio did not return an auth URL")
    return {
        "success": True,
        "auth_url": result["auth_url"],
        "connection_id": result.get("connection_id"),
    }


# ---------------------------------------------------------------------------
# 2. Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def status_endpoint(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    client = _client(current_user)
    try:
        return await asyncio.to_thread(client.connection_status)
    except Exception as e:
        logger.warning(f"SharePoint status check failed: {e}")
        return {"connected": False, "status": None, "connection_id": None}


# ---------------------------------------------------------------------------
# 3. Libraries — the pickable units (document libraries / drives)
# ---------------------------------------------------------------------------

@router.get("/libraries")
async def list_libraries(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    client = _client(current_user)
    status = await asyncio.to_thread(client.connection_status)
    if not status.get("connected"):
        raise HTTPException(status_code=400, detail="SharePoint is not connected. Hit /connect first.")
    try:
        libraries = await asyncio.to_thread(client.list_libraries)
    except SharePointError as e:
        raise HTTPException(status_code=502, detail=f"SharePoint API error: {e}")
    return {"libraries": libraries}


# ---------------------------------------------------------------------------
# 4. Ingest selected libraries
# ---------------------------------------------------------------------------

class IngestLibrarySpec(BaseModel):
    id: str = Field(..., description="Drive (document library) id")
    name: str = Field(..., description="Library name — used as the KB folder")


class IngestLibrariesRequest(BaseModel):
    libraries: List[IngestLibrarySpec] = Field(..., min_length=1)


@router.post("/ingest-libraries")
async def ingest_libraries(
    body: IngestLibrariesRequest,
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    """Queue discovery for each selected library. Each library's files
    (recursively) land in a KB folder named after the library. Files already
    ingested are skipped via metadata.sharepoint_item_id dedup."""
    user_id = current_user.get("id")
    organization_id = current_user.get("organization_id")
    if not user_id or not organization_id:
        raise HTTPException(status_code=400, detail="User missing id or organization_id")

    client = _client(current_user)
    status = await asyncio.to_thread(client.connection_status)
    if not status.get("connected"):
        raise HTTPException(status_code=400, detail="SharePoint is not connected. Hit /connect first.")

    libraries = [{"id": lib.id, "name": lib.name} for lib in body.libraries]
    discover_sharepoint_files_task.delay(
        organization_id=organization_id,
        user_id=user_id,
        libraries=libraries,
    )
    logger.info(
        f"📂 SharePoint ingest queued: user={str(user_id)[:8]}… "
        f"{len(libraries)} library(ies)"
    )
    return {
        "success": True,
        "message": f"Ingesting {len(libraries)} library(ies)",
        "library_count": len(libraries),
    }


# ---------------------------------------------------------------------------
# 5. Disconnect
# ---------------------------------------------------------------------------

@router.delete("/disconnect")
async def disconnect(
    current_user: dict = Depends(get_current_user_keycloak),
) -> Dict[str, Any]:
    client = _client(current_user)
    try:
        removed = await asyncio.to_thread(client.disconnect)
    except Exception as e:
        logger.warning(f"SharePoint disconnect error: {e}")
        removed = 0
    return {"success": True, "removed": removed}
