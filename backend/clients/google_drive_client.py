"""
Google Drive client — delegated OAuth (user-scoped, not service account).

Design notes
------------
This is a deliberately *thin* Drive wrapper. Other Drive integrations I've
seen in the wild conflate three orthogonal concerns:

  1. OAuth — exchange code for tokens, refresh expired tokens.
  2. Discovery — walk folders, build paths, paginate file lists.
  3. Download — fetch a single file's bytes (handling Google-native export).

For this codebase we use **Google Picker** as the discovery UX (the JS-side
picker hands us file ids directly), so we don't need (2) at all. This client
covers (1) and (3) only.

Responsibilities
----------------
- Refresh an access token when it's near expiry.
- Persist the rotated token back to PostgreSQL (so the next Celery task gets
  the fresh one without re-doing the round-trip).
- Download a file's bytes; export Google-native types (Docs/Sheets/Slides)
  as text/csv automatically so the rest of the ingestion pipeline doesn't
  need to know they're not real files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from app.settings import settings
from clients.postgres_client import get_postgres_client

logger = logging.getLogger(__name__)

# Google endpoints
TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
DRIVE_API = "https://www.googleapis.com/drive/v3"

# Refresh proactively before actual expiry so we don't race a 401.
_REFRESH_LEEWAY = timedelta(seconds=60)

# Google-native MIME types → what we export them as. The keys are what
# Drive reports; the values are what we send to MarkItDown / the parser.
_EXPORT_MIME: Dict[str, str] = {
    "application/vnd.google-apps.document":     "text/plain",
    "application/vnd.google-apps.spreadsheet":  "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Only enumerate these during auto-discovery. Everything else is either
# unparseable for our pipeline, irrelevant (images, video unless we add
# a path for them), or would spend LLM tokens on noise (Drawings, Sites,
# Apps Scripts, Jupyter notebooks, etc.).
_INCLUDE_MIME = {
    # Google-native (handled via export to text/csv)
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/rtf",
    # Plain-ish text
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
}


@dataclass
class DriveFile:
    """One discovered file's metadata — kept lean since we only need this much."""
    id: str
    name: str
    mime_type: str
    size: int
    modified_at: Optional[datetime]
    web_url: Optional[str]


class GoogleDriveError(RuntimeError):
    """Anything from Drive that's not a 2xx response."""


@dataclass
class _Tokens:
    access_token: str
    refresh_token: str
    expires_at: datetime  # always tz-aware (UTC)


def build_authorize_url(state: str) -> str:
    """Build the URL we redirect the user to for the consent screen."""
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise GoogleDriveError(
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured"
        )
    from urllib.parse import urlencode

    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "scope": settings.GOOGLE_SCOPES,
        # access_type=offline → we get a refresh_token. Without this the user
        # would have to re-consent every hour when the access_token expires.
        "access_type": "offline",
        # prompt=consent → forces the consent screen even if the user has
        # already approved this app. Required to RE-receive a refresh_token,
        # because Google only returns one on the first consent (or on
        # explicit re-consent).
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """POST /token with grant_type=authorization_code. Returns Google's JSON."""
    return await _token_request({
        "grant_type": "authorization_code",
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
    })


async def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """POST /token with grant_type=refresh_token.

    Note: Google does NOT return a new refresh_token on refresh (the response
    has access_token, expires_in, scope, token_type — no refresh_token). Keep
    using the original one.
    """
    return await _token_request({
        "grant_type": "refresh_token",
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
    })


async def fetch_userinfo(access_token: str) -> Dict[str, Any]:
    """Fetch the connected user's email + name — for display in our UI."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if r.status_code >= 400:
        raise GoogleDriveError(f"userinfo {r.status_code}: {r.text[:200]}")
    return r.json()


async def _token_request(form: Dict[str, str]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(TOKEN_URL, data=form)
    if r.status_code >= 400:
        raise GoogleDriveError(f"token endpoint {r.status_code}: {r.text[:300]}")
    return r.json()


class GoogleDriveClient:
    """Per-(org, user) Drive client. Refreshes + persists tokens automatically."""

    def __init__(
        self,
        organization_id: str,
        user_id: str,
        tokens: _Tokens,
    ):
        self.organization_id = organization_id
        self.user_id = user_id
        self._tokens = tokens

    # ---- construction ------------------------------------------------------

    @classmethod
    async def for_user(
        cls, organization_id: str, user_id: str
    ) -> Optional["GoogleDriveClient"]:
        """Load the persisted connection for (org, user). Returns None if not connected."""
        pg = get_postgres_client()
        row = await pg.get_google_drive_connection(organization_id, user_id)
        if row is None:
            return None
        expires = row["access_token_expires_at"]
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return cls(
            organization_id=organization_id,
            user_id=user_id,
            tokens=_Tokens(
                access_token=row["access_token"],
                refresh_token=row["refresh_token"],
                expires_at=expires,
            ),
        )

    # ---- token lifecycle ---------------------------------------------------

    async def _ensure_valid_token(self) -> str:
        """Refresh if near expiry, persist, return a usable access token."""
        now = datetime.now(timezone.utc)
        if now + _REFRESH_LEEWAY < self._tokens.expires_at:
            return self._tokens.access_token

        logger.info(f"🔄 Refreshing Google Drive token for user={self.user_id[:8]}…")
        try:
            data = await refresh_access_token(self._tokens.refresh_token)
        except GoogleDriveError as e:
            # invalid_grant = the refresh token is dead: the user revoked our
            # app in their Google account, or it expired (6-month inactivity,
            # or password change for some account types). Mark the connection
            # so the UI can prompt a reconnect instead of silently failing.
            if "invalid_grant" in str(e).lower():
                logger.warning(
                    f"⚠️ Google Drive refresh token invalid for "
                    f"user={self.user_id[:8]}… — flagging needs_reconnect"
                )
                try:
                    pg = get_postgres_client()
                    await pg.set_google_drive_needs_reconnect(
                        self.organization_id, self.user_id, True
                    )
                except Exception as flag_err:  # pragma: no cover - defensive
                    logger.warning(f"Could not set needs_reconnect: {flag_err}")
                raise GoogleDriveError(
                    "Google Drive connection expired — please reconnect"
                ) from e
            raise
        new_access = data["access_token"]
        new_expires = now + timedelta(seconds=int(data.get("expires_in", 3600)))
        # Google may rotate refresh_token in some cases (per docs); persist if so.
        new_refresh = data.get("refresh_token")  # usually None

        self._tokens.access_token = new_access
        self._tokens.expires_at = new_expires
        if new_refresh:
            self._tokens.refresh_token = new_refresh

        # Write back to the DB so the next Celery task starts hot
        pg = get_postgres_client()
        await pg.update_google_drive_tokens(
            organization_id=self.organization_id,
            user_id=self.user_id,
            access_token=new_access,
            access_token_expires_at=new_expires,
            refresh_token=new_refresh,  # None == "don't touch"
        )
        return new_access

    # ---- Drive API ---------------------------------------------------------

    async def _get_bytes(self, url: str, params: Optional[Dict[str, str]] = None) -> bytes:
        token = await self._ensure_valid_token()
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
        if r.status_code >= 400:
            raise GoogleDriveError(
                f"Drive GET {url} → {r.status_code}: {r.text[:300]}"
            )
        return r.content

    async def _get_json(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        token = await self._ensure_valid_token()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
        if r.status_code >= 400:
            raise GoogleDriveError(
                f"Drive GET {url} → {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    async def list_files(self) -> AsyncIterator[DriveFile]:
        """Yield every supported file in the connected Drive.

        Filters server-side by MIME type so we don't enumerate (and pay for
        pagination on) irrelevant content like Drawings, Sites, etc. Walks
        all pages until exhausted.

        Notes:
          - `trashed = false` skips the user's Drive trash.
          - `supportsAllDrives=true` + `includeItemsFromAllDrives=true` make
            shared drives visible to the same query — important for orgs.
        """
        token = await self._ensure_valid_token()
        # Build the server-side MIME filter so Drive does the work
        mime_clause = " or ".join(f"mimeType = '{m}'" for m in _INCLUDE_MIME)
        query = f"({mime_clause}) and trashed = false"

        params: Dict[str, str] = {
            "q": query,
            "pageSize": "1000",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        page_token: Optional[str] = None
        seen = 0
        while True:
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)
            # Refresh token per page in case we cross the access-token expiry
            # mid-discovery (large drives take minutes).
            token = await self._ensure_valid_token()

            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.get(
                    f"{DRIVE_API}/files",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
            if r.status_code >= 400:
                raise GoogleDriveError(
                    f"Drive list_files → {r.status_code}: {r.text[:300]}"
                )
            data = r.json()

            for f in data.get("files", []):
                mime = f.get("mimeType", "")
                if mime not in _INCLUDE_MIME:
                    # Defensive: server filter occasionally leaks
                    continue
                modified: Optional[datetime] = None
                if (m := f.get("modifiedTime")):
                    modified = datetime.fromisoformat(m.replace("Z", "+00:00"))
                seen += 1
                yield DriveFile(
                    id=f["id"],
                    name=f.get("name", "(unnamed)"),
                    mime_type=mime,
                    size=int(f.get("size") or 0),
                    modified_at=modified,
                    web_url=f.get("webViewLink"),
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                logger.info(f"Drive list_files: yielded {seen} files total")
                return

    async def list_files_page(
        self,
        page_token: Optional[str] = None,
        page_size: int = 50,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return ONE page of files for our custom picker UI.

        Returns {"files": [...], "next_page_token": str | None}.

        Unlike list_files() (which yields everything for discovery), this
        gives the frontend lazy pagination so the picker stays responsive
        on drives with thousands of files.

        Args:
            page_token: Opaque token from a previous call's next_page_token.
            page_size: Files per page (Drive caps at 1000).
            search: Optional name filter — case-insensitive substring match
                via Drive's `name contains '...'` clause.
        """
        token = await self._ensure_valid_token()

        mime_clause = " or ".join(f"mimeType = '{m}'" for m in _INCLUDE_MIME)
        query_parts = [f"({mime_clause})", "trashed = false"]
        if search and search.strip():
            # Escape single quotes in Drive query string
            safe = search.strip().replace("'", "\\'")
            query_parts.append(f"name contains '{safe}'")
        query = " and ".join(query_parts)

        params: Dict[str, str] = {
            "q": query,
            "pageSize": str(min(max(page_size, 1), 1000)),
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
            # corpora=allDrives spans My Drive + every shared drive the user
            # can access (requires the two supportsAllDrives flags below).
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "orderBy": "modifiedTime desc",
        }
        if page_token:
            params["pageToken"] = page_token

        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.get(
                f"{DRIVE_API}/files",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
        if r.status_code >= 400:
            raise GoogleDriveError(
                f"Drive list_files_page → {r.status_code}: {r.text[:300]}"
            )
        data = r.json()

        files_out: List[Dict[str, Any]] = []
        for f in data.get("files", []):
            mime = f.get("mimeType", "")
            if mime not in _INCLUDE_MIME:
                continue
            files_out.append({
                "id": f["id"],
                "name": f.get("name", "(unnamed)"),
                "mime_type": mime,
                "size": int(f.get("size") or 0),
                "modified_time": f.get("modifiedTime"),
                "web_view_link": f.get("webViewLink"),
            })

        return {
            "files": files_out,
            "next_page_token": data.get("nextPageToken"),
        }

    async def _list_shared_drives(self) -> Dict[str, str]:
        """Return {shared_drive_id: name} for every shared drive the user can
        access. Best-effort: returns {} if the user has none or the call fails
        (e.g. the account has no shared-drive access)."""
        token = await self._ensure_valid_token()
        out: Dict[str, str] = {}
        params: Dict[str, str] = {
            "pageSize": "100",
            "fields": "nextPageToken,drives(id,name)",
        }
        page_token: Optional[str] = None
        try:
            while True:
                if page_token:
                    params["pageToken"] = page_token
                else:
                    params.pop("pageToken", None)
                async with httpx.AsyncClient(timeout=30.0) as http:
                    r = await http.get(
                        f"{DRIVE_API}/drives",
                        headers={"Authorization": f"Bearer {token}"},
                        params=params,
                    )
                if r.status_code >= 400:
                    logger.info(
                        f"Drive _list_shared_drives → {r.status_code} "
                        f"(treating as no shared drives)"
                    )
                    return out
                data = r.json()
                for d in data.get("drives", []):
                    out[d["id"]] = d.get("name", "Shared drive")
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"_list_shared_drives failed: {e}")
        return out

    async def _fetch_folders_query(
        self,
        query: str,
        raw: Dict[str, Dict[str, Any]],
        *,
        corpora: Optional[str] = None,
        drive_id: Optional[str] = None,
        shared_with_me: bool = False,
    ) -> None:
        """Paginate a folder query and merge results into `raw` (dedup by id)."""
        params: Dict[str, str] = {
            "q": query,
            "pageSize": "1000",
            "fields": "nextPageToken,files(id,name,parents,driveId)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if corpora:
            params["corpora"] = corpora
        if drive_id:
            params["driveId"] = drive_id
        page_token: Optional[str] = None
        while True:
            if page_token:
                params["pageToken"] = page_token
            else:
                params.pop("pageToken", None)
            token = await self._ensure_valid_token()
            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.get(
                    f"{DRIVE_API}/files",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
            if r.status_code >= 400:
                raise GoogleDriveError(
                    f"Drive folder query → {r.status_code}: {r.text[:300]}"
                )
            data = r.json()
            for f in data.get("files", []):
                if f["id"] in raw:
                    continue
                raw[f["id"]] = {
                    "id": f["id"],
                    "name": f.get("name", "(unnamed)"),
                    "parents": f.get("parents") or [],
                    "driveId": f.get("driveId"),
                    "shared_with_me": shared_with_me,
                }
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    async def list_folders(self) -> List[Dict[str, Any]]:
        """Return every folder the user can pick from, with a computed path.

        Covers THREE distinct Drive categories, each needing a different query
        (this is the part people get wrong — there is no single query that
        returns all of them):
          1. My Drive folders            → corpora=user
          2. Shared drive folders        → corpora=drive&driveId=<id>, ONE
             query per shared drive (corpora=allDrives is unreliable here)
          3. "Shared with me" folders    → sharedWithMe=true

        Each item: {"id", "name", "parents", "path", "shared_drive": <label|None>}.
        """
        FOLDER = "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        shared_drives = await self._list_shared_drives()
        logger.info(f"Drive: user has {len(shared_drives)} shared drive(s)")
        raw: Dict[str, Dict[str, Any]] = {}

        # Pass 1: My Drive
        await self._fetch_folders_query(FOLDER, raw, corpora="user")

        # Pass 2: each shared drive, individually (reliable; allDrives is not)
        for did in shared_drives:
            try:
                await self._fetch_folders_query(
                    FOLDER, raw, corpora="drive", drive_id=did
                )
            except GoogleDriveError as e:
                logger.info(f"shared drive {did} folder query failed: {e}")

        # Pass 3: "Shared with me" folders
        try:
            await self._fetch_folders_query(
                f"{FOLDER} and sharedWithMe = true", raw, shared_with_me=True
            )
        except GoogleDriveError as e:
            logger.info(f"sharedWithMe folder query failed (non-fatal): {e}")

        # Build each folder's slash-joined path by walking parents.
        def _path_and_label(node: Dict[str, Any]) -> tuple[str, Optional[str]]:
            crumbs: List[str] = []
            seen: set = set()
            cur: Optional[str] = node["id"]
            while cur and cur in raw and cur not in seen:
                seen.add(cur)
                n = raw[cur]
                crumbs.append(n["name"])
                parents = n["parents"]
                cur = parents[0] if parents else None
            label: Optional[str] = None
            # cur is now the first parent NOT in our map. If it's a shared
            # drive root, surface the drive name.
            if cur and cur in shared_drives:
                label = shared_drives[cur]
                crumbs.append(label)
            elif node.get("driveId") in shared_drives:
                label = shared_drives[node["driveId"]]
            elif node.get("shared_with_me"):
                label = "Shared with me"
            crumbs.reverse()
            return "/".join(crumbs), label

        out: List[Dict[str, Any]] = []
        for node in raw.values():
            path, label = _path_and_label(node)
            out.append({
                "id": node["id"],
                "name": node["name"],
                "parents": node["parents"],
                "path": path,
                "shared_drive": label,
            })

        # Add each shared drive as a top-level pickable entry. A shared drive's
        # ROOT is not a folder (so it never shows up in the folder query), but
        # users want to pick "HR" / "Services" directly. Using the drive id as
        # the folder id works: list_files_in_folders walks `'<driveId>' in
        # parents` and finds everything in the drive (root files + subfolders).
        for did, dname in shared_drives.items():
            out.append({
                "id": did,
                "name": dname,
                "parents": [],
                "path": dname,
                "shared_drive": dname,
            })

        out.sort(key=lambda x: x["path"].lower())
        n_swm = sum(1 for f in out if f["shared_drive"] == "Shared with me")
        n_team = sum(1 for f in out if f["shared_drive"] and f["shared_drive"] != "Shared with me")
        logger.info(
            f"Drive list_folders: {len(out)} folders "
            f"(my drive + {n_team} in shared drives + {n_swm} shared-with-me)"
        )
        return out

    async def list_files_in_folders(
        self, folder_ids: List[str]
    ) -> AsyncIterator[DriveFile]:
        """Yield supported files inside the given folders AND their subfolders.

        Implemented as a breadth-first walk: for each folder we query its direct
        children (files + subfolders), yield the supported files, and queue the
        subfolders. Pure parent-based traversal — works uniformly for My Drive,
        shared drives, and "shared with me" folders without depending on any
        corpora setting or a pre-fetched folder map.
        """
        if not folder_ids:
            return

        seen_files: set = set()
        visited: set = set()
        queue: List[str] = list(folder_ids)
        folders_walked = 0

        while queue:
            fid = queue.pop()
            if fid in visited:
                continue
            visited.add(fid)
            folders_walked += 1

            # Direct children of this folder (any type).
            query = f"'{fid}' in parents and trashed = false"
            params: Dict[str, str] = {
                "q": query,
                "pageSize": "1000",
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
                # Explicit-parent query + these flags returns children of any
                # folder the user can access (My Drive / shared drive / SWM).
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            page_token: Optional[str] = None
            while True:
                if page_token:
                    params["pageToken"] = page_token
                else:
                    params.pop("pageToken", None)
                token = await self._ensure_valid_token()
                async with httpx.AsyncClient(timeout=30.0) as http:
                    r = await http.get(
                        f"{DRIVE_API}/files",
                        headers={"Authorization": f"Bearer {token}"},
                        params=params,
                    )
                if r.status_code >= 400:
                    raise GoogleDriveError(
                        f"Drive list_files_in_folders → {r.status_code}: {r.text[:300]}"
                    )
                data = r.json()
                for f in data.get("files", []):
                    mime = f.get("mimeType", "")
                    if mime == "application/vnd.google-apps.folder":
                        if f["id"] not in visited:
                            queue.append(f["id"])  # recurse into subfolder
                        continue
                    if mime not in _INCLUDE_MIME or f["id"] in seen_files:
                        continue
                    seen_files.add(f["id"])
                    modified: Optional[datetime] = None
                    if (m := f.get("modifiedTime")):
                        modified = datetime.fromisoformat(m.replace("Z", "+00:00"))
                    yield DriveFile(
                        id=f["id"],
                        name=f.get("name", "(unnamed)"),
                        mime_type=mime,
                        size=int(f.get("size") or 0),
                        modified_at=modified,
                        web_url=f.get("webViewLink"),
                    )
                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        logger.info(
            f"Drive folder-scoped discovery: walked {folders_walked} folders, "
            f"found {len(seen_files)} files"
        )

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Fetch a file's metadata — useful if the Picker only gave us the id."""
        return await self._get_json(
            f"{DRIVE_API}/files/{file_id}",
            params={
                "fields": "id,name,mimeType,size,modifiedTime,webViewLink",
                "supportsAllDrives": "true",
            },
        )

    async def download_file(
        self, file_id: str, mime_type: str
    ) -> Tuple[bytes, str]:
        """Return (bytes, effective_mime).

        Google-native types (Docs, Sheets, Slides) are exported as plain
        text / CSV so the rest of the pipeline doesn't need special handling.
        Native types we don't know how to export raise immediately so we
        don't waste an API call on a 403.
        """
        if mime_type.startswith("application/vnd.google-apps."):
            export_as = _EXPORT_MIME.get(mime_type)
            if export_as is None:
                raise GoogleDriveError(
                    f"Unsupported Google-native MIME type: {mime_type}"
                )
            content = await self._get_bytes(
                f"{DRIVE_API}/files/{file_id}/export",
                params={"mimeType": export_as},
            )
            return content, export_as

        # Regular binary file (PDF, DOCX, image, etc.)
        content = await self._get_bytes(
            f"{DRIVE_API}/files/{file_id}",
            params={"alt": "media"},
        )
        return content, mime_type
