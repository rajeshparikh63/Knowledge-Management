"""
SharePoint client backed by Composio.

We deliberately do NOT manage Microsoft OAuth tokens — Composio owns the OAuth
flow and refresh. We identify each of OUR users to Composio by their real
user_id (the Composio "entity") — one connection per user, NOT a shared account.

Connect uses Composio's hosted link flow (`connected_accounts.link`), which
returns a connect.composio.dev URL that collects the tenant subdomain and runs
Microsoft consent itself, then redirects to our callback_url. After that, tool
calls (list sites/drives/children, download) just work for this user_id.

Patterns (tool execution, site-scoped listing, response parsing) mirror the
working AI-Agency implementation. File bytes are fetched from the temporary S3
locator Composio's download tool returns — the single swap-point if a later
compliance review requires pulling straight from Microsoft Graph.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.logger import logger
from app.settings import settings


class SharePointError(Exception):
    """Raised for any SharePoint/Composio interaction failure."""


# File types our ingestion pipeline can handle (mirrors the Drive whitelist).
_SUPPORTED_EXT = {
    "pdf", "doc", "docx", "ppt", "pptx", "txt", "md", "markdown",
    "csv", "tsv", "xls", "xlsx", "rtf", "html", "htm", "epub",
}

_client_singleton = None
_client_lock = threading.Lock()


def _composio():
    """Process-wide singleton Composio client (cheap to reuse, thread-safe)."""
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:
                from composio import Composio
                if not settings.COMPOSIO_API_KEY:
                    raise SharePointError("COMPOSIO_API_KEY is not configured")
                _client_singleton = Composio(api_key=settings.COMPOSIO_API_KEY)
                logger.info("✅ Composio client initialized (SharePoint)")
    return _client_singleton


class SharePointClient:
    """Per-user SharePoint access via Composio. Methods are synchronous (the
    Composio SDK is sync); async callers wrap with asyncio.to_thread, the Celery
    worker calls directly."""

    TOOLKIT = "share_point"

    def __init__(self, user_id: str):
        if not user_id:
            raise SharePointError("user_id is required")
        self.user_id = str(user_id)
        self._c = _composio()

    # ---- connection management (no tokens stored on our side) -----------
    def initiate_connection(self, callback_url: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Start the hosted Composio connect flow for this user. Returns the
        auth URL to redirect to (it collects the tenant subdomain + runs MS
        consent) and the new connection id."""
        req = self._c.connected_accounts.link(
            user_id=self.user_id,
            auth_config_id=settings.COMPOSIO_SHAREPOINT_AUTH_CONFIG_ID,
            callback_url=callback_url,
        )
        return {
            "auth_url": getattr(req, "redirect_url", None),
            "connection_id": getattr(req, "id", None),
        }

    def _list_my_connections(self) -> List[Any]:
        try:
            res = self._c.connected_accounts.list(
                user_ids=[self.user_id], toolkit_slugs=[self.TOOLKIT]
            )
        except TypeError:
            res = self._c.connected_accounts.list()
        items = getattr(res, "items", res) or []
        mine: List[Any] = []
        for a in items:
            tk = getattr(a, "toolkit", None)
            slug = getattr(tk, "slug", tk)
            auid = getattr(a, "user_id", None)
            if slug == self.TOOLKIT and auid in (None, self.user_id):
                mine.append(a)
        return mine

    def connection_status(self) -> Dict[str, Any]:
        mine = self._list_my_connections()
        active = next(
            (a for a in mine if str(getattr(a, "status", "")).upper() == "ACTIVE"),
            None,
        )
        if active is not None:
            return {"connected": True, "status": "ACTIVE",
                    "connection_id": getattr(active, "id", None)}
        if mine:
            a = mine[0]
            return {"connected": False, "status": str(getattr(a, "status", "")).upper() or None,
                    "connection_id": getattr(a, "id", None)}
        return {"connected": False, "status": None, "connection_id": None}

    def disconnect(self) -> int:
        n = 0
        for a in self._list_my_connections():
            try:
                self._c.connected_accounts.delete(getattr(a, "id"))
                n += 1
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"SharePoint disconnect failed for {getattr(a, 'id', '?')}: {e}")
        return n

    # ---- tool execution helper ------------------------------------------
    def _execute(self, slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._c.tools.execute(
            slug, arguments, user_id=self.user_id, dangerously_skip_version_check=True
        )
        ok, err = _resp_ok(resp)
        if not ok:
            raise SharePointError(f"{slug} failed: {err}")
        return _resp_data(resp)

    # ---- listing -------------------------------------------------------
    def list_libraries(self) -> List[Dict[str, Optional[str]]]:
        """Document libraries (drives) across EVERY SharePoint site the user can
        see — the root/communication site, all /sites/ team sites, and any other
        site — EXCEPT personal OneDrive. Each is a pickable unit (like a Drive
        folder) and carries enough scope info for walk/download to target it.

        Scoping per site (LIST_DRIVES_REST_API):
          - /sites/<name>           → site_name=<name>          (team site)
          - bare root host          → no scope arg              (root/comms site)
          - anything else           → sharepoint_site_url=<url> (best effort)
        """
        out: List[Dict[str, Optional[str]]] = []
        seen_drives: set = set()
        sites = _items(self._execute("SHARE_POINT_LIST_SITES", {"search": "*", "top": 200}))
        for s in sites:
            web_url = _first(s, "webUrl", "name", "url", default="")
            if not web_url:
                continue
            low = web_url.lower()
            # Skip personal OneDrive only — include all real SharePoint sites.
            if "-my.sharepoint.com" in low or "/personal/" in low:
                continue

            scope = _site_scope(web_url)  # '<name>' for /sites/<name>, else None
            site_disp = _first(s, "displayName", "Title", "name", default=scope or "SharePoint")

            args: Dict[str, Any] = {"select": "id,name,webUrl,driveType", "top": 100}
            if scope:
                args["site_name"] = scope
            elif urlparse(web_url).path.strip("/"):
                # Non-root, non-/sites/ site (e.g. contentstorage) — scope by URL.
                args["sharepoint_site_url"] = web_url
            # else: bare root host → no scope arg (defaults to the root site).

            try:
                drives = _items(self._execute("SHARE_POINT_LIST_DRIVES_REST_API", args))
            except SharePointError as e:
                logger.warning(f"list drives failed for site '{site_disp}': {e}")
                continue

            for d in drives:
                did = _first(d, "id", "Id")
                if not did or did in seen_drives:
                    continue
                seen_drives.add(did)
                lib_name = _first(d, "name", "Name", default="Documents")
                out.append({
                    "id": did,
                    "name": lib_name,
                    "site_name": scope,
                    "site_display": site_disp,
                    "path": f"{site_disp}/{lib_name}",
                    "web_url": _first(d, "webUrl"),
                })
        return out

    def walk_drive(self, drive_id: str) -> List[Dict[str, Any]]:
        """BFS over a drive's folder tree; returns supported FILE items (each
        carries id/name/webUrl/size needed to download)."""
        files: List[Dict[str, Any]] = []
        queue: List[Optional[str]] = [None]
        seen: set = set()
        select = "id,name,size,folder,file,webUrl,parentReference"

        while queue:
            folder_id = queue.pop(0)
            key = folder_id or "root"
            if key in seen:
                continue
            seen.add(key)

            args: Dict[str, Any] = {"drive_id": drive_id, "top": 200, "select": select}
            if folder_id:
                args["folder_id"] = folder_id
            try:
                children = _items(self._execute("SHARE_POINT_LIST_DRIVE_CHILDREN", args))
            except SharePointError as e:
                logger.warning(f"list children failed (drive={drive_id[:8]}… folder={key}): {e}")
                continue

            for it in children:
                if it.get("folder") is not None:
                    iid = _first(it, "id", "Id")
                    if iid:
                        queue.append(iid)
                else:
                    name = _clean_name_str(_first(it, "name", "Name", default=""))
                    if _is_supported(name):
                        it["name"] = name  # store the cleaned filename
                        it["_drive_id"] = drive_id
                        files.append(it)
        return files

    # ---- download -------------------------------------------------------
    def download_file(self, web_url: str, name: str = "file") -> Tuple[bytes, str, str]:
        """Download one file's bytes via Composio's locator, addressed by its
        server-relative URL (derived from the item's webUrl)."""
        sru = _server_relative_url(web_url)
        if not sru:
            raise SharePointError(f"cannot derive server-relative URL for {name} (web_url={web_url})")

        data = self._execute(
            "SHARE_POINT_DOWNLOAD_FILE_BY_SERVER_RELATIVE_URL",
            {"server_relative_url": sru},
        )
        content = (data or {}).get("content") or {}
        s3url = content.get("s3url") or content.get("s3Url")
        mime = content.get("mimetype") or content.get("mimeType") or "application/octet-stream"
        if not s3url:
            raise SharePointError(f"no download locator returned for {name} (sru={sru})")

        r = httpx.get(s3url, timeout=180, follow_redirects=True)
        r.raise_for_status()
        # Use the clean filename we were given — NOT content.get("name"), which
        # echoes back the (Doc.aspx) server-relative URL we queried with and is
        # useless for extension-based extractor routing.
        return r.content, mime, _clean_name_str(name)


# ---- module helpers (mirror AI-Agency) ----------------------------------
def _resp_data(resp) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp.get("data") or {}
    return getattr(resp, "data", None) or {}


def _resp_ok(resp) -> Tuple[bool, Optional[str]]:
    if isinstance(resp, dict):
        return bool(resp.get("successful", True)), resp.get("error")
    return bool(getattr(resp, "successful", True)), getattr(resp, "error", None)


def _items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the records list out of a Composio/Graph response, whatever the key."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("value", "items", "drives", "sites", "results", "children", "lists"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        inner = data.get("data")
        if isinstance(inner, (list, dict)):
            return _items(inner)
    return []


def _first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


def _site_scope(web_url: str) -> Optional[str]:
    """'https://tenant.sharepoint.com/sites/kroolo.com/...' -> 'kroolo.com'.
    None for personal/root sites (no team document libraries to crawl)."""
    if not web_url or "/sites/" not in web_url:
        return None
    return web_url.split("/sites/", 1)[1].split("/", 1)[0]


def _server_relative_url(web_url: str) -> Optional[str]:
    """Server-relative URL (starts with '/') the download tool needs.
    For team sites: strip scheme+host, keep the rest (incl. %20 encoding)."""
    if not web_url:
        return None
    if "/sites/" in web_url:
        host = web_url.split("/sites/", 1)[0]  # https://<sub>.sharepoint.com
        return web_url[len(host):]             # /sites/<scope>/<lib>/<path>
    path = urlparse(web_url).path
    return path or None


def _is_supported(name: str) -> bool:
    ext = os.path.splitext(name or "")[1].lstrip(".").lower()
    return ext in _SUPPORTED_EXT


def _clean_name_str(name: str) -> str:
    """Return a usable filename. SharePoint sometimes hands us a viewer URL for
    Office files (…/Doc.aspx?sourcedoc={GUID}&file=Real-Name.docx&action=…);
    pull the real filename out of the `file=` query param, else take the last
    path segment, and always drop any query string."""
    from urllib.parse import parse_qs, unquote, urlparse

    n = (name or "").strip()
    if not n:
        return "file"
    if "?" in n or "/" in n or "Doc.aspx" in n:
        parsed = urlparse(n)
        q = parse_qs(parsed.query)
        if q.get("file"):
            return unquote(q["file"][0])
        seg = (parsed.path or n.split("?", 1)[0]).rstrip("/").split("/")[-1]
        return unquote(seg) or "file"
    return n


def get_sharepoint_client(user_id: str) -> SharePointClient:
    return SharePointClient(user_id)
