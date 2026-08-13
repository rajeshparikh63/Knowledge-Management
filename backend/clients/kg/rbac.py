"""
RBAC subgraph module — separate from the document writer (different lifecycle).

Subgraph:
  (:Organization {org_id, name})-[:HAS_MEMBER]->(:User {email, username})
  (:User)-[:HAS_ACCESS {granted_at, granted_by}]->(:Document {document_id})

Called by three flows, NEVER by the document writer:
  * invite / accept        -> add_user
  * assign / revoke access -> grant_access / revoke_access
  * manifest ingestion     -> grant_access_bulk (after the writer creates the Doc)

`accessible_document_ids(email)` is THE retrieval pre-filter: it returns the doc
set a caller may see, which retrieval scopes the vector search to (so results
are always non-empty and always permitted — never post-filtered to zero).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.logger import logger
from clients.kg.store import get_graph, ensure_indexes


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RBACManager:
    def __init__(self, organization_id: str):
        self.org_id = organization_id

    def _run(self, cypher: str, params: Optional[dict] = None):
        g = get_graph(self.org_id)
        return g.query(cypher, params or {}).result_set

    # -- organization / users ---------------------------------------------

    async def ensure_organization(self, name: str = "") -> None:
        await asyncio.to_thread(lambda: (
            ensure_indexes(get_graph(self.org_id)),
            self._run(
                "MERGE (o:Organization {org_id: $id}) "
                "SET o.name = coalesce($name, o.name), o.updated_at = $ts",
                {"id": self.org_id, "name": name, "ts": _now()},
            ),
        ))

    async def add_user(self, email: str, username: str = "") -> None:
        """Invite/accept: create the User and attach to the org."""
        email = email.strip().lower()
        await asyncio.to_thread(lambda: self._run(
            """
            MERGE (o:Organization {org_id: $org})
            MERGE (u:User {email: $email})
            SET u.username = coalesce($username, u.username), u.updated_at = $ts
            MERGE (o)-[:HAS_MEMBER]->(u)
            """,
            {"org": self.org_id, "email": email, "username": username, "ts": _now()},
        ))

    async def remove_user(self, email: str) -> None:
        email = email.strip().lower()
        await asyncio.to_thread(lambda: self._run(
            "MATCH (u:User {email: $email}) DETACH DELETE u", {"email": email}))

    # -- access grants -----------------------------------------------------

    async def grant_access(self, email: str, document_id: str, granted_by: str = "") -> None:
        """Assign a single document to a user. Idempotent; ensures the User and
        org membership exist so a grant can precede the user's acceptance."""
        email = email.strip().lower()
        await asyncio.to_thread(lambda: self._run(
            """
            MERGE (o:Organization {org_id: $org})
            MERGE (u:User {email: $email})
            MERGE (o)-[:HAS_MEMBER]->(u)
            WITH u
            MATCH (d:Document {document_id: $doc})
            MERGE (u)-[a:HAS_ACCESS]->(d)
            SET a.granted_at = $ts, a.granted_by = $by
            """,
            {"org": self.org_id, "email": email, "doc": document_id,
             "ts": _now(), "by": granted_by},
        ))

    async def grant_access_bulk(self, document_id: str, emails: List[str],
                                granted_by: str = "") -> int:
        """Manifest flow: grant one document to many emails at once."""
        rows = [{"email": e.strip().lower()} for e in emails if e and e.strip()]
        if not rows:
            return 0
        await asyncio.to_thread(lambda: self._run(
            """
            MATCH (d:Document {document_id: $doc})
            MERGE (o:Organization {org_id: $org})
            UNWIND $rows AS row
            MERGE (u:User {email: row.email})
            MERGE (o)-[:HAS_MEMBER]->(u)
            MERGE (u)-[a:HAS_ACCESS]->(d)
            SET a.granted_at = $ts, a.granted_by = $by
            """,
            {"org": self.org_id, "doc": document_id, "rows": rows,
             "ts": _now(), "by": granted_by},
        ))
        return len(rows)

    async def revoke_access(self, email: str, document_id: str) -> None:
        email = email.strip().lower()
        await asyncio.to_thread(lambda: self._run(
            """
            MATCH (:User {email: $email})-[a:HAS_ACCESS]->(:Document {document_id: $doc})
            DELETE a
            """,
            {"email": email, "doc": document_id},
        ))

    # -- queries -----------------------------------------------------------

    async def accessible_document_ids(self, email: str) -> List[str]:
        """THE pre-filter: document_ids this user may see."""
        email = email.strip().lower()
        rows = await asyncio.to_thread(lambda: self._run(
            "MATCH (:User {email: $email})-[:HAS_ACCESS]->(d:Document) "
            "RETURN d.document_id",
            {"email": email},
        ))
        return [r[0] for r in rows if r[0]]

    async def who_can_access(self, document_id: str) -> List[str]:
        rows = await asyncio.to_thread(lambda: self._run(
            "MATCH (u:User)-[:HAS_ACCESS]->(:Document {document_id: $doc}) "
            "RETURN u.email",
            {"doc": document_id},
        ))
        return [r[0] for r in rows if r[0]]

    async def list_users(self) -> List[Dict[str, str]]:
        rows = await asyncio.to_thread(lambda: self._run(
            "MATCH (:Organization {org_id: $org})-[:HAS_MEMBER]->(u:User) "
            "RETURN u.email, u.username",
            {"org": self.org_id},
        ))
        return [{"email": r[0], "username": r[1]} for r in rows]
