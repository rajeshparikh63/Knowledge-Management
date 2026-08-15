"""
MongoDB data access for the multi-organization domain.

Collections (in the `soldieriq` database):
- organizations   : {_id, name, slug, owner_user_id, created_at, updated_at}
- user_profiles   : {_id=<keycloak sub>, email, firstName, lastName,
                     current_organization_id,
                     organizations: [{organization_id, role, status, joined_at}],
                     created_at, updated_at}
- invitations     : {_id, organization_id, organization_name, email, role,
                     token_hash, status, invited_by, invited_by_name,
                     document_ids, created_at, expires_at, accepted_by, accepted_at}

The user id is the Keycloak subject (`sub`) string — identity stays in Keycloak;
this layer only owns organization membership, roles, and invites.
"""

import hashlib
import secrets
import threading
import uuid
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from auth.database import get_mongodb_client
from app.logger import logger


VALID_ROLES = {"admin", "user"}
INVITATION_TTL_DAYS = 7


def _now() -> datetime:
    return datetime.utcnow()


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (name or "org").lower()).strip("-") or "org"
    return f"{base}-{uuid.uuid4().hex[:6]}"


# ============================================================================
# Organizations
# ============================================================================
class OrganizationCRUD:
    def __init__(self):
        self.collection = get_mongodb_client().get_collection("organizations")

    def get(self, org_id: str) -> Optional[dict]:
        return self.collection.find_one({"_id": org_id})

    def create(self, name: str, owner_user_id: str, org_id: Optional[str] = None) -> dict:
        """Create an organization. If org_id is provided (e.g. an existing
        Keycloak personal-org UUID), it is used as the _id so the FalkorDB
        graph `org_<id>` continues to line up."""
        oid = org_id or str(uuid.uuid4())
        doc = {
            "_id": oid,
            "name": name,
            "slug": _slugify(name),
            "owner_user_id": owner_user_id,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.collection.insert_one(doc)
        return doc

    def ensure(self, org_id: str, name: str, owner_user_id: str) -> dict:
        """Return the org doc, creating it with the given id if missing."""
        existing = self.get(org_id)
        if existing:
            return existing
        return self.create(name=name, owner_user_id=owner_user_id, org_id=org_id)

    def update_name(self, org_id: str, name: str) -> Optional[dict]:
        self.collection.update_one(
            {"_id": org_id}, {"$set": {"name": name, "updated_at": _now()}}
        )
        return self.get(org_id)

    def names_for(self, org_ids: List[str]) -> Dict[str, str]:
        """Batch id -> name lookup."""
        if not org_ids:
            return {}
        cursor = self.collection.find({"_id": {"$in": list(set(org_ids))}}, {"name": 1})
        return {d["_id"]: d.get("name", "Organization") for d in cursor}


# ============================================================================
# User profiles (memberships + active-org pointer), keyed by Keycloak sub
# ============================================================================
class UserProfileCRUD:
    def __init__(self):
        self.collection = get_mongodb_client().get_collection("user_profiles")

    def get(self, user_id: str) -> Optional[dict]:
        return self.collection.find_one({"_id": user_id})

    def get_or_provision(
        self,
        user_id: str,
        email: Optional[str],
        first_name: Optional[str],
        last_name: Optional[str],
        personal_org_id: Optional[str],
        personal_org_name: Optional[str],
    ) -> dict:
        """Return the user's profile, creating it on first sight.

        On creation the user's Keycloak personal organization (the
        `organization_id` attribute set at signup) becomes their first
        membership as admin. This lazily migrates pre-existing single-org
        users the first time they hit an org-scoped endpoint.
        """
        profile = self.get(user_id)
        if profile:
            return profile

        org_id = personal_org_id or str(uuid.uuid4())
        org_name = personal_org_name or (
            f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()
            or (email or "My")
        ) + "'s Organization"

        # Make sure the organization document exists (owned by this user).
        OrganizationCRUD().ensure(org_id=org_id, name=org_name, owner_user_id=user_id)

        now = _now()
        profile = {
            "_id": user_id,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
            "current_organization_id": org_id,
            "organizations": [
                {
                    "organization_id": org_id,
                    "role": "admin",
                    "status": "active",
                    "joined_at": now,
                }
            ],
            "created_at": now,
            "updated_at": now,
        }
        try:
            self.collection.insert_one(profile)
        except Exception as e:
            # Race: another concurrent request created it first — re-read.
            logger.warning(f"[orgs] profile insert race for {user_id}: {e}")
            existing = self.get(user_id)
            if existing:
                return existing
            raise
        return profile

    def _active_memberships(self, profile: dict) -> List[dict]:
        return [
            m for m in profile.get("organizations", [])
            if m.get("status") == "active"
        ]

    def resolve_active_membership(self, profile: dict) -> Optional[dict]:
        """Return the membership for the profile's current organization,
        falling back to the first active membership (and persisting the
        change) when the current one is missing or inactive."""
        current_id = profile.get("current_organization_id")
        memberships = self._active_memberships(profile)
        if not memberships:
            return None

        current = next(
            (m for m in memberships if m["organization_id"] == current_id), None
        )
        if current is None:
            current = memberships[0]
            self.set_current(profile["_id"], current["organization_id"])
            profile["current_organization_id"] = current["organization_id"]

        # Attach the org display name for convenience.
        org = OrganizationCRUD().get(current["organization_id"])
        enriched = dict(current)
        enriched["organization_name"] = org.get("name") if org else None
        return enriched

    def list_workspace(self, profile: dict) -> List[dict]:
        """Active memberships shaped for the workspace switcher."""
        memberships = self._active_memberships(profile)
        names = OrganizationCRUD().names_for([m["organization_id"] for m in memberships])
        current_id = profile.get("current_organization_id")
        return [
            {
                "id": m["organization_id"],
                "name": names.get(m["organization_id"], "Organization"),
                "role": m.get("role", "user"),
                "status": m.get("status", "active"),
                "is_current": m["organization_id"] == current_id,
            }
            for m in memberships
        ]

    def get_membership(self, user_id: str, org_id: str) -> Optional[dict]:
        profile = self.get(user_id)
        if not profile:
            return None
        return next(
            (m for m in profile.get("organizations", []) if m["organization_id"] == org_id),
            None,
        )

    def is_active_member(self, user_id: str, org_id: str) -> bool:
        m = self.get_membership(user_id, org_id)
        return bool(m and m.get("status") == "active")

    def set_current(self, user_id: str, org_id: str) -> None:
        self.collection.update_one(
            {"_id": user_id},
            {"$set": {"current_organization_id": org_id, "updated_at": _now()}},
        )

    def add_or_reactivate_membership(
        self, user_id: str, org_id: str, role: str, make_current: bool = True
    ) -> None:
        """Add a membership, or reactivate one previously removed."""
        if role not in VALID_ROLES:
            role = "user"
        existing = self.get_membership(user_id, org_id)
        now = _now()
        if existing is None:
            update: Dict[str, Any] = {
                "$push": {
                    "organizations": {
                        "organization_id": org_id,
                        "role": role,
                        "status": "active",
                        "joined_at": now,
                    }
                },
                "$set": {"updated_at": now},
            }
            if make_current:
                update["$set"]["current_organization_id"] = org_id
            self.collection.update_one({"_id": user_id}, update)
        else:
            set_fields = {
                "organizations.$.status": "active",
                "organizations.$.role": role,
                "organizations.$.joined_at": now,
                "updated_at": now,
            }
            if make_current:
                set_fields["current_organization_id"] = org_id
            self.collection.update_one(
                {"_id": user_id, "organizations.organization_id": org_id},
                {"$set": set_fields},
            )

    def set_membership_status(self, user_id: str, org_id: str, status: str) -> None:
        self.collection.update_one(
            {"_id": user_id, "organizations.organization_id": org_id},
            {"$set": {"organizations.$.status": status, "updated_at": _now()}},
        )

    def set_membership_role(self, user_id: str, org_id: str, role: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError("Invalid role")
        self.collection.update_one(
            {"_id": user_id, "organizations.organization_id": org_id},
            {"$set": {"organizations.$.role": role, "updated_at": _now()}},
        )

    def list_members(self, org_id: str, include_removed: bool = False) -> List[dict]:
        """All user profiles with a membership in this organization, flattened
        to that org's role/status."""
        query = {"organizations.organization_id": org_id}
        members = []
        for profile in self.collection.find(query):
            membership = next(
                (m for m in profile.get("organizations", []) if m["organization_id"] == org_id),
                None,
            )
            if not membership:
                continue
            if not include_removed and membership.get("status") != "active":
                continue
            members.append(
                {
                    "user_id": profile["_id"],
                    "email": profile.get("email"),
                    "firstName": profile.get("firstName"),
                    "lastName": profile.get("lastName"),
                    "role": membership.get("role", "user"),
                    "status": membership.get("status", "active"),
                    "joined_at": membership.get("joined_at"),
                }
            )
        return members

    def find_by_email(self, email: str) -> Optional[dict]:
        return self.collection.find_one({"email": email})


# ============================================================================
# Invitations
# ============================================================================
class InvitationCRUD:
    def __init__(self):
        self.collection = get_mongodb_client().get_collection("invitations")

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(
        self,
        org_id: str,
        org_name: str,
        email: str,
        role: str,
        invited_by: str,
        invited_by_name: Optional[str],
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, str]:
        """Create a pending invitation. Returns (doc, plain_token).
        The plain token is only returned here (for the email link); only its
        hash is stored."""
        if role not in VALID_ROLES:
            raise ValueError("Role must be 'admin' or 'user'")

        # Reject if there's already a pending invite for this email+org.
        existing = self.collection.find_one(
            {"organization_id": org_id, "email": email, "status": "pending"}
        )
        if existing:
            raise ValueError("A pending invitation already exists for this email")

        token = secrets.token_urlsafe(32)
        now = _now()
        doc = {
            "_id": str(uuid.uuid4()),
            "organization_id": org_id,
            "organization_name": org_name,
            "email": email,
            "role": role,
            "token_hash": self._hash(token),
            "status": "pending",
            "invited_by": invited_by,
            "invited_by_name": invited_by_name,
            "document_ids": document_ids or [],
            "created_at": now,
            "expires_at": now + timedelta(days=INVITATION_TTL_DAYS),
            "accepted_by": None,
            "accepted_at": None,
        }
        self.collection.insert_one(doc)
        return doc, token

    def validate(self, token: str) -> dict:
        """Return the invitation for a token, or raise ValueError if invalid,
        expired, or already used."""
        doc = self.collection.find_one({"token_hash": self._hash(token)})
        if not doc:
            raise ValueError("Invalid invitation")
        if doc.get("status") != "pending":
            raise ValueError("This invitation has already been used or revoked")
        expires_at = doc.get("expires_at")
        if expires_at and expires_at < _now():
            self.collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "expired"}})
            raise ValueError("This invitation has expired")
        return doc

    def accept(self, token: str, user_id: str) -> None:
        self.collection.update_one(
            {"token_hash": self._hash(token)},
            {"$set": {"status": "accepted", "accepted_by": user_id, "accepted_at": _now()}},
        )

    def revoke(self, invitation_id: str, org_id: str) -> bool:
        result = self.collection.update_one(
            {"_id": invitation_id, "organization_id": org_id, "status": "pending"},
            {"$set": {"status": "revoked"}},
        )
        return result.modified_count > 0

    def list_for_org(self, org_id: str, status: Optional[str] = None) -> List[dict]:
        query: Dict[str, Any] = {"organization_id": org_id}
        if status:
            query["status"] = status
        return list(self.collection.find(query).sort("created_at", -1))


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_org_crud: Optional[OrganizationCRUD] = None
_profile_crud: Optional[UserProfileCRUD] = None
_invitation_crud: Optional[InvitationCRUD] = None
_lock = threading.RLock()


def get_organization_crud() -> OrganizationCRUD:
    global _org_crud
    with _lock:
        if _org_crud is None:
            _org_crud = OrganizationCRUD()
        return _org_crud


def get_user_profile_crud() -> UserProfileCRUD:
    global _profile_crud
    with _lock:
        if _profile_crud is None:
            _profile_crud = UserProfileCRUD()
        return _profile_crud


def get_invitation_crud() -> InvitationCRUD:
    global _invitation_crud
    with _lock:
        if _invitation_crud is None:
            _invitation_crud = InvitationCRUD()
        return _invitation_crud
