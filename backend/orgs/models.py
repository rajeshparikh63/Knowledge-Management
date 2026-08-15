"""
Pydantic models for the multi-organization domain.

Roles are a fixed two-tier set: "admin" and "user" (an org's owner is the
member whose user id equals Organization.owner_user_id; there is no separate
"owner" role value).
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


VALID_ROLES = {"admin", "user"}


# ----------------------------------------------------------------------------
# Organizations
# ----------------------------------------------------------------------------
class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    owner_user_id: str
    created_at: Optional[datetime] = None


# ----------------------------------------------------------------------------
# Workspace (the orgs the current user belongs to + switching)
# ----------------------------------------------------------------------------
class WorkspaceOrganization(BaseModel):
    """One organization the current user is an active member of."""
    id: str                 # organization_id
    name: str
    role: str               # "admin" | "user"
    status: str             # "active"
    is_current: bool = False


class SwitchOrganizationRequest(BaseModel):
    organization_id: str


# ----------------------------------------------------------------------------
# Members (people inside a given organization)
# ----------------------------------------------------------------------------
class OrganizationMember(BaseModel):
    user_id: str
    email: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    role: str
    status: str
    joined_at: Optional[datetime] = None
    is_owner: bool = False


# ----------------------------------------------------------------------------
# Invitations
# ----------------------------------------------------------------------------
class InviteUserRequest(BaseModel):
    """Body for POST /api/invitations (org admin only)."""
    email: EmailStr = Field(..., description="Email address to invite")
    role: str = Field("user", description="Role in the org: 'admin' or 'user'")
    document_ids: List[str] = Field(
        default_factory=list,
        description="Document ids to grant the invitee access to on acceptance",
    )


class AcceptInvitationRequest(BaseModel):
    """Body for POST /api/invitations/accept (public).

    firstName / lastName / password are required only when the invited email
    does not yet have an account (a brand-new user is provisioned in Keycloak).
    """
    token: str = Field(..., description="Invitation token from the email link")
    firstName: Optional[str] = Field(None, min_length=1, max_length=100)
    lastName: Optional[str] = Field(None, min_length=1, max_length=100)
    password: Optional[str] = Field(None, min_length=8, max_length=128)


class InvitationResponse(BaseModel):
    id: str
    organization_id: str
    organization_name: Optional[str] = None
    email: str
    role: str
    status: str                      # pending | accepted | revoked | expired
    invited_by_name: Optional[str] = None
    createdAt: Optional[datetime] = None
    expiresAt: Optional[datetime] = None


class ValidateInvitationResponse(BaseModel):
    email: str
    organization_name: str
    role: str
    invited_by_name: Optional[str] = None
    user_exists: bool
    createdAt: Optional[datetime] = None
    expiresAt: Optional[datetime] = None
