"""
Workspace router — list the organizations the current user belongs to and
switch the active one.

Switching updates server-side state only (the user_profiles
`current_organization_id`); no token is reissued. Because every org-scoped
route resolves its organization through `get_current_context`, the next
request after a switch is automatically scoped to the new organization.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List

from orgs.dependencies import get_current_context
from orgs.crud import get_user_profile_crud
from orgs.models import WorkspaceOrganization, SwitchOrganizationRequest
from app.logger import logger

router = APIRouter(prefix="/workspace", tags=["workspace"])


@router.get("/organizations", response_model=List[WorkspaceOrganization])
async def get_user_organizations(context: Dict = Depends(get_current_context)):
    """All organizations the current user is an active member of."""
    profiles = get_user_profile_crud()
    profile = context.get("profile") or profiles.get(context["id"])
    return profiles.list_workspace(profile)


@router.post("/switch")
async def switch_organization(
    payload: SwitchOrganizationRequest,
    context: Dict = Depends(get_current_context),
):
    """Switch the active organization.

    Verifies the user is an active member of the target org, then repoints
    their current organization. Returns void-ish; the client refetches
    `/auth/me` (and reloads) to pick up the new scope.
    """
    user_id = context["id"]
    org_id = payload.organization_id
    profiles = get_user_profile_crud()

    if not profiles.is_active_member(user_id, org_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization",
        )

    profiles.set_current(user_id, org_id)
    logger.info(f"[workspace] {user_id} switched to org {org_id}")
    return {"message": "Organization switched successfully", "organization_id": org_id}
