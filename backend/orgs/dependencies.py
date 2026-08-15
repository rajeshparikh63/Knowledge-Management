"""
Organization-aware request context.

`get_current_context` wraps Keycloak identity (`get_current_user_keycloak`)
and layers on the active organization resolved from MongoDB. It returns the
same dict shape existing routes already consume — `current_user["organization_id"]`
and `current_user["role"]` — but the values now reflect the user's *active*
organization (which they can switch), not a fixed Keycloak attribute.

Existing routes migrate by swapping their dependency:
    Depends(get_current_user_keycloak)  ->  Depends(get_current_context)
"""

from fastapi import Depends, HTTPException, status
from typing import Dict

from auth.keycloak_auth import get_current_user_keycloak
from orgs.crud import get_user_profile_crud
from app.logger import logger


async def get_current_context(
    current_user: Dict = Depends(get_current_user_keycloak),
) -> Dict:
    """Identity (Keycloak) + active organization (Mongo).

    Adds/overwrites on the returned dict:
    - organization_id   : the active org's id (namespaces the FalkorDB graph)
    - organization_name : the active org's display name
    - role              : the user's role in the active org ("admin" | "user")
    - profile           : the raw user_profiles document
    """
    user_id = current_user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing subject",
        )

    profiles = get_user_profile_crud()

    # Provision on first sight (auto-migrates existing single-org users using
    # the personal-org UUID carried in their Keycloak token).
    profile = profiles.get_or_provision(
        user_id=user_id,
        email=current_user.get("email"),
        first_name=current_user.get("firstName"),
        last_name=current_user.get("lastName"),
        personal_org_id=current_user.get("organization_id"),
        personal_org_name=current_user.get("organization_name"),
    )

    active = profiles.resolve_active_membership(profile)
    if not active:
        # Every provisioned user has at least their personal org, so this only
        # happens if they were removed from all orgs.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not an active member of any organization",
        )

    current_user["organization_id"] = active["organization_id"]
    current_user["organization_name"] = active.get("organization_name")
    current_user["role"] = active.get("role", "user")
    current_user["profile"] = profile
    return current_user


async def require_org_admin(context: Dict = Depends(get_current_context)) -> Dict:
    """Require the caller to be an admin of their active organization."""
    if context.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this organization",
        )
    return context
