"""
Multi-organization domain package.

Keycloak remains the identity provider (authenticates the person). This package
owns the *organization domain* in MongoDB: organizations, per-user memberships
(with role), the active-organization pointer, and email invitations.

Layers:
- models.py        Pydantic request/response models
- crud.py          MongoDB data access (organizations, user_profiles, invitations)
- dependencies.py  get_current_context / require_org_admin FastAPI dependencies
"""
