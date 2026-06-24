-- ============================================================================
-- MAIN DATABASE SCHEMA
-- SoldierIQ Knowledge Management System
-- Generated: 2026-03-09 04:13:24
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE: documents
-- Current rows: 5
-- ============================================================================

-- Indexes for documents
CREATE INDEX IF NOT EXISTS idx_documents_created ON public.documents USING btree (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_folder ON public.documents USING btree (folder_name);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON public.documents USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_documents_org_user ON public.documents USING btree (organization_id, user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents USING btree (status);

-- ============================================================================
-- TABLE: podcasts
-- Current rows: 0
-- ============================================================================

-- Indexes for podcasts
CREATE INDEX IF NOT EXISTS idx_podcasts_created ON public.podcasts USING btree (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_podcasts_org_user ON public.podcasts USING btree (organization_id, user_id);
CREATE INDEX IF NOT EXISTS idx_podcasts_status ON public.podcasts USING btree (status);

-- ============================================================================
-- TABLE: tak_configuration
-- Current rows: 0
-- ============================================================================

-- Indexes for tak_configuration
CREATE INDEX IF NOT EXISTS idx_tak_config_enabled ON public.tak_configuration USING btree (tak_enabled);
CREATE INDEX IF NOT EXISTS idx_tak_config_org ON public.tak_configuration USING btree (organization_id);
CREATE UNIQUE INDEX tak_configuration_organization_id_key ON public.tak_configuration USING btree (organization_id);

-- ============================================================================
-- TABLE: workflows
-- Current rows: 0
-- ============================================================================

-- Indexes for workflows
CREATE INDEX IF NOT EXISTS idx_workflows_created ON public.workflows USING btree (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflows_document_ids ON public.workflows USING gin (document_ids);
CREATE INDEX IF NOT EXISTS idx_workflows_org_user ON public.workflows USING btree (organization_id, user_id);
CREATE INDEX IF NOT EXISTS idx_workflows_type ON public.workflows USING btree (type);
CREATE INDEX IF NOT EXISTS idx_workflows_user ON public.workflows USING btree (user_id);


-- ============================================================================
-- TABLE: google_drive_connections
-- One row per (organization_id, user_id) pair. Stores OAuth tokens for the
-- user's connected Google Drive. Refresh tokens never expire (until revoked);
-- access tokens get refreshed lazily by GoogleDriveClient when expired.
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.google_drive_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL,
    user_id UUID NOT NULL,
    email TEXT,
    display_name TEXT,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    access_token_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (organization_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_gdrive_org_user
    ON public.google_drive_connections USING btree (organization_id, user_id);

-- Flag set when a token refresh fails with invalid_grant (user revoked the
-- app, or the refresh token expired). The UI shows a "reconnect" banner.
ALTER TABLE public.google_drive_connections
    ADD COLUMN IF NOT EXISTS needs_reconnect BOOLEAN NOT NULL DEFAULT FALSE;
