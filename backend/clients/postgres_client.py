"""
PostgreSQL Client
Database operations for documents, podcasts, workflows, and TAK configuration
Replaces MongoDB for non-agent collections
"""

import asyncio
import asyncpg
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.settings import settings
from app.logger import logger
import uuid
import json


class PostgresClient:
    """Async client for PostgreSQL operations"""

    def __init__(self):
        """Initialize PostgreSQL client"""
        self.connection_string = settings.POSTGRES_URL

        if not self.connection_string:
            raise ValueError("PostgreSQL connection string not configured")

        self.pool: Optional[asyncpg.Pool] = None
        logger.info("PostgreSQL client initialized (pool will be created on first use)")

    @staticmethod
    def _normalize_document_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document dict to match expected field names.
        PostgreSQL column 'filename' -> Python key 'file_name'
        Convert UUID objects to strings for JSON serialization
        """
        if doc and 'filename' in doc:
            doc['file_name'] = doc.pop('filename')

        # Convert UUID objects to strings
        import uuid as uuid_module
        for key, value in doc.items():
            if isinstance(value, uuid_module.UUID):
                doc[key] = str(value)

        # asyncpg returns JSONB columns as strings (no codec registered).
        # Parse them so consumers get real dicts (e.g. metadata.source,
        # processing_progress.current).
        for json_key in ("metadata", "processing_progress"):
            v = doc.get(json_key)
            if isinstance(v, str):
                try:
                    doc[json_key] = json.loads(v)
                except (ValueError, TypeError):
                    pass

        return doc

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool — loop-safe across Celery tasks.

        Celery ``--pool=solo`` reuses one Python process across tasks. Each
        task calls ``asyncio.run()`` which spins up a fresh event loop and
        tears it down at the end. The asyncpg pool was bound to whichever
        loop created it; on the *next* task the loop is dead and any pool
        op throws ``RuntimeError: Event loop is closed`` /
        ``InterfaceError: another operation is in progress``.

        Fix: detect when the existing pool's loop differs from the current
        one (or is closed) and rebuild the pool for the current loop.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self.pool is not None and current_loop is not None:
            pool_loop = getattr(self.pool, "_loop", None)
            if pool_loop is None or pool_loop is not current_loop or pool_loop.is_closed():
                logger.warning(
                    "🔄 Postgres pool was bound to a different/closed event loop; "
                    "rebuilding for current task"
                )
                # Don't await close() — the old loop is dead. Let GC handle it.
                self.pool = None

        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )
            logger.info("✅ PostgreSQL connection pool created")
        return self.pool

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ PostgreSQL connection pool closed")

    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================

    async def insert_document(self, document: Dict[str, Any]) -> str:
        """
        Insert a document into documents table

        Args:
            document: Document data (keys match table columns)

        Returns:
            str: Document ID (UUID)
        """
        pool = await self.get_pool()

        # Generate ID if not provided
        doc_id = document.get('id') or str(uuid.uuid4())

        # Handle both 'filename' and 'file_name' keys
        filename = document.get('filename') or document.get('file_name')

        # Debug logging
        logger.info(f"📝 Inserting document: filename={filename}, file_key={document.get('file_key')}")

        query = """
            INSERT INTO documents (
                id, organization_id, user_id, filename, file_key, folder_name,
                content_type, file_size_mb, raw_content, status, processing_stage,
                processing_stage_description, error, metadata, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
            RETURNING id
        """

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                uuid.UUID(doc_id),
                uuid.UUID(document['organization_id']),
                uuid.UUID(document['user_id']),
                filename,
                document.get('file_key'),
                document.get('folder_name'),
                document.get('content_type'),
                document.get('file_size_mb'),
                document.get('raw_content'),
                document.get('status', 'pending'),
                document.get('processing_stage'),
                document.get('processing_stage_description'),
                document.get('error'),
                json.dumps(document.get('metadata', {})),  # Convert dict to JSON string
                document.get('created_at', datetime.utcnow()),
                document.get('updated_at', datetime.utcnow())
            )

        logger.info(f"✅ Document inserted: {result}")
        return str(result)

    async def find_document(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a document by ID

        Args:
            organization_id: Organization UUID (optional for additional filtering)
            user_id: User UUID (optional for additional filtering)
            document_id: Document UUID (required)

        Returns:
            Document dict or None if not found
        """
        pool = await self.get_pool()

        # Build query dynamically based on provided filters
        conditions = ["id = $1"]
        params = [uuid.UUID(document_id)]
        param_index = 2

        if organization_id:
            conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))

        query = f"""
            SELECT * FROM documents
            WHERE {' AND '.join(conditions)}
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        return self._normalize_document_dict(dict(row)) if row else None

    async def find_documents(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        folder_name: Optional[str] = None,
        status: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents with filters

        Args:
            organization_id: Organization UUID (optional)
            user_id: User UUID (optional)
            folder_name: Optional folder filter
            status: Optional status filter
            filters: Optional dict of filters (takes precedence over individual params)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of document dicts
        """
        pool = await self.get_pool()

        # Build filters from dict or individual params
        if filters:
            organization_id = filters.get('organization_id', organization_id)
            user_id = filters.get('user_id', user_id)
            folder_name = filters.get('folder_name', folder_name)
            status = filters.get('status', status)

        conditions = []
        params: List[Any] = []
        param_index = 1

        if organization_id:
            conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))
            param_index += 1

        if folder_name:
            conditions.append(f"folder_name = ${param_index}")
            params.append(folder_name)
            param_index += 1

        if status:
            conditions.append(f"status = ${param_index}")
            params.append(status)
            param_index += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT * FROM documents
            {where_clause}
            ORDER BY created_at DESC
        """

        if limit:
            query += f" LIMIT ${param_index}"
            params.append(limit)
            param_index += 1

        if offset:
            query += f" OFFSET ${param_index}"
            params.append(offset)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        logger.info(f"✅ Found {len(rows)} documents")
        return [self._normalize_document_dict(dict(row)) for row in rows]

    async def update_document(
        self,
        organization_id: str,
        user_id: str,
        document_id: str,
        updates: Dict[str, Any]
    ) -> int:
        """
        Update a document

        Args:
            organization_id: Organization UUID
            user_id: User UUID
            document_id: Document UUID
            updates: Fields to update

        Returns:
            Number of rows updated
        """
        pool = await self.get_pool()

        # Build SET clause dynamically
        set_clauses = []
        params: List[Any] = []
        param_index = 1

        for key, value in updates.items():
            set_clauses.append(f"{key} = ${param_index}")
            # Convert dict to JSON string for JSONB columns (like 'metadata')
            if isinstance(value, dict):
                params.append(json.dumps(value))
            else:
                params.append(value)
            param_index += 1

        # Add updated_at
        set_clauses.append(f"updated_at = ${param_index}")
        params.append(datetime.utcnow())
        param_index += 1

        # Add WHERE conditions
        params.extend([
            uuid.UUID(document_id),
            uuid.UUID(organization_id),
            uuid.UUID(user_id)
        ])

        query = f"""
            UPDATE documents
            SET {', '.join(set_clauses)}
            WHERE id = ${param_index} AND organization_id = ${param_index + 1} AND user_id = ${param_index + 2}
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)

        # Extract number from "UPDATE N"
        count = int(result.split()[-1])
        logger.info(f"✅ Updated {count} document(s)")
        return count

    async def delete_document(
        self,
        organization_id: str,
        user_id: str,
        document_id: str
    ) -> int:
        """
        Delete a document

        Args:
            organization_id: Organization UUID (optional if already in document context)
            user_id: User UUID (optional if already in document context)
            document_id: Document UUID

        Returns:
            Number of rows deleted
        """
        pool = await self.get_pool()

        # Build query dynamically based on provided filters
        conditions = ["id = $1"]
        params = [uuid.UUID(document_id)]
        param_index = 2

        if organization_id:
            conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))

        query = f"""
            DELETE FROM documents
            WHERE {' AND '.join(conditions)}
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)

        count = int(result.split()[-1])
        logger.info(f"✅ Deleted {count} document(s)")
        return count

    async def count_documents(
        self,
        organization_id: str,
        user_id: str,
        status: Optional[str] = None
    ) -> int:
        """
        Count documents

        Args:
            organization_id: Organization UUID
            user_id: User UUID
            status: Optional status filter

        Returns:
            Number of documents
        """
        pool = await self.get_pool()

        conditions = ["organization_id = $1", "user_id = $2"]
        params: List[Any] = [uuid.UUID(organization_id), uuid.UUID(user_id)]

        if status:
            conditions.append("status = $3")
            params.append(status)

        query = f"""
            SELECT COUNT(*) FROM documents
            WHERE {' AND '.join(conditions)}
        """

        async with pool.acquire() as conn:
            count = await conn.fetchval(query, *params)

        return count

    async def distinct_folders(
        self,
        filters: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Get distinct folder names from documents

        Args:
            filters: Optional dict of filters
            organization_id: Optional organization UUID
            user_id: Optional user UUID

        Returns:
            List of unique folder names
        """
        pool = await self.get_pool()

        # Build filters from dict or individual params
        if filters:
            organization_id = filters.get('organization_id', organization_id)
            user_id = filters.get('user_id', user_id)

        conditions = []
        params: List[Any] = []
        param_index = 1

        if organization_id:
            conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))
            param_index += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT DISTINCT folder_name FROM documents
            {where_clause}
            ORDER BY folder_name
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        folders = [row['folder_name'] for row in rows]
        logger.info(f"✅ Found {len(folders)} distinct folders")
        return folders

    # ========================================================================
    # PODCAST OPERATIONS
    # ========================================================================

    async def insert_podcast(self, podcast: Dict[str, Any]) -> str:
        """Insert a podcast"""
        pool = await self.get_pool()
        podcast_id = podcast.get('id') or str(uuid.uuid4())

        query = """
            INSERT INTO podcasts (
                id, organization_id, user_id, title, summary, script,
                audio_file_key, status, error_message, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                uuid.UUID(podcast_id),
                uuid.UUID(podcast['organization_id']),
                uuid.UUID(podcast['user_id']),
                podcast.get('title'),
                podcast.get('summary'),
                json.dumps(podcast.get('script')) if podcast.get('script') else None,  # Convert dict to JSON string
                podcast.get('audio_file_key'),
                podcast.get('status', 'pending'),
                podcast.get('error_message'),
                podcast.get('created_at', datetime.utcnow()),
                podcast.get('updated_at', datetime.utcnow())
            )

        logger.info(f"✅ Podcast inserted: {result}")
        return str(result)

    async def find_podcast(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        podcast_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a podcast by ID

        Args:
            organization_id: Organization UUID (optional for additional filtering)
            user_id: User UUID (optional for additional filtering)
            podcast_id: Podcast UUID (required)

        Returns:
            Podcast dict or None if not found
        """
        pool = await self.get_pool()

        # Build query dynamically based on provided filters
        conditions = ["id = $1"]
        params = [uuid.UUID(podcast_id)]
        param_index = 2

        if organization_id:
            conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))

        query = f"""
            SELECT * FROM podcasts
            WHERE {' AND '.join(conditions)}
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row:
            result = dict(row)
            # Convert UUID objects to strings
            for key, value in result.items():
                if isinstance(value, uuid.UUID):
                    result[key] = str(value)
            return result
        return None

    async def update_podcast(
        self,
        podcast_id: str,
        updates: Dict[str, Any],
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Update a podcast

        Args:
            podcast_id: Podcast UUID (required)
            updates: Dictionary of fields to update
            organization_id: Organization UUID (optional for additional filtering)
            user_id: User UUID (optional for additional filtering)

        Returns:
            Number of rows updated
        """
        pool = await self.get_pool()

        set_clauses = []
        params: List[Any] = []
        param_index = 1

        for key, value in updates.items():
            set_clauses.append(f"{key} = ${param_index}")
            # Convert dict to JSON string for JSONB columns (like 'script')
            if isinstance(value, dict):
                params.append(json.dumps(value))
            else:
                params.append(value)
            param_index += 1

        set_clauses.append(f"updated_at = ${param_index}")
        params.append(datetime.utcnow())
        param_index += 1

        # Build WHERE clause dynamically
        where_conditions = [f"id = ${param_index}"]
        params.append(uuid.UUID(podcast_id))
        param_index += 1

        if organization_id:
            where_conditions.append(f"organization_id = ${param_index}")
            params.append(uuid.UUID(organization_id))
            param_index += 1

        if user_id:
            where_conditions.append(f"user_id = ${param_index}")
            params.append(uuid.UUID(user_id))

        query = f"""
            UPDATE podcasts
            SET {', '.join(set_clauses)}
            WHERE {' AND '.join(where_conditions)}
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)

        count = int(result.split()[-1])
        logger.info(f"✅ Updated {count} podcast(s)")
        return count

    # ========================================================================
    # WORKFLOW OPERATIONS
    # ========================================================================

    async def insert_workflow(self, workflow: Dict[str, Any]) -> str:
        """Insert a workflow (flashcard, mindmap, report, etc.)"""
        pool = await self.get_pool()
        workflow_id = workflow.get('id') or str(uuid.uuid4())

        # Convert document_ids to UUID array
        doc_ids = [uuid.UUID(doc_id) for doc_id in workflow['document_ids']]

        query = """
            INSERT INTO workflows (
                id, organization_id, user_id, type, data, document_ids,
                status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                uuid.UUID(workflow_id),
                uuid.UUID(workflow['organization_id']) if workflow.get('organization_id') else None,
                uuid.UUID(workflow['user_id']),
                workflow['type'],
                json.dumps(workflow['data']),  # Convert dict to JSON string
                doc_ids,
                workflow.get('status', 'pending'),
                workflow.get('created_at', datetime.utcnow()),
                workflow.get('updated_at', datetime.utcnow())
            )

        logger.info(f"✅ Workflow inserted: {result}")
        return str(result)

    async def find_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Find a document by ID"""
        pool = await self.get_pool()

        query = """
            SELECT * FROM documents
            WHERE id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, uuid.UUID(document_id))

        return self._normalize_document_dict(dict(row)) if row else None

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update a workflow"""
        pool = await self.get_pool()

        # Build SET clause dynamically
        set_clauses = []
        params = [uuid.UUID(workflow_id)]
        param_index = 2

        for key, value in updates.items():
            set_clauses.append(f"{key} = ${param_index}")
            # Convert dict to JSON string for JSONB columns (like 'data')
            if isinstance(value, dict):
                params.append(json.dumps(value))
            else:
                params.append(value)
            param_index += 1

        query = f"""
            UPDATE workflows
            SET {', '.join(set_clauses)}
            WHERE id = $1
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)

        return result != "UPDATE 0"

    async def find_workflow_by_id(
        self,
        workflow_id: str,
        workflow_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find a workflow by ID"""
        pool = await self.get_pool()

        conditions = ["id = $1"]
        params = [uuid.UUID(workflow_id)]

        if workflow_type:
            conditions.append("type = $2")
            params.append(workflow_type)

        query = f"""
            SELECT * FROM workflows
            WHERE {' AND '.join(conditions)}
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row:
            result = dict(row)
            # Parse JSON data field
            if result.get('data'):
                result['data'] = json.loads(result['data']) if isinstance(result['data'], str) else result['data']
            # Convert UUID array to string list
            if result.get('document_ids'):
                result['document_ids'] = [str(doc_id) for doc_id in result['document_ids']]
            # Convert UUIDs to strings
            if result.get('id'):
                result['id'] = str(result['id'])
            if result.get('organization_id'):
                result['organization_id'] = str(result['organization_id'])
            if result.get('user_id'):
                result['user_id'] = str(result['user_id'])
            return result

        return None

    async def find_workflow_by_documents(
        self,
        workflow_type: str,
        document_ids: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Find a workflow by document IDs (exact match)"""
        pool = await self.get_pool()

        # Convert to UUID array
        doc_uuids = [uuid.UUID(doc_id) for doc_id in document_ids]

        query = """
            SELECT * FROM workflows
            WHERE type = $1
            AND document_ids = $2
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, workflow_type, doc_uuids)

        if row:
            result = dict(row)
            # Parse JSON data field
            if result.get('data'):
                result['data'] = json.loads(result['data']) if isinstance(result['data'], str) else result['data']
            # Convert UUID array to string list
            if result.get('document_ids'):
                result['document_ids'] = [str(doc_id) for doc_id in result['document_ids']]
            # Convert UUIDs to strings
            if result.get('id'):
                result['id'] = str(result['id'])
            if result.get('organization_id'):
                result['organization_id'] = str(result['organization_id'])
            if result.get('user_id'):
                result['user_id'] = str(result['user_id'])
            return result

        return None

    async def find_workflows_by_user(
        self,
        workflow_type: str,
        user_id: str,
        organization_id: str
    ) -> List[Dict[str, Any]]:
        """Find workflows by user and organization"""
        pool = await self.get_pool()

        query = """
            SELECT * FROM workflows
            WHERE type = $1
            AND user_id = $2
            AND organization_id = $3
            ORDER BY created_at DESC
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                workflow_type,
                uuid.UUID(user_id),
                uuid.UUID(organization_id)
            )

        results = []
        for row in rows:
            result = dict(row)
            # Parse JSON data field
            if result.get('data'):
                result['data'] = json.loads(result['data']) if isinstance(result['data'], str) else result['data']
            # Convert UUID array to string list
            if result.get('document_ids'):
                result['document_ids'] = [str(doc_id) for doc_id in result['document_ids']]
            # Convert UUIDs to strings
            if result.get('id'):
                result['id'] = str(result['id'])
            if result.get('organization_id'):
                result['organization_id'] = str(result['organization_id'])
            if result.get('user_id'):
                result['user_id'] = str(result['user_id'])
            results.append(result)

        return results

    # ========================================================================
    # TAK CONFIGURATION OPERATIONS
    # ========================================================================

    async def upsert_tak_config(self, config: Dict[str, Any]) -> str:
        """Insert or update TAK configuration for an organization"""
        pool = await self.get_pool()

        query = """
            INSERT INTO tak_configuration (
                id, organization_id, tak_enabled, tak_host, tak_port,
                tak_username, tak_password, agent_callsign, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (organization_id) DO UPDATE SET
                tak_enabled = EXCLUDED.tak_enabled,
                tak_host = EXCLUDED.tak_host,
                tak_port = EXCLUDED.tak_port,
                tak_username = EXCLUDED.tak_username,
                tak_password = EXCLUDED.tak_password,
                agent_callsign = EXCLUDED.agent_callsign,
                updated_at = EXCLUDED.updated_at
            RETURNING id
        """

        config_id = config.get('id') or str(uuid.uuid4())

        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                uuid.UUID(config_id),
                uuid.UUID(config['organization_id']),
                config.get('tak_enabled', False),
                config['tak_host'],
                config['tak_port'],
                config.get('tak_username'),
                config.get('tak_password'),
                config.get('agent_callsign', 'SoldierIQ-Agent'),
                config.get('created_at', datetime.utcnow()),
                datetime.utcnow()
            )

        logger.info(f"✅ TAK config upserted: {result}")
        return str(result)

    async def get_tak_config(self, organization_id: str) -> Optional[Dict[str, Any]]:
        """Get TAK configuration for an organization"""
        pool = await self.get_pool()

        query = """
            SELECT * FROM tak_configuration
            WHERE organization_id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, uuid.UUID(organization_id))

        return dict(row) if row else None

    async def delete_tak_config(self, organization_id: str) -> bool:
        """Delete TAK configuration for an organization"""
        pool = await self.get_pool()

        query = """
            DELETE FROM tak_configuration
            WHERE organization_id = $1
        """

        async with pool.acquire() as conn:
            result = await conn.execute(query, uuid.UUID(organization_id))

        count = int(result.split()[-1])
        logger.info(f"✅ Deleted {count} TAK config(s)")
        return count > 0

    # ----------------------------------------------------------------------
    # Google Drive connection tokens (one row per (org, user) pair)
    # ----------------------------------------------------------------------

    async def upsert_google_drive_connection(
        self,
        *,
        organization_id: str,
        user_id: str,
        email: Optional[str],
        display_name: Optional[str],
        access_token: str,
        refresh_token: str,
        access_token_expires_at: datetime,
    ) -> str:
        """Insert or update a Google Drive OAuth connection for (org, user).

        Returns the row id. Updates refresh_token on conflict because Google
        sometimes re-issues it when you re-consent.
        """
        pool = await self.get_pool()

        query = """
            INSERT INTO google_drive_connections (
                id, organization_id, user_id, email, display_name,
                access_token, refresh_token, access_token_expires_at,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (organization_id, user_id) DO UPDATE SET
                email = EXCLUDED.email,
                display_name = EXCLUDED.display_name,
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                access_token_expires_at = EXCLUDED.access_token_expires_at,
                updated_at = EXCLUDED.updated_at
            RETURNING id
        """

        now = datetime.utcnow()
        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                query,
                uuid.uuid4(),
                uuid.UUID(organization_id),
                uuid.UUID(user_id),
                email,
                display_name,
                access_token,
                refresh_token,
                access_token_expires_at,
                now,
                now,
            )
        logger.info(f"✅ Google Drive connection upserted: {row_id} for user={user_id[:8]}…")
        return str(row_id)

    async def get_google_drive_connection(
        self, organization_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the connection row for (org, user) or None."""
        pool = await self.get_pool()
        query = """
            SELECT * FROM google_drive_connections
            WHERE organization_id = $1 AND user_id = $2
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                query, uuid.UUID(organization_id), uuid.UUID(user_id)
            )
        return dict(row) if row else None

    async def set_google_drive_needs_reconnect(
        self, organization_id: str, user_id: str, value: bool
    ) -> None:
        """Flag (or clear) the needs_reconnect state for a Drive connection.

        Set True when a token refresh fails with invalid_grant; cleared back
        to False when the user reconnects. Wrapped defensively by callers in
        case the column doesn't exist yet (pre-migration).
        """
        pool = await self.get_pool()
        query = """
            UPDATE google_drive_connections
            SET needs_reconnect = $1, updated_at = NOW()
            WHERE organization_id = $2 AND user_id = $3
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query, value, uuid.UUID(organization_id), uuid.UUID(user_id)
            )

    async def update_google_drive_tokens(
        self,
        *,
        organization_id: str,
        user_id: str,
        access_token: str,
        access_token_expires_at: datetime,
        refresh_token: Optional[str] = None,
    ) -> None:
        """Persist a refreshed access token (called by DriveClient after refresh)."""
        pool = await self.get_pool()
        if refresh_token is not None:
            query = """
                UPDATE google_drive_connections
                SET access_token = $1, access_token_expires_at = $2,
                    refresh_token = $3, updated_at = NOW()
                WHERE organization_id = $4 AND user_id = $5
            """
            args = (access_token, access_token_expires_at, refresh_token,
                    uuid.UUID(organization_id), uuid.UUID(user_id))
        else:
            query = """
                UPDATE google_drive_connections
                SET access_token = $1, access_token_expires_at = $2,
                    updated_at = NOW()
                WHERE organization_id = $3 AND user_id = $4
            """
            args = (access_token, access_token_expires_at,
                    uuid.UUID(organization_id), uuid.UUID(user_id))
        async with pool.acquire() as conn:
            await conn.execute(query, *args)

    async def list_ingested_drive_file_ids(
        self, organization_id: str, user_id: str
    ) -> set:
        """Return the set of Drive file ids this user has already ingested.

        Used by the discovery task to skip files that are already in our system
        — so re-running sync doesn't create duplicate document rows.

        We store the Drive file id in the document's metadata jsonb under the
        key `drive_file_id` (set by the Drive router during ingest fan-out).
        """
        pool = await self.get_pool()
        query = """
            SELECT metadata->>'drive_file_id' AS drive_file_id
            FROM documents
            WHERE organization_id = $1
              AND user_id = $2
              AND metadata->>'source' = 'google_drive'
              AND metadata->>'drive_file_id' IS NOT NULL
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query, uuid.UUID(organization_id), uuid.UUID(user_id)
            )
        return {r["drive_file_id"] for r in rows if r["drive_file_id"]}

    async def list_ingested_sharepoint_item_ids(
        self, organization_id: str, user_id: str
    ) -> set:
        """Return the set of SharePoint drive-item ids this user already ingested.

        Mirrors list_ingested_drive_file_ids — dedup so re-running discovery
        doesn't create duplicate document rows. The item id is stored under
        metadata.sharepoint_item_id by the SharePoint discovery task.
        """
        pool = await self.get_pool()
        query = """
            SELECT metadata->>'sharepoint_item_id' AS item_id
            FROM documents
            WHERE organization_id = $1
              AND user_id = $2
              AND metadata->>'source' = 'sharepoint'
              AND metadata->>'sharepoint_item_id' IS NOT NULL
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                query, uuid.UUID(organization_id), uuid.UUID(user_id)
            )
        return {r["item_id"] for r in rows if r["item_id"]}

    async def delete_google_drive_connection(
        self, organization_id: str, user_id: str
    ) -> bool:
        """Delete the connection (revoke). Returns True if a row was deleted."""
        pool = await self.get_pool()
        query = """
            DELETE FROM google_drive_connections
            WHERE organization_id = $1 AND user_id = $2
        """
        async with pool.acquire() as conn:
            result = await conn.execute(
                query, uuid.UUID(organization_id), uuid.UUID(user_id)
            )
        count = int(result.split()[-1])
        return count > 0


# Singleton instance
_postgres_client: Optional[PostgresClient] = None


def get_postgres_client() -> PostgresClient:
    """
    Get or create PostgresClient singleton instance

    Returns:
        PostgresClient: Singleton client instance
    """
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresClient()
    return _postgres_client
