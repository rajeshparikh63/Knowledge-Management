"""
Document Ingestion Service
Orchestrates the complete ingestion pipeline:
1. Upload files to iDrive E2
2. Extract raw content
3. Store in MongoDB
4. Chunk content
5. Store chunks in Pinecone
"""

import io
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from fastapi import UploadFile
from bson import ObjectId
from clients.idrivee2_client import get_idrivee2_client
from clients.mongodb_client import get_mongodb_client
from clients.pinecone_client import get_pinecone_client
from clients.chunker_client import (
    get_chunker_client,
    prepare_content_for_vectorization,
    validate_content_for_embeddings
)
from utils.file_utils import (
    extract_raw_data,
    validate_extracted_content,
    sanitize_filename,
    get_file_size_mb,
    get_file_extension
)
from app.logger import logger
from app.settings import settings


class IngestionService:
    """Service for handling document ingestion pipeline"""

    def __init__(self):
        """Initialize service with all required clients"""
        self.idrivee2_client = get_idrivee2_client()
        self.mongodb_client = get_mongodb_client()
        self.pinecone_client = get_pinecone_client()
        self.chunker_client = get_chunker_client()

    async def ingest_documents(
        self,
        files: List[UploadFile],
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents through the complete pipeline

        Args:
            files: List of uploaded files
            folder_name: Folder name for organization
            user_id: Optional user ID for tracking
            organization_id: Optional organization ID for tracking
            additional_metadata: Optional additional metadata to attach

        Returns:
            Dict with ingestion results and statistics
        """
        logger.info(f"🚀 Starting ingestion of {len(files)} files for folder: {folder_name}")

        results = {
            "folder_name": folder_name,
            "total_files": len(files),
            "successful_files": 0,
            "failed_files": 0,
            "documents": [],
            "errors": [],
            "statistics": {
                "total_chunks": 0,
                "total_size_mb": 0,
                "processing_time": None
            }
        }

        start_time = datetime.utcnow()

        for file in files:
            try:
                # Process single document
                result = await self._process_single_document(
                    file=file,
                    folder_name=folder_name,
                    user_id=user_id,
                    organization_id=organization_id,
                    additional_metadata=additional_metadata
                )

                results["successful_files"] += 1
                results["documents"].append(result)
                results["statistics"]["total_chunks"] += result.get("total_chunks", 0)
                results["statistics"]["total_size_mb"] += result.get("file_size_mb", 0)

                logger.info(f"✅ Successfully processed: {file.filename}")

            except Exception as e:
                results["failed_files"] += 1
                error_info = {
                    "file_name": file.filename,
                    "error": str(e)
                }
                results["errors"].append(error_info)
                logger.error(f"❌ Failed to process {file.filename}: {str(e)}")

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        results["statistics"]["processing_time"] = processing_time

        logger.info(
            f"🎉 Ingestion completed: {results['successful_files']}/{results['total_files']} successful, "
            f"{results['statistics']['total_chunks']} chunks, "
            f"{processing_time:.2f}s"
        )

        return results

    async def _process_single_document(
        self,
        file: UploadFile,
        folder_name: str,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline

        Args:
            file: Uploaded file
            folder_name: Folder name for organization
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional additional metadata

        Returns:
            Dict with processing results
        """
        logger.info(f"📄 Processing document: {file.filename}")

        # Read file content
        file_content = await file.read()
        file_size_mb = get_file_size_mb(file_content)

        # Step 1: Upload to iDrive E2
        logger.info(f"⬆️ Uploading to iDrive E2: {file.filename}")
        safe_filename = sanitize_filename(file.filename)
        file_key = f"{folder_name}/{safe_filename}"

        # Upload file (async operation)
        await self.idrivee2_client.upload_file(
            file_obj=io.BytesIO(file_content),
            object_name=file_key,
            content_type=file.content_type
        )

        # Step 2: Extract raw content (run in thread pool to avoid blocking)
        logger.info(f"🔍 Extracting content from: {file.filename}")
        raw_content = await asyncio.to_thread(extract_raw_data, file_content, file.filename)

        # Validate extracted content
        if not validate_extracted_content(raw_content):
            raise ValueError(f"Extracted content is invalid or empty for: {file.filename}")

        # Validate content for embeddings (check token limits - run in thread pool)
        validation_result = await asyncio.to_thread(validate_content_for_embeddings, raw_content)
        logger.info(
            f"📊 Content validation: {validation_result['token_count']} tokens, "
            f"needs_chunking={validation_result['needs_chunking']}"
        )

        # Step 3: Store in MongoDB
        logger.info(f"💾 Storing document in MongoDB: {file.filename}")
        document_id = await self._store_document_in_mongodb(
            file_name=file.filename,
            folder_name=folder_name,
            raw_content=raw_content,
            file_key=file_key,
            file_size_mb=file_size_mb,
            user_id=user_id,
            organization_id=organization_id,
            additional_metadata=additional_metadata
        )

        # Step 4: Prepare content for vectorization (pre-chunking if > 200k tokens)
        logger.info(f"📦 Preparing content for vectorization: {file.filename}")
        base_metadata = {
            "document_id": document_id,
            "file_name": file.filename,
            "folder_name": folder_name,
            "file_key": file_key,
            "user_id": user_id,
            # Note: organization_id is used as namespace, not metadata
            **(additional_metadata or {})
        }

        # This handles token-based pre-chunking if content > 200k tokens (run in thread pool)
        prepared_documents = await asyncio.to_thread(
            prepare_content_for_vectorization,
            content=raw_content,
            metadata=base_metadata
        )

        # Step 5: Apply semantic chunking to each prepared document and store in Pinecone
        logger.info(f"✂️ Semantic chunking and storing in Pinecone: {file.filename}")
        total_chunks = 0

        for pre_chunk_doc in prepared_documents:
            pre_chunk_content = pre_chunk_doc["content"]
            pre_chunk_metadata = pre_chunk_doc["metadata"]

            # Apply semantic chunking to this pre-chunk (run in thread pool)
            chunks = await asyncio.to_thread(
                self.chunker_client.chunk_with_metadata,
                text=pre_chunk_content,
                base_metadata=pre_chunk_metadata,
                chunker_type="default"
            )

            # Store chunks in Pinecone (use organization_id as namespace)
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]

            # Generate unique IDs for chunks
            pre_chunk_index = pre_chunk_metadata.get("pre_chunk_index", 0)
            ids = [
                f"{document_id}_pre{pre_chunk_index}_chunk{i}"
                for i in range(len(chunks))
            ]

            # Add to Pinecone in thread pool (embedding + upload is blocking)
            await asyncio.to_thread(
                self.pinecone_client.add_documents,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                namespace=organization_id  # Use organization_id as namespace for multi-tenancy
            )

            total_chunks += len(chunks)

        logger.info(f"✅ Stored {total_chunks} chunks in Pinecone for: {file.filename}")

        return {
            "document_id": document_id,
            "file_name": file.filename,
            "folder_name": folder_name,
            "file_key": file_key,
            "file_size_mb": file_size_mb,
            "total_chunks": total_chunks,
            "token_count": validation_result['token_count']
        }

    async def _store_document_in_mongodb(
        self,
        file_name: str,
        folder_name: str,
        raw_content: str,
        file_key: str,
        file_size_mb: float,
        user_id: str = None,
        organization_id: str = None,
        additional_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store document in MongoDB

        Args:
            file_name: Original file name
            folder_name: Folder name
            raw_content: Extracted raw content
            file_key: iDrive E2 object key/path (not URL, since bucket is private)
            file_size_mb: File size in MB
            user_id: Optional user ID
            organization_id: Optional organization ID
            additional_metadata: Optional additional metadata

        Returns:
            Document ID
        """
        # Convert user_id and organization_id to ObjectId
        user_object_id = ObjectId(user_id) if user_id else None
        organization_object_id = ObjectId(organization_id) if organization_id else None

        document = {
            "file_name": file_name,
            "folder_name": folder_name,
            "raw_content": raw_content,
            "file_key": file_key,
            "file_size_mb": file_size_mb,
            "file_extension": get_file_extension(file_name),
            "user_id": user_object_id,
            "organization_id": organization_object_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            **(additional_metadata or {})
        }

        result = await self.mongodb_client.async_insert_document(
            collection="documents",
            document=document
        )

        return str(result)

    async def delete_document(
        self,
        document_id: str,
        delete_from_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Delete document and its chunks from all systems

        Args:
            document_id: MongoDB document ID (string that will be converted to ObjectId)
            delete_from_storage: Whether to delete from iDrive E2

        Returns:
            Dict with deletion results
        """
        logger.info(f"🗑️ Deleting document: {document_id}")

        # Convert document_id to ObjectId for MongoDB query
        doc_object_id = ObjectId(document_id)

        # Get document from MongoDB (async)
        document = await self.mongodb_client.async_find_document(
            collection="documents",
            query={"_id": doc_object_id}
        )

        if not document:
            raise ValueError(f"Document not found: {document_id}")

        # Delete from iDrive E2 if requested (async)
        if delete_from_storage and document.get("file_key"):
            try:
                await self.idrivee2_client.delete_file(document["file_key"])
                logger.info(f"✅ Deleted from iDrive E2: {document['file_key']}")
            except Exception as e:
                logger.warning(f"Failed to delete from iDrive E2: {str(e)}")

        # Delete chunks from Pinecone (use organization_id as namespace)
        try:
            # organization_id is stored as ObjectId in MongoDB, convert to string for Pinecone namespace
            organization_id = str(document.get("organization_id")) if document.get("organization_id") else None
            # Delete all chunks for this document (run in thread pool)
            await asyncio.to_thread(
                self.pinecone_client.delete_documents,
                filter={"document_id": document_id},
                namespace=organization_id
            )
            logger.info(f"✅ Deleted chunks from Pinecone for document: {document_id} (namespace: {organization_id})")
        except Exception as e:
            logger.warning(f"Failed to delete from Pinecone: {str(e)}")

        # Delete from MongoDB (async)
        await self.mongodb_client.async_delete_document(
            collection="documents",
            query={"_id": doc_object_id}
        )

        logger.info(f"✅ Document deleted: {document_id}")

        return {
            "document_id": document_id,
            "deleted": True
        }

    async def _convert_file_key_to_url(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert file_key to presigned URL in document (async)

        Args:
            document: Document dict with file_key

        Returns:
            Document dict with file_url instead of file_key
        """
        if document and document.get("file_key"):
            try:
                # Generate presigned URL (async, expiration from settings, default 7 days)
                file_url = await self.idrivee2_client.generate_presigned_url(
                    object_name=document["file_key"],
                    expiration=settings.PRESIGNED_URL_EXPIRATION
                )
                # Replace file_key with file_url
                document["file_url"] = file_url
                # Remove file_key from response
                document.pop("file_key", None)
            except Exception as e:
                logger.warning(f"Failed to generate presigned URL for {document.get('file_key')}: {str(e)}")
                # Keep file_key if URL generation fails
        return document

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get document from MongoDB and convert file_key to presigned URL (async)

        Args:
            document_id: MongoDB document ID (string that will be converted to ObjectId)

        Returns:
            Document dict with fresh presigned URL
        """
        # Convert document_id string to ObjectId for MongoDB query (async)
        document = await self.mongodb_client.async_find_document(
            collection="documents",
            query={"_id": ObjectId(document_id)}
        )

        if document:
            # Convert ObjectId to string for JSON serialization
            document["_id"] = str(document["_id"])
            if document.get("user_id"):
                document["user_id"] = str(document["user_id"])
            if document.get("organization_id"):
                document["organization_id"] = str(document["organization_id"])

            document = await self._convert_file_key_to_url(document)

        return document

    async def list_documents(
        self,
        folder_name: str = None,
        user_id: str = None,
        organization_id: str = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents from MongoDB with optional filters and convert file_keys to presigned URLs (async)

        Args:
            folder_name: Optional folder name filter (string)
            user_id: Optional user ID filter (string that will be converted to ObjectId)
            organization_id: Optional organization ID filter (string that will be converted to ObjectId)
            limit: Maximum number of documents to return
            skip: Number of documents to skip

        Returns:
            List of documents with fresh presigned URLs
        """
        filter_query = {}

        if folder_name:
            filter_query["folder_name"] = folder_name

        if user_id:
            filter_query["user_id"] = ObjectId(user_id)

        if organization_id:
            filter_query["organization_id"] = ObjectId(organization_id)

        documents = await self.mongodb_client.async_find_documents(
            collection="documents",
            query=filter_query,
            limit=limit,
            skip=skip
        )

        # Convert ObjectIds to strings and file_key to presigned URL for each document (async)
        documents_with_urls = []
        for doc in documents:
            # Convert ObjectId to string for JSON serialization
            doc["_id"] = str(doc["_id"])
            if doc.get("user_id"):
                doc["user_id"] = str(doc["user_id"])
            if doc.get("organization_id"):
                doc["organization_id"] = str(doc["organization_id"])

            doc = await self._convert_file_key_to_url(doc)
            documents_with_urls.append(doc)

        return documents_with_urls


# Singleton instance
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    """
    Get or create IngestionService singleton instance

    Returns:
        IngestionService instance
    """
    global _ingestion_service

    if _ingestion_service is None:
        _ingestion_service = IngestionService()

    return _ingestion_service
