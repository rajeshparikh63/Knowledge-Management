"""
Upload Router - Document ingestion endpoints
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from bson import ObjectId
from services.ingestion_service import get_ingestion_service
from clients.mongodb_client import get_mongodb_client
from app.logger import logger


router = APIRouter(prefix="/upload", tags=["upload"])


async def process_ingestion_task(
    task_id: str,
    file_data: List[Dict[str, Any]],
    folder_name: str,
    user_id: Optional[str],
    organization_id: Optional[str]
):
    """
    Background task for document ingestion

    Args:
        task_id: Task ID for tracking
        file_data: List of dicts with 'content', 'filename', 'content_type'
        folder_name: Folder name for organization
        user_id: Optional user ID
        organization_id: Optional organization ID
    """
    mongodb_client = get_mongodb_client()
    ingestion_service = get_ingestion_service()

    try:
        # Update task status to processing
        await mongodb_client.async_update_document(
            collection="ingestion_tasks",
            query={"_id": ObjectId(task_id)},
            update={
                "$set": {
                    "status": "processing",
                    "updated_at": datetime.utcnow()
                }
            }
        )

        logger.info(f"🚀 Background task {task_id} started processing {len(file_data)} files")

        # Reconstruct UploadFile-like objects from stored data
        from io import BytesIO
        from fastapi import UploadFile

        upload_files = []
        for file_info in file_data:
            file_obj = BytesIO(file_info["content"])
            upload_file = UploadFile(
                file=file_obj,
                filename=file_info["filename"],
                headers={"content-type": file_info["content_type"]}
            )
            upload_files.append(upload_file)

        # Run ingestion
        result = await ingestion_service.ingest_documents(
            files=upload_files,
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id
        )

        # Update task status to completed
        await mongodb_client.async_update_document(
            collection="ingestion_tasks",
            query={"_id": ObjectId(task_id)},
            update={
                "$set": {
                    "status": "completed",
                    "result": result,
                    "updated_at": datetime.utcnow(),
                    "completed_at": datetime.utcnow()
                }
            }
        )

        logger.info(f"✅ Background task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Background task {task_id} failed: {str(e)}")

        # Update task status to failed
        await mongodb_client.async_update_document(
            collection="ingestion_tasks",
            query={"_id": ObjectId(task_id)},
            update={
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.utcnow(),
                    "failed_at": datetime.utcnow()
                }
            }
        )


@router.post("/documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple files to upload"),
    folder_name: str = Form(..., description="Folder name for organization"),
    user_id: Optional[str] = Form(None, description="Optional user ID"),
    organization_id: Optional[str] = Form(None, description="Optional organization ID")
):
    """
    Upload multiple documents for ingestion (background processing)

    This endpoint:
    1. Validates input and creates a task record
    2. Returns immediately with task_id
    3. Processes ingestion in background:
       - Uploads files to iDrive E2
       - Extracts raw content
       - Stores in MongoDB
       - Chunks content semantically
       - Stores chunks in Pinecone

    Args:
        background_tasks: FastAPI background tasks
        files: List of files to upload
        folder_name: Folder name for organization and filtering
        user_id: Optional user ID for tracking
        organization_id: Optional organization ID for tracking

    Returns:
        Task ID for tracking ingestion status
    """
    try:
        # Validate input
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if not folder_name or not folder_name.strip():
            raise HTTPException(status_code=400, detail="Folder name is required")

        # Validate ObjectIds
        if user_id and not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}")

        if organization_id and not ObjectId.is_valid(organization_id):
            raise HTTPException(status_code=400, detail=f"Invalid organization_id format: {organization_id}")

        logger.info(f"📤 Upload request: {len(files)} files, folder={folder_name}")

        # Read file contents into memory (UploadFile streams won't be available in background task)
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append({
                "content": content,
                "filename": file.filename,
                "content_type": file.content_type
            })
            await file.seek(0)  # Reset file pointer

        # Create task record in MongoDB
        mongodb_client = get_mongodb_client()
        task_document = {
            "status": "queued",
            "folder_name": folder_name.strip(),
            "user_id": ObjectId(user_id) if user_id else None,
            "organization_id": ObjectId(organization_id) if organization_id else None,
            "total_files": len(files),
            "file_names": [f["filename"] for f in file_data],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        task_id = await mongodb_client.async_insert_document(
            collection="ingestion_tasks",
            document=task_document
        )

        # Add background task
        background_tasks.add_task(
            process_ingestion_task,
            task_id=str(task_id),
            file_data=file_data,
            folder_name=folder_name.strip(),
            user_id=user_id,
            organization_id=organization_id
        )

        logger.info(f"✅ Task {task_id} queued for background processing")

        return {
            "success": True,
            "message": f"Ingestion task queued for {len(files)} files",
            "data": {
                "task_id": str(task_id),
                "status": "queued",
                "total_files": len(files),
                "folder_name": folder_name.strip()
            }
        }

    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    Get ingestion task status

    Args:
        task_id: Task ID returned from upload endpoint

    Returns:
        Task status and result (if completed)
    """
    try:
        # Validate task_id
        if not ObjectId.is_valid(task_id):
            raise HTTPException(status_code=400, detail=f"Invalid task_id format: {task_id}")

        mongodb_client = get_mongodb_client()
        task = await mongodb_client.async_find_document(
            collection="ingestion_tasks",
            query={"_id": ObjectId(task_id)}
        )

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Convert ObjectId to string for JSON serialization
        task["_id"] = str(task["_id"])
        if task.get("user_id"):
            task["user_id"] = str(task["user_id"])
        if task.get("organization_id"):
            task["organization_id"] = str(task["organization_id"])

        return {
            "success": True,
            "data": task
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get task status failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get document by ID

    Args:
        document_id: MongoDB document ID

    Returns:
        Document data
    """
    try:
        # Validate document_id
        if not ObjectId.is_valid(document_id):
            raise HTTPException(status_code=400, detail=f"Invalid document_id format: {document_id}")

        ingestion_service = get_ingestion_service()
        document = await ingestion_service.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "success": True,
            "data": document
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.get("/documents")
async def list_documents(
    folder_name: Optional[str] = None,
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    limit: int = 100,
    skip: int = 0
):
    """
    List documents with optional filters

    Args:
        folder_name: Optional folder name filter
        user_id: Optional user ID filter
        organization_id: Optional organization ID filter
        limit: Maximum number of documents to return (default: 100)
        skip: Number of documents to skip (default: 0)

    Returns:
        List of documents
    """
    try:
        # Validate ObjectIds
        if user_id and not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail=f"Invalid user_id format: {user_id}")

        if organization_id and not ObjectId.is_valid(organization_id):
            raise HTTPException(status_code=400, detail=f"Invalid organization_id format: {organization_id}")

        ingestion_service = get_ingestion_service()
        documents = await ingestion_service.list_documents(
            folder_name=folder_name,
            user_id=user_id,
            organization_id=organization_id,
            limit=limit,
            skip=skip
        )

        return {
            "success": True,
            "data": documents,
            "count": len(documents)
        }

    except Exception as e:
        logger.error(f"❌ List documents failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    delete_from_storage: bool = True
):
    """
    Delete document and its chunks from all systems

    Args:
        document_id: MongoDB document ID
        delete_from_storage: Whether to delete from iDrive E2 (default: True)

    Returns:
        Deletion result
    """
    try:
        # Validate document_id
        if not ObjectId.is_valid(document_id):
            raise HTTPException(status_code=400, detail=f"Invalid document_id format: {document_id}")

        ingestion_service = get_ingestion_service()
        result = await ingestion_service.delete_document(
            document_id=document_id,
            delete_from_storage=delete_from_storage
        )

        return {
            "success": True,
            "message": "Document deleted successfully",
            "data": result
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Delete document failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
