"""
Chat API Endpoints
Handles conversational AI interactions with knowledge base using SSE streaming
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.chat import create_chat_agent
from services.agent_streaming import stream_agent_response
from utils.streaming import create_sse_event_stream
from app.logger import logger
from auth.keycloak_auth import get_current_user_keycloak
from clients.mongodb_client import get_mongodb_client
import uuid
import json

router = APIRouter(tags=["chat"])


class TAKCredentials(BaseModel):
    """TAK server credentials from frontend"""
    tak_host: str
    tak_port: int
    tak_username: str
    tak_password: str
    agent_callsign: Optional[str] = "SoldierIQ-Agent"


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Note: `file_names` was removed — filenames were never authoritative and the
    LLM was told to ignore them anyway. The agent learns what's in each
    selected doc from the actual retrieved chunks via document_ids scoping.
    """
    message: str
    session_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    model: Optional[str] = "anthropic/claude-sonnet-4.5"  # Use "functiongemma:270m" for hybrid mode
    tak_credentials: Optional[TAKCredentials] = None  # Optional TAK integration
    # user_id and organization_id are extracted from JWT token by backend


@router.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user_keycloak)):
    """
    Send a message to the chat agent and get a streaming response

    Args:
        request: ChatRequest with message and optional parameters

    Returns:
        StreamingResponse with SSE events
    """
    try:
        # Extract user_id and organization_id from JWT token
        user_id = current_user.get("id")
        organization_id = current_user.get("organization_id")

        # Validate input
        if not request.message or request.message.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Message is required and cannot be empty"
            )

        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(
            f"Chat request: session={session_id}, "
            f"message='{request.message[:50]}...'"
        )

        # Create chat agent
        agent = await create_chat_agent(
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id,
            document_ids=request.document_ids,
            model=request.model,
            tak_credentials=request.tak_credentials
        )

        # SSE headers
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Vary": "Accept",
        }

        # Stream response
        async def sse_stream():
            events = stream_agent_response(request.message, agent)
            async for chunk in create_sse_event_stream(events):
                yield chunk

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers=headers
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.get("/chat/sessions")
async def list_chat_sessions(
    current_user: dict = Depends(get_current_user_keycloak),
    limit: int = 50,
    skip: int = 0
) -> Dict[str, Any]:
    """
    List all chat sessions for the current user

    Returns sessions sorted by most recently updated first.
    Each session includes:
    - session_id: Unique identifier
    - name: Session name or preview of first message
    - created_at: When session was created
    - updated_at: When session was last updated
    - message_preview: Preview of first message

    Note: agent_sessions collection only stores user_id (not organization_id).
    Organization-level filtering is handled by user authentication.

    Args:
        current_user: Authenticated user from token
        limit: Maximum number of sessions to return (default: 50)
        skip: Number of sessions to skip for pagination (default: 0)

    Returns:
        Dict with success status and list of sessions
    """
    try:
        # Extract user_id from JWT token
        user_id = current_user.get("id")

        logger.info(f"📋 Listing chat sessions for user: {user_id}")

        mongodb = get_mongodb_client()

        # Query by user_id only (agent_sessions collection doesn't store organization_id)
        query = {"user_id": user_id}

        # Get sessions with limited projection for performance
        sessions = await mongodb.async_find_documents(
            collection="agent_sessions",
            query=query,
            projection={
                "session_id": 1,
                "session_data": 1,
                "created_at": 1,
                "updated_at": 1,
                "runs": {"$slice": 1}  # Just first run for message preview
            },
            limit=limit,
            skip=skip
        )

        logger.info(f"Retrieved {len(sessions)} raw sessions from MongoDB")

        # Sort by updated_at descending (most recent first)
        sessions_sorted = sorted(
            sessions,
            key=lambda x: x.get("updated_at", 0),
            reverse=True
        )

        logger.info(f"Sorted {len(sessions_sorted)} sessions")

        # Transform for frontend
        result = []
        for session in sessions_sorted:
            # Get first message as preview
            message_preview = ""
            first_user_message = ""

            if session.get("runs") and len(session["runs"]) > 0:
                messages = session["runs"][0].get("messages", [])
                for msg in messages:
                    if msg.get("role") == "user":
                        first_user_message = msg.get("content", "")
                        message_preview = first_user_message[:100]
                        if len(first_user_message) > 100:
                            message_preview += "..."
                        break

            # Use session name if available, otherwise use first message
            session_name = session.get("session_data", {}).get("session_name")
            if not session_name:
                session_name = first_user_message[:50] if first_user_message else "New Chat"
                if first_user_message and len(first_user_message) > 50:
                    session_name += "..."

            result.append({
                "session_id": session["session_id"],
                "name": session_name,
                "message_preview": message_preview,
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at")
            })

        logger.info(f"✅ Found {len(result)} chat sessions")

        return {
            "success": True,
            "sessions": result,
            "count": len(result)
        }

    except Exception as e:
        logger.error(f"❌ Failed to list chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_keycloak)
) -> Dict[str, Any]:
    """
    Get full chat history for a specific session

    Returns all messages in the session in chronological order.
    Only returns sessions owned by the authenticated user.

    Note: agent_sessions collection only stores user_id (not organization_id).

    Args:
        session_id: The unique session identifier
        current_user: Authenticated user from token

    Returns:
        Dict with session details and full message history
    """
    try:
        # Extract user_id from JWT token
        user_id = current_user.get("id")

        logger.info(f"📖 Getting chat session: {session_id} for user: {user_id}")

        mongodb = get_mongodb_client()

        # Security: Ensure user owns this session (filter by user_id only)
        query = {
            "session_id": session_id,
            "user_id": user_id
        }

        session = await mongodb.async_find_document(
            collection="agent_sessions",
            query=query
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or you don't have access to it"
            )

        # Extract clean message history (avoiding duplicates from from_history)
        messages = []
        for run_idx, run in enumerate(session.get("runs", [])):
            run_messages = run.get("messages", [])
            logger.debug(f"Processing run {run_idx}: {len(run_messages)} messages")

            # Build a map of tool call IDs to their results from role=tool messages
            tool_results_map = {}
            for message in run_messages:
                if message.get("role") == "tool":
                    tool_call_id = message.get("tool_call_id")
                    tool_name = message.get("tool_name")
                    content = message.get("content")

                    if tool_call_id and tool_name and content:
                        tool_results_map[tool_call_id] = {
                            "tool_name": tool_name,
                            "result": content
                        }

            # Extract user and assistant messages
            for msg_idx, message in enumerate(run_messages):
                # Only include new messages, not historical ones
                if message.get("from_history", False):
                    continue

                # Skip tool response messages
                message_role = message.get("role")
                if message_role == "tool":
                    continue

                # Skip messages without content (e.g., tool call requests)
                if "content" not in message:
                    continue

                # Skip system messages (AI instructions/prompts)
                if message_role == "system":
                    continue

                # Only include user and assistant messages
                if message_role in ["user", "assistant"]:
                    message_data = {
                        "role": message_role,
                        "content": message["content"],
                        "created_at": message.get("created_at"),
                        "model": message.get("model") if message_role == "assistant" else None
                    }

                    # If this is an assistant message with content, find associated tool calls
                    if message_role == "assistant":
                        sources = []

                        # Look backward to find the most recent assistant message with tool_calls
                        for prev_idx in range(msg_idx - 1, -1, -1):
                            prev_msg = run_messages[prev_idx]
                            if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                                tool_calls = prev_msg.get("tool_calls", [])

                                for tool_call in tool_calls:
                                    tool_call_id = tool_call.get("id")
                                    tool_function_name = tool_call.get("function", {}).get("name")

                                    if tool_call_id and tool_call_id in tool_results_map:
                                        tool_info = tool_results_map[tool_call_id]

                                        # Check if it's search_knowledge_base
                                        if tool_info["tool_name"] == "search_knowledge_base" or tool_function_name == "search_knowledge_base":
                                            # Parse the result JSON to extract sources
                                            try:
                                                result_data = json.loads(tool_info["result"])
                                                if isinstance(result_data, list):
                                                    for item in result_data:
                                                        if isinstance(item, dict):
                                                            sources.append({
                                                                "document_id": item.get("file_id"),
                                                                "filename": item.get("metadata", {}).get("file_name", ""),
                                                                "folder_name": item.get("metadata", {}).get("folder_name", ""),
                                                                "text": item.get("text", ""),
                                                                "score": item.get("metadata", {}).get("score", 0),
                                                                "file_key": item.get("metadata", {}).get("file_key", ""),
                                                                # Video fields
                                                                "video_id": item.get("metadata", {}).get("video_id"),
                                                                "video_name": item.get("metadata", {}).get("video_name"),
                                                                "clip_start": item.get("metadata", {}).get("clip_start"),
                                                                "clip_end": item.get("metadata", {}).get("clip_end"),
                                                                "scene_id": item.get("metadata", {}).get("scene_id"),
                                                                "key_frame_timestamp": item.get("metadata", {}).get("key_frame_timestamp"),
                                                                "keyframe_file_key": item.get("metadata", {}).get("keyframe_file_key"),
                                                            })
                                            except json.JSONDecodeError as e:
                                                logger.warning(f"Failed to parse tool result for tool_call_id {tool_call_id}: {e}")
                                # Stop after finding the first assistant message with tool_calls
                                break

                        if sources:
                            message_data["sources"] = sources
                            logger.info(f"Added {len(sources)} sources to assistant message")

                    messages.append(message_data)

        logger.info(f"✅ Retrieved {len(messages)} messages from session")

        return {
            "success": True,
            "session_id": session_id,
            "session_name": session.get("session_data", {}).get("session_name"),
            "messages": messages,
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_keycloak)
) -> Dict[str, Any]:
    """
    Delete a chat session

    Permanently deletes a session and all its messages.
    Only the owner can delete their sessions.

    Note: agent_sessions collection only stores user_id (not organization_id).

    Args:
        session_id: The unique session identifier
        current_user: Authenticated user from token

    Returns:
        Dict with success status
    """
    try:
        # Extract user_id from JWT token
        user_id = current_user.get("id")

        logger.info(f"🗑️ Deleting chat session: {session_id} for user: {user_id}")

        mongodb = get_mongodb_client()

        # Security: Ensure user owns this session (filter by user_id only)
        query = {
            "session_id": session_id,
            "user_id": user_id
        }

        deleted_count = await mongodb.async_delete_document(
            collection="agent_sessions",
            query=query
        )

        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Session not found or you don't have access to it"
            )

        logger.info(f"✅ Deleted chat session: {session_id}")

        return {
            "success": True,
            "message": "Session deleted successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.put("/chat/sessions/{session_id}/name")
async def rename_chat_session(
    session_id: str,
    new_name: str,
    current_user: dict = Depends(get_current_user_keycloak)
) -> Dict[str, Any]:
    """
    Rename a chat session

    Updates the session name to a custom value.
    Only the owner can rename their sessions.

    Note: agent_sessions collection only stores user_id (not organization_id).

    Args:
        session_id: The unique session identifier
        new_name: The new name for the session
        current_user: Authenticated user from token

    Returns:
        Dict with success status
    """
    try:
        # Extract user_id from JWT token
        user_id = current_user.get("id")

        if not new_name or not new_name.strip():
            raise HTTPException(status_code=400, detail="Session name cannot be empty")

        logger.info(f"✏️ Renaming chat session: {session_id} to '{new_name}'")

        mongodb = get_mongodb_client()

        # Security: Ensure user owns this session (filter by user_id only)
        query = {
            "session_id": session_id,
            "user_id": user_id
        }

        # Update session name
        modified_count = await mongodb.async_update_document(
            collection="agent_sessions",
            query=query,
            update={"$set": {"session_data.session_name": new_name.strip()}}
        )

        if modified_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Session not found or you don't have access to it"
            )

        logger.info(f"✅ Renamed chat session: {session_id}")

        return {
            "success": True,
            "message": "Session renamed successfully",
            "session_id": session_id,
            "new_name": new_name.strip()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to rename chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for chat service"""
    return {"status": "healthy", "service": "chat"}
