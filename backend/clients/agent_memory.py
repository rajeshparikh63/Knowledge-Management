"""
Agent Memory Manager
Provides database and memory management for agents
"""

import threading
from datetime import datetime
from typing import Optional
from agno.db.mongo import MongoDb
from agno.db.schemas import UserMemory
from agno.memory.manager import MemoryManager
from app.settings import settings
from app.logger import logger


# Patch for legacy memory format compatibility
_orig_user_memory_from_dict = UserMemory.from_dict.__func__


def _user_memory_from_dict_with_alias(cls, data):
    """Handle both old 'id' and new 'memory_id' formats"""
    data = dict(data)
    if "id" in data and "memory_id" not in data:
        data["memory_id"] = data.pop("id")

    allowed_keys = {
        "memory",
        "memory_id",
        "topics",
        "user_id",
        "input",
        "updated_at",
        "feedback",
        "agent_id",
        "team_id",
    }
    data = {k: v for k, v in data.items() if k in allowed_keys}
    return _orig_user_memory_from_dict(cls, data)


UserMemory.from_dict = classmethod(_user_memory_from_dict_with_alias)


class LimitedMemoryManager(MemoryManager):
    """Memory manager that enforces a cap on stored memories per user"""

    def __init__(self, max_memories_per_user: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.max_memories_per_user = max_memories_per_user

    def add_user_memory(
        self, memory: UserMemory, user_id: Optional[str] = None
    ) -> Optional[str]:
        """Add user memory with retention limit enforcement"""
        memory_id = super().add_user_memory(memory, user_id)

        # Enforce per-user limit
        if memory_id and self.db:
            target_user_id = user_id or memory.user_id or "default"
            memories = self.db.get_user_memories(user_id=target_user_id)

            if isinstance(memories, tuple):
                memories = memories[0]

            if memories and len(memories) > self.max_memories_per_user:
                # Sort by timestamp and keep newest N
                def memory_timestamp(mem: UserMemory) -> datetime:
                    return mem.updated_at or datetime.now()

                sorted_memories = sorted(memories, key=memory_timestamp, reverse=False)
                # Remove oldest memories
                for stale_memory in sorted_memories[: -self.max_memories_per_user]:
                    if stale_memory.memory_id:
                        self.db.delete_user_memory(memory_id=stale_memory.memory_id)

        return memory_id


class AgentMemoryManager:
    """
    Singleton manager for agent memory and database
    Thread-safe lazy initialization
    """

    _instance: Optional["AgentMemoryManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AgentMemoryManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._db_cache: Optional[MongoDb] = None
        self._memory_manager_cache: Optional[LimitedMemoryManager] = None
        self._db_lock = threading.Lock()
        self._memory_lock = threading.Lock()
        self._initialized = True

    def get_db(self) -> MongoDb:
        """Get or create shared MongoDb instance for agent storage"""
        if self._db_cache is None:
            with self._db_lock:
                if self._db_cache is None:
                    logger.info("Initializing MongoDb instance for agent storage")
                    self._db_cache = MongoDb(
                        db_url=settings.MONGODB_URL,
                        db_name=settings.MONGODB_DATABASE,
                        session_collection="agent_sessions",
                        memory_collection="agent_memory",
                        metrics_collection="agent_metrics",
                        eval_collection="agent_eval_runs",
                        knowledge_collection="agent_knowledge",
                    )
        return self._db_cache

    def get_memory_manager(self) -> LimitedMemoryManager:
        """Get or create memory manager with retention limits"""
        if self._memory_manager_cache is None:
            with self._memory_lock:
                if self._memory_manager_cache is None:
                    logger.info("Initializing limited memory manager")
                    self._memory_manager_cache = LimitedMemoryManager(
                        max_memories_per_user=20,
                        db=self.get_db(),
                        memory_capture_instructions="""
You are helping users find and work with their knowledge base. Focus on capturing meaningful information to provide personalized, context-aware assistance.

PROFESSIONAL & WORK CONTEXT:
- Job role, responsibilities, and team structure
- Current projects, initiatives, and their priorities
- Client needs, stakeholder requirements, and deadlines
- Work processes, workflows, and methodologies they follow
- Areas of expertise and professional domains
- Tools, technologies, and platforms they use

COMMUNICATION & STYLE PREFERENCES:
- Preferred communication tone (formal, casual, technical)
- How they like information presented (detailed vs. concise, visual vs. text)
- Response format preferences (bullet points, paragraphs, tables)
- Language complexity and terminology they use
- Their approach to problem-solving and decision-making

DOCUMENT & CONTENT PATTERNS:
- Types of documents they work with frequently (PDFs, videos, presentations, spreadsheets)
- Specific folders, projects, or content areas they reference often
- Topics and subjects they regularly search for
- Key files or videos they return to repeatedly
- Document organization and naming conventions they use
- Preferred file formats and content types

PROJECT & TASK MEMORY:
- Ongoing projects and their objectives
- Project-specific context and requirements
- Tasks and deliverables they're working on
- Project timelines and milestones
- Project team members and stakeholders

KNOWLEDGE & LEARNING PATTERNS:
- Recurring questions and information requests
- Topics they're learning about or researching
- Knowledge gaps and areas where they seek help
- Subject matter they're building expertise in
- Research interests and learning goals

PERSONAL CONTEXT (Work-Relevant):
- Important dates, deadlines, and recurring events
- Time zone and working hours patterns
- Key people, team members, or contacts they mention frequently
- Personal preferences that affect work (e.g., dietary restrictions for catering docs)
- Interests that inform their professional work

PRIORITIES & GOALS:
- What matters most to them in their work
- Short-term and long-term objectives
- Success criteria and how they measure progress
- Challenges and obstacles they're facing
- Strategic priorities that guide decisions

BEHAVIORAL PATTERNS:
- How they typically search for information
- Query patterns and common search terms
- Content consumption habits (prefer videos over text documents, etc.)
- Time-sensitive information needs or recurring requests
- Patterns that indicate future needs

IMPORTANT CONTEXT:
- Significant conversations, decisions, or insights
- Context that eliminates need to re-explain things
- Information that makes future interactions more efficient
- Historical patterns that help anticipate needs
- Long-term context that shapes recommendations

DO NOT CAPTURE UNLESS EXPLICITLY PROVIDED:
- Sensitive personal information (health, financial details)
- Confidential or privileged business information without clear work context
- Casual greetings or small talk without meaningful context
- Temporary one-off questions without lasting relevance
- Repetitive confirmations or basic acknowledgments
- Information the user explicitly asks not to remember

GUIDING PRINCIPLES:
- Remember information that makes you more helpful over time
- Capture context that personalizes the experience
- Focus on patterns that help anticipate user needs
- Prioritize professional and productivity-related details
- Respect privacy and only remember what's useful for assistance
- Store information that reduces friction in future interactions
""",
                    )
        return self._memory_manager_cache

    def clear_cache(self):
        """Clear cached instances"""
        with self._db_lock, self._memory_lock:
            self._db_cache = None
            self._memory_manager_cache = None

    @classmethod
    def reset_singleton(cls):
        """Reset singleton instance"""
        with cls._lock:
            if cls._instance:
                cls._instance.clear_cache()
                cls._instance = None


# Singleton instance
_agent_memory_manager: Optional[AgentMemoryManager] = None
_manager_lock = threading.Lock()


def get_agent_memory_manager() -> AgentMemoryManager:
    """Get or create singleton AgentMemoryManager instance"""
    global _agent_memory_manager

    if _agent_memory_manager is None:
        with _manager_lock:
            if _agent_memory_manager is None:
                _agent_memory_manager = AgentMemoryManager()

    return _agent_memory_manager


def get_agent_db() -> MongoDb:
    """Get agent database instance"""
    return get_agent_memory_manager().get_db()


def get_memory_manager() -> LimitedMemoryManager:
    """Get memory manager instance"""
    return get_agent_memory_manager().get_memory_manager()
