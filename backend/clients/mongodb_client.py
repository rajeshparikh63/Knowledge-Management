"""
MongoDB Client
Database operations for document metadata and raw content storage
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.settings import settings
from app.logger import logger


class MongoDBClient:
    """Client for MongoDB operations"""

    def __init__(self):
        """Initialize MongoDB client"""
        self.connection_string =  settings.MONGODB_URL
        self.database_name =  settings.MONGODB_DATABASE

        if not self.connection_string:
            raise ValueError("MongoDB connection string not configured")

        # Sync client for non-async operations
        self.sync_client = MongoClient(self.connection_string)
        self.sync_db = self.sync_client[self.database_name]

        # Async client for async operations
        self.async_client = AsyncIOMotorClient(self.connection_string)
        self.async_db = self.async_client[self.database_name]

        # Test connection
        try:
            self.sync_client.admin.command('ping')
            logger.info(f"✅ MongoDB client initialized: {self.database_name}")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {str(e)}")
            raise

    @staticmethod
    def to_object_id(id_string: Optional[str]) -> Optional[ObjectId]:
        """
        Convert string ID to ObjectId

        Args:
            id_string: String representation of ObjectId

        Returns:
            ObjectId or None if conversion fails
        """
        if not id_string:
            return None
        try:
            return ObjectId(id_string)
        except Exception:
            return None

    @staticmethod
    def is_valid_object_id(id_string: str) -> bool:
        """
        Check if string is a valid ObjectId

        Args:
            id_string: String to validate

        Returns:
            True if valid ObjectId format
        """
        return ObjectId.is_valid(id_string)

    # Document Operations
    def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """
        Insert a document into MongoDB collection

        Args:
            collection: Collection name
            document: Document data

        Returns:
            str: Inserted document ID
        """
        result = self.sync_db[collection].insert_one(document)
        logger.info(f"✅ Document inserted into {collection}: {result.inserted_id}")
        return str(result.inserted_id)

    async def async_insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        """
        Async insert a document into MongoDB collection

        Args:
            collection: Collection name
            document: Document data

        Returns:
            str: Inserted document ID
        """
        result = await self.async_db[collection].insert_one(document)
        logger.info(f"✅ Document inserted into {collection}: {result.inserted_id}")
        return str(result.inserted_id)

    def find_document(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            Optional[Dict[str, Any]]: Document if found, None otherwise
        """
        document = self.sync_db[collection].find_one(query)
        return document

    async def async_find_document(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Async find a single document in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            Optional[Dict[str, Any]]: Document if found, None otherwise
        """
        document = await self.async_db[collection].find_one(query)
        return document

    def find_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter
            limit: Maximum number of documents to return
            skip: Number of documents to skip

        Returns:
            List[Dict[str, Any]]: List of documents
        """
        cursor = self.sync_db[collection].find(query)

        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        documents = list(cursor)
        logger.info(f"✅ Found {len(documents)} documents in {collection}")
        return documents

    async def async_find_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Async find multiple documents in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter
            limit: Maximum number of documents to return
            skip: Number of documents to skip

        Returns:
            List[Dict[str, Any]]: List of documents
        """
        cursor = self.async_db[collection].find(query)

        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        documents = await cursor.to_list(length=limit)
        logger.info(f"✅ Found {len(documents)} documents in {collection}")
        return documents

    def update_document(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """
        Update a document in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter
            update: Update operations

        Returns:
            int: Number of documents modified
        """
        result = self.sync_db[collection].update_one(query, {"$set": update})
        logger.info(f"✅ Updated {result.modified_count} document(s) in {collection}")
        return result.modified_count

    async def async_update_document(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """
        Async update a document in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter
            update: Update operations (should include operators like $set, $push, etc.)

        Returns:
            int: Number of documents modified
        """
        result = await self.async_db[collection].update_one(query, update)
        logger.info(f"✅ Updated {result.modified_count} document(s) in {collection}")
        return result.modified_count

    def delete_document(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Delete a document from MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            int: Number of documents deleted
        """
        result = self.sync_db[collection].delete_one(query)
        logger.info(f"✅ Deleted {result.deleted_count} document(s) from {collection}")
        return result.deleted_count

    async def async_delete_document(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Async delete a document from MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            int: Number of documents deleted
        """
        result = await self.async_db[collection].delete_one(query)
        logger.info(f"✅ Deleted {result.deleted_count} document(s) from {collection}")
        return result.deleted_count

    def count_documents(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Count documents in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            int: Number of documents matching query
        """
        count = self.sync_db[collection].count_documents(query)
        return count

    async def async_count_documents(self, collection: str, query: Dict[str, Any]) -> int:
        """
        Async count documents in MongoDB collection

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            int: Number of documents matching query
        """
        count = await self.async_db[collection].count_documents(query)
        return count

    def close(self):
        """Close MongoDB connections"""
        self.sync_client.close()
        self.async_client.close()
        logger.info("✅ MongoDB connections closed")


# Singleton instance
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """
    Get or create MongoDBClient singleton instance

    Returns:
        MongoDBClient: Singleton client instance
    """
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
    return _mongodb_client
