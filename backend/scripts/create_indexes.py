#!/usr/bin/env python3
"""
MongoDB Index Creation Script

Run this script to create all required indexes for the application.
This should be run once during deployment or when new indexes are added.

Usage:
    python scripts/create_indexes.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import ASCENDING, DESCENDING
from clients.mongodb_client import get_mongodb_client
from app.logger import logger


def create_documents_indexes(db):
    """Create indexes for documents collection"""
    logger.info("📊 Creating indexes for 'documents' collection...")

    documents_collection = db["documents"]
    existing_indexes = documents_collection.index_information()

    indexes_to_create = [
        ("user_id_1", [("user_id", ASCENDING)]),
        ("organization_id_1", [("organization_id", ASCENDING)]),
        ("folder_name_1", [("folder_name", ASCENDING)]),
        ("created_at_-1", [("created_at", DESCENDING)]),
        ("file_name_1", [("file_name", ASCENDING)]),
        ("organization_id_1_folder_name_1", [("organization_id", ASCENDING), ("folder_name", ASCENDING)]),
        ("organization_id_1_user_id_1", [("organization_id", ASCENDING), ("user_id", ASCENDING)])
    ]

    created_count = 0
    skipped_count = 0

    for index_name, index_keys in indexes_to_create:
        if index_name not in existing_indexes:
            documents_collection.create_index(index_keys, name=index_name)
            logger.info(f"  ✅ Created index: {index_name}")
            created_count += 1
        else:
            logger.info(f"  ⏭️  Index already exists: {index_name}")
            skipped_count += 1

    logger.info(f"📊 Documents collection: {created_count} created, {skipped_count} skipped\n")


def create_ingestion_tasks_indexes(db):
    """Create indexes for ingestion_tasks collection"""
    logger.info("📊 Creating indexes for 'ingestion_tasks' collection...")

    tasks_collection = db["ingestion_tasks"]
    existing_indexes = tasks_collection.index_information()

    indexes_to_create = [
        ("status_1", [("status", ASCENDING)]),
        ("user_id_1", [("user_id", ASCENDING)]),
        ("organization_id_1", [("organization_id", ASCENDING)]),
        ("created_at_-1", [("created_at", DESCENDING)]),
        ("updated_at_-1", [("updated_at", DESCENDING)])
    ]

    created_count = 0
    skipped_count = 0

    for index_name, index_keys in indexes_to_create:
        if index_name not in existing_indexes:
            tasks_collection.create_index(index_keys, name=index_name)
            logger.info(f"  ✅ Created index: {index_name}")
            created_count += 1
        else:
            logger.info(f"  ⏭️  Index already exists: {index_name}")
            skipped_count += 1

    logger.info(f"📊 Ingestion tasks collection: {created_count} created, {skipped_count} skipped\n")


def main():
    """Main function to create all indexes"""
    try:
        logger.info("🚀 Starting MongoDB index creation...\n")

        # Get MongoDB client
        mongodb_client = get_mongodb_client()
        db = mongodb_client.sync_db

        # Create indexes for each collection
        create_documents_indexes(db)
        create_ingestion_tasks_indexes(db)

        logger.info("✅ All indexes created successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Failed to create indexes: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
