"""
Pinecone Vector Database Client
Uses LangChain for vector storage and retrieval operations
"""

from typing import Optional, List, Dict, Any
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from app.settings import settings
from app.logger import logger


class PineconeClient:
    """Client for Pinecone vector database operations using LangChain"""

    def __init__(self):
        """Initialize Pinecone client with LangChain"""
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embedding_model = settings.OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not configured")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME not configured")
        if not self.embedding_model:
            raise ValueError("OPENAI_API_KEY not configured for embeddings")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.embedding_model,
            model="text-embedding-3-small"  # Efficient embedding model
        )

        # Check if index exists, create if not
        self._ensure_index_exists()

        # Initialize LangChain vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        logger.info(f"✅ Pinecone client initialized with index: {self.index_name}")

    def _ensure_index_exists(self):
        """Ensure Pinecone index exists, create if not"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")

                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )

                logger.info(f"✅ Pinecone index created: {self.index_name}")
            else:
                logger.info(f"✅ Pinecone index exists: {self.index_name}")

        except Exception as e:
            logger.error(f"❌ Failed to ensure index exists: {str(e)}")
            raise

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to Pinecone vector store

        Args:
            texts: List of text chunks to embed and store
            metadatas: List of metadata dicts for each text chunk
            ids: Optional list of IDs for each document
            namespace: Optional namespace for multi-tenancy (e.g., organization_id)

        Returns:
            List[str]: List of document IDs

        Raises:
            Exception: If adding documents fails
        """
        try:
            # Create LangChain Document objects
            documents = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]

            # Create vector store with namespace if provided
            if namespace:
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=namespace
                )
            else:
                vector_store = self.vector_store

            # Add to vector store
            if ids:
                doc_ids = vector_store.add_documents(documents, ids=ids)
            else:
                doc_ids = vector_store.add_documents(documents)

            logger.info(f"✅ Added {len(doc_ids)} documents to Pinecone (namespace: {namespace or 'default'})")
            return doc_ids

        except Exception as e:
            logger.error(f"❌ Failed to add documents to Pinecone: {str(e)}")
            raise Exception(f"Failed to add documents: {str(e)}")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic similarity

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            namespace: Optional namespace for multi-tenancy

        Returns:
            List[Document]: List of similar documents with metadata

        Raises:
            Exception: If search fails
        """
        try:
            # Create vector store with namespace if provided
            if namespace:
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=namespace
                )
            else:
                vector_store = self.vector_store

            if filter:
                results = vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter
                )
            else:
                results = vector_store.similarity_search(query, k=k)

            logger.info(f"✅ Found {len(results)} similar documents (namespace: {namespace or 'default'})")
            return results

        except Exception as e:
            logger.error(f"❌ Similarity search failed: {str(e)}")
            raise Exception(f"Similarity search failed: {str(e)}")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores

        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            namespace: Optional namespace for multi-tenancy

        Returns:
            List[tuple]: List of (Document, score) tuples

        Raises:
            Exception: If search fails
        """
        try:
            # Create vector store with namespace if provided
            if namespace:
                vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=namespace
                )
            else:
                vector_store = self.vector_store

            if filter:
                results = vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter
                )
            else:
                results = vector_store.similarity_search_with_score(query, k=k)

            logger.info(f"✅ Found {len(results)} similar documents with scores (namespace: {namespace or 'default'})")
            return results

        except Exception as e:
            logger.error(f"❌ Similarity search with score failed: {str(e)}")
            raise Exception(f"Similarity search failed: {str(e)}")

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Delete documents from Pinecone by IDs or filter

        Args:
            ids: Optional list of document IDs to delete
            filter: Optional metadata filter for deletion
            namespace: Optional namespace for multi-tenancy

        Returns:
            bool: True if deletion was successful

        Raises:
            Exception: If deletion fails
        """
        try:
            index = self.pc.Index(self.index_name)

            if ids:
                if namespace:
                    index.delete(ids=ids, namespace=namespace)
                else:
                    index.delete(ids=ids)
                logger.info(f"✅ Deleted {len(ids)} documents from Pinecone (namespace: {namespace or 'default'})")
            elif filter:
                if namespace:
                    index.delete(filter=filter, namespace=namespace)
                else:
                    index.delete(filter=filter)
                logger.info(f"✅ Deleted documents matching filter from Pinecone (namespace: {namespace or 'default'})")
            else:
                raise ValueError("Either ids or filter must be provided")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete documents from Pinecone: {str(e)}")
            raise Exception(f"Failed to delete documents: {str(e)}")

    def delete_by_knowledge_base(self, kb_name: str) -> bool:
        """
        Delete all documents belonging to a knowledge base

        Args:
            kb_name: Knowledge base name

        Returns:
            bool: True if deletion was successful
        """
        try:
            return self.delete_documents(filter={"kb_name": kb_name})
        except Exception as e:
            logger.error(f"❌ Failed to delete KB {kb_name}: {str(e)}")
            raise

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks belonging to a document

        Args:
            document_id: Document ID

        Returns:
            bool: True if deletion was successful
        """
        try:
            return self.delete_documents(filter={"document_id": document_id})
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {str(e)}")
            raise

    def get_retriever(self, k: int = 5, filter: Optional[Dict[str, Any]] = None):
        """
        Get a LangChain retriever for RAG applications

        Args:
            k: Number of documents to retrieve
            filter: Optional metadata filter

        Returns:
            VectorStoreRetriever: LangChain retriever object
        """
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

        logger.info(f"✅ Created retriever with k={k}")
        return retriever


# Singleton instance
_pinecone_client: Optional[PineconeClient] = None


def get_pinecone_client() -> PineconeClient:
    """
    Get or create PineconeClient singleton instance

    Returns:
        PineconeClient: Singleton client instance
    """
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = PineconeClient()
    return _pinecone_client
