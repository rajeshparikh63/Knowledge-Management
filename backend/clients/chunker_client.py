"""
Semantic Chunker Client - Chonkie Integration for Document Chunking
Thread-safe singleton with cached chunker instances for optimal performance
"""

import threading
from functools import lru_cache
from typing import List, Literal, Any, Union, Dict
from chonkie import SemanticChunker
from chonkie.embeddings.openai import OpenAIEmbeddings
import tiktoken
from app.logger import logger
from app.settings import settings


# Default chunking parameters
DEFAULT_THRESHOLD = 0.8
DEFAULT_CHUNK_SIZE = 1000  # Optimized for Pinecone
DEFAULT_SIMILARITY_WINDOW = 3
DEFAULT_MIN_SENTENCES_PER_CHUNK = 1
DEFAULT_MIN_CHARACTERS_PER_SENTENCE = 24
DEFAULT_DELIM = [". ", "! ", "? ", "\n"]
DEFAULT_INCLUDE_DELIM = "prev"

# Token limits for OpenAI
OPENAI_MAX_TOKENS_PER_REQUEST = 300000
SAFE_TOKEN_LIMIT = 200000  # Safety buffer
DEFAULT_ENCODING = "cl100k_base"  # For text-embedding models


@lru_cache(maxsize=4, typed=True)
def get_chonkie_embeddings(
    model: str = "text-embedding-3-small"
) -> OpenAIEmbeddings:
    """
    Get cached chonkie OpenAI embeddings instance

    Args:
        model: OpenAI embedding model to use

    Returns:
        Cached OpenAIEmbeddings instance
    """
    logger.info(f"Creating chonkie OpenAI embeddings with model: {model}")
    return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model=model)


@lru_cache(maxsize=16, typed=True)
def get_semantic_chunker(
    embedding_model: str = "text-embedding-3-small",
    threshold: Union[str, float, int] = DEFAULT_THRESHOLD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    similarity_window: int = DEFAULT_SIMILARITY_WINDOW,
    min_sentences_per_chunk: int = DEFAULT_MIN_SENTENCES_PER_CHUNK,
    min_characters_per_sentence: int = DEFAULT_MIN_CHARACTERS_PER_SENTENCE,
    delim: Union[str, List[str]] = None,
    include_delim: Union[Literal['prev', 'next'], None] = DEFAULT_INCLUDE_DELIM,
    **kwargs: Any
) -> SemanticChunker:
    """
    Get cached semantic chunker instance

    Args:
        embedding_model: OpenAI embedding model to use
        threshold: Threshold for semantic similarity (0-1)
        chunk_size: Maximum tokens allowed per chunk
        similarity_window: Number of sentences to consider for similarity
        min_sentences_per_chunk: Minimum number of sentences per chunk
        min_characters_per_sentence: Minimum number of characters per sentence
        delim: Delimiters to split sentences on
        include_delim: Whether to include the delimiters in the sentences
        **kwargs: Additional keyword arguments

    Returns:
        Cached SemanticChunker instance
    """
    try:
        # Set default delim if None
        if delim is None:
            delim = DEFAULT_DELIM.copy()

        # Get cached embeddings instance
        embeddings = get_chonkie_embeddings(model=embedding_model)

        # Create semantic chunker with all specified parameters
        chunker = SemanticChunker(
            embedding_model=embeddings,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=delim,
            include_delim=include_delim,
            **kwargs
        )

        logger.info(
            f"Created semantic chunker: threshold={threshold}, "
            f"chunk_size={chunk_size}, min_sentences_per_chunk={min_sentences_per_chunk}, "
            f"embedding_model={embedding_model}"
        )

        return chunker

    except Exception as e:
        logger.error(f"Failed to create semantic chunker: {e}")
        raise RuntimeError(f"Failed to create semantic chunker: {e}")


@lru_cache(maxsize=4)
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Get cached tokenizer for efficient token counting"""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Count tokens in text using OpenAI's tokenizer"""
    try:
        tokenizer = get_tokenizer(encoding_name)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}, falling back to character-based estimate")
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        return len(text) // 4


def chunk_large_content_for_embeddings(
    content: str,
    max_tokens: int = SAFE_TOKEN_LIMIT,
    overlap_tokens: int = 100,
    encoding_name: str = DEFAULT_ENCODING
) -> List[str]:
    """
    Split large content into chunks that fit within OpenAI's token limits

    Args:
        content: Text content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        encoding_name: Tokenizer encoding to use

    Returns:
        List of content chunks, each under the token limit
    """
    try:
        total_tokens = count_tokens(content, encoding_name)

        if total_tokens <= max_tokens:
            return [content]

        logger.info(f"Content has {total_tokens} tokens, chunking into max {max_tokens} token pieces")

        tokenizer = get_tokenizer(encoding_name)
        tokens = tokenizer.encode(content)

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + max_tokens, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start index forward, accounting for overlap
            if end_idx < len(tokens):  # Not the last chunk
                start_idx = end_idx - overlap_tokens
            else:
                break

        logger.info(f"Split {total_tokens} tokens into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk large content: {e}")
        # Fallback: split by characters if token processing fails
        chunk_size = max_tokens * 4  # Rough character estimate
        overlap_size = overlap_tokens * 4

        chunks = []
        for i in range(0, len(content), chunk_size - overlap_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)

        logger.warning(f"Used character-based chunking fallback: {len(chunks)} chunks")
        return chunks


def prepare_content_for_vectorization(
    content: str,
    metadata: Dict[str, Any],
    max_tokens_per_chunk: int = SAFE_TOKEN_LIMIT
) -> List[Dict[str, Any]]:
    """
    Prepare content for vectorization by ensuring it's under token limits

    Args:
        content: Text content to prepare
        metadata: Metadata to attach to each chunk
        max_tokens_per_chunk: Maximum tokens allowed per chunk

    Returns:
        List of documents ready for vectorization, each with content and metadata
    """
    try:
        # Check if content needs chunking
        token_count = count_tokens(content)

        if token_count <= max_tokens_per_chunk:
            return [{
                "content": content,
                "metadata": {**metadata, "token_count": token_count, "chunk_index": 0, "total_pre_chunks": 1}
            }]

        # Content needs chunking
        logger.warning(f"Content has {token_count} tokens, splitting for vectorization")

        content_chunks = chunk_large_content_for_embeddings(
            content=content,
            max_tokens=max_tokens_per_chunk
        )

        # Create documents for each chunk
        documents = []
        for i, chunk in enumerate(content_chunks):
            chunk_metadata = {
                **metadata,
                "token_count": count_tokens(chunk),
                "pre_chunk_index": i,
                "total_pre_chunks": len(content_chunks),
                "original_token_count": token_count
            }

            documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })

        logger.info(f"Prepared {len(documents)} documents for vectorization")
        return documents

    except Exception as e:
        logger.error(f"Failed to prepare content for vectorization: {e}")
        # Return original content with error metadata
        return [{
            "content": content,
            "metadata": {
                **metadata,
                "token_count": -1,
                "chunk_preparation_error": str(e),
                "chunk_index": 0,
                "total_pre_chunks": 1
            }
        }]


def validate_content_for_embeddings(content: str, max_tokens: int = OPENAI_MAX_TOKENS_PER_REQUEST) -> Dict[str, Any]:
    """
    Validate if content can be processed by OpenAI embeddings API

    Args:
        content: Text content to validate
        max_tokens: Maximum tokens allowed

    Returns:
        Dict with validation results and recommendations
    """
    try:
        token_count = count_tokens(content)

        return {
            "valid": token_count <= max_tokens,
            "token_count": token_count,
            "max_tokens": max_tokens,
            "needs_chunking": token_count > SAFE_TOKEN_LIMIT,
            "recommended_chunks": max(1, (token_count + SAFE_TOKEN_LIMIT - 1) // SAFE_TOKEN_LIMIT),
            "size_ratio": token_count / max_tokens
        }

    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        return {
            "valid": False,
            "token_count": -1,
            "max_tokens": max_tokens,
            "needs_chunking": True,
            "recommended_chunks": 1,
            "size_ratio": -1,
            "error": str(e)
        }


class ChunkerClient:
    """Thread-safe chunker client for semantic document chunking"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern with thread locking"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize chunker client"""
        if not hasattr(self, '_initialized'):
            self._chunking_lock = threading.Lock()
            self._initialized = True
            logger.info("✅ Chunker client initialized")

    def chunk_text(
        self,
        text: str,
        chunker_type: str = "default",
        **kwargs
    ) -> List[Any]:
        """
        Chunk text using semantic chunker

        Args:
            text: Text content to chunk
            chunker_type: Type of chunker to use (default, large_chunk, precise, fast)
            **kwargs: Additional parameters for chunker

        Returns:
            List of chunks (Chonkie Chunk objects)
        """
        with self._chunking_lock:
            try:
                # Get appropriate chunker based on type
                if chunker_type == "large_chunk":
                    chunker = get_semantic_chunker(
                        chunk_size=1200,
                        min_sentences_per_chunk=3,
                        threshold=0.4,
                        **kwargs
                    )
                elif chunker_type == "precise":
                    chunker = get_semantic_chunker(
                        chunk_size=300,
                        min_sentences_per_chunk=2,
                        threshold=0.7,
                        similarity_window=2,
                        **kwargs
                    )
                elif chunker_type == "fast":
                    chunker = get_semantic_chunker(
                        chunk_size=800,
                        threshold=0.8,
                        similarity_window=1,
                        **kwargs
                    )
                else:  # default
                    chunker = get_semantic_chunker(**kwargs)

                # Perform chunking
                chunks = chunker.chunk(text)

                logger.info(f"Chunked text into {len(chunks)} semantic chunks using {chunker_type} chunker")
                return chunks

            except Exception as e:
                logger.error(f"Chunking failed: {str(e)}")
                raise Exception(f"Chunking failed: {str(e)}")

    def chunk_with_metadata(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        chunker_type: str = "default",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and attach metadata to each chunk

        Args:
            text: Text content to chunk
            base_metadata: Base metadata to attach to all chunks
            chunker_type: Type of chunker to use
            **kwargs: Additional parameters for chunker

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        chunks = self.chunk_text(text, chunker_type, **kwargs)

        result = []
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk.text,
                "metadata": {
                    **base_metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.token_count
                }
            }
            result.append(chunk_data)

        return result

    def clear_cache(self) -> None:
        """Clear chunker and embeddings caches"""
        get_semantic_chunker.cache_clear()
        get_chonkie_embeddings.cache_clear()
        logger.info("Cleared chunker and embeddings caches")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the chunker and embeddings caches"""
        chunker_cache = get_semantic_chunker.cache_info()
        embeddings_cache = get_chonkie_embeddings.cache_info()

        return {
            "chunker_cache": {
                "hits": chunker_cache.hits,
                "misses": chunker_cache.misses,
                "maxsize": chunker_cache.maxsize,
                "currsize": chunker_cache.currsize,
                "hit_rate": chunker_cache.hits / (chunker_cache.hits + chunker_cache.misses) if (chunker_cache.hits + chunker_cache.misses) > 0 else 0.0
            },
            "embeddings_cache": {
                "hits": embeddings_cache.hits,
                "misses": embeddings_cache.misses,
                "maxsize": embeddings_cache.maxsize,
                "currsize": embeddings_cache.currsize,
                "hit_rate": embeddings_cache.hits / (embeddings_cache.hits + embeddings_cache.misses) if (embeddings_cache.hits + embeddings_cache.misses) > 0 else 0.0
            }
        }


# Singleton instance
_chunker_client = None
_client_lock = threading.Lock()


def get_chunker_client() -> ChunkerClient:
    """
    Get or create thread-safe ChunkerClient singleton instance

    Returns:
        ChunkerClient: Singleton client instance
    """
    global _chunker_client

    if _chunker_client is None:
        with _client_lock:
            if _chunker_client is None:
                _chunker_client = ChunkerClient()

    return _chunker_client
