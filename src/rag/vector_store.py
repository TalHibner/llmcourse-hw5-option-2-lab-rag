"""
Vector Store Wrapper for ChromaDB

Provides unified interface for vector storage and retrieval.
Handles embeddings, metadata, and similarity search.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Document representation

    Input Data:
    - id: str - unique document identifier
    - text: str - document text content
    - metadata: Dict - additional metadata

    Output Data:
    - Document object with all fields populated
    """
    id: str
    text: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate document"""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.text:
            raise ValueError("Document text cannot be empty")


class VectorStore:
    """
    Vector store for embedding storage and retrieval

    Setup Data:
    - collection_name: str - ChromaDB collection name
    - persist_directory: str - directory for persistence
    - embedding_function: callable - function to generate embeddings
    - distance_metric: str - similarity metric (cosine, l2, ip)

    Input Data (add_documents):
    - documents: List[Document] - documents to add
    - embeddings: Optional[List[List[float]]] - precomputed embeddings

    Input Data (search):
    - query: str - search query
    - query_embedding: Optional[List[float]] - precomputed query embedding
    - top_k: int - number of results to return

    Output Data (search):
    - List[Dict] with: document, score, metadata
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_function: callable,
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store

        Args:
            collection_name: Name for the collection
            persist_directory: Directory to persist data
            embedding_function: Function to generate embeddings
            distance_metric: Distance metric (cosine, l2, ip)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.distance_metric = distance_metric

        # Validate distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")

        # Initialize ChromaDB client
        self.client = chromadb.Client(ChromaSettings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

        logger.info(f"Initialized VectorStore: {collection_name} ({distance_metric})")

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Add documents to vector store

        Args:
            documents: List of documents to add
            embeddings: Optional precomputed embeddings

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Extract components
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings if not provided
        if embeddings is None:
            logger.debug(f"Generating embeddings for {len(documents)} documents")
            embeddings = [self.embedding_function(text) for text in texts]

        # Validate embedding dimensions
        if embeddings and len(embeddings) != len(documents):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of documents ({len(documents)})"
            )

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents to {self.collection_name}")

    def search(
        self,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query: Search query text
            query_embedding: Optional precomputed query embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of dictionaries containing:
            - id: str - document ID
            - text: str - document text
            - score: float - similarity score
            - metadata: Dict - document metadata

        Raises:
            ValueError: If neither query nor query_embedding provided
        """
        if not query and query_embedding is None:
            raise ValueError("Must provide either query or query_embedding")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Generate query embedding if not provided
        if query_embedding is None:
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embedding_function(query)

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': 1.0 - results['distances'][0][i] if self.distance_metric == "cosine" else results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })

        logger.debug(f"Found {len(formatted_results)} results for query")
        return formatted_results

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by IDs

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            return

        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")

    def clear(self) -> None:
        """Clear all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    def count(self) -> int:
        """
        Get document count

        Returns:
            Number of documents in collection
        """
        return self.collection.count()

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        results = self.collection.get(ids=[doc_id])

        if results['ids'] and len(results['ids']) > 0:
            return Document(
                id=results['ids'][0],
                text=results['documents'][0],
                metadata=results['metadatas'][0]
            )

        return None
