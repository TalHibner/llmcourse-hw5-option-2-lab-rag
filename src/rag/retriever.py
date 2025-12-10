"""
RAG Retriever

Implements retrieval-augmented generation pipeline:
1. Embed query
2. Retrieve top-k relevant documents
3. Generate response with context
"""

from typing import List, Dict, Any, Optional
import logging
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG retrieval pipeline

    Setup Data:
    - vector_store: VectorStore - initialized vector store
    - llm_client: LLM client with generate() method
    - top_k: int - number of documents to retrieve
    - reranking_enabled: bool - whether to rerank results

    Input Data:
    - query: str - user query
    - system_prompt: Optional[str] - system prompt

    Output Data:
    - Dict with: response, retrieved_docs, generation_time_ms, tokens
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: Any,
        top_k: int = 5,
        reranking_enabled: bool = False
    ):
        """
        Initialize RAG retriever

        Args:
            vector_store: Vector store for retrieval
            llm_client: LLM client for generation
            top_k: Number of documents to retrieve
            reranking_enabled: Enable reranking of results
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k
        self.reranking_enabled = reranking_enabled

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        logger.info(f"Initialized RAGRetriever with top_k={top_k}, reranking={reranking_enabled}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query: Query text
            top_k: Override default top_k

        Returns:
            List of retrieved documents with scores

        Raises:
            ValueError: If query is empty
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        k = top_k if top_k is not None else self.top_k

        logger.debug(f"Retrieving top {k} documents for query: {query[:50]}...")
        results = self.vector_store.search(query=query, top_k=k)

        if self.reranking_enabled:
            results = self._rerank(query, results)

        logger.info(f"Retrieved {len(results)} documents")
        return results

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using cross-encoder

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Reranked results
        """
        # Placeholder for reranking logic
        # In a full implementation, this would use a cross-encoder model
        logger.debug("Reranking disabled (placeholder)")
        return results

    def generate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using RAG pipeline

        Args:
            query: User query
            system_prompt: Optional system prompt
            top_k: Override default top_k

        Returns:
            Dictionary with:
            - response: str - generated response
            - retrieved_docs: List[Dict] - retrieved documents
            - generation_time_ms: float - generation time
            - tokens: int - tokens generated
            - retrieval_scores: List[float] - retrieval scores

        Raises:
            ValueError: If query is empty
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)

        if not retrieved_docs:
            logger.warning("No documents retrieved, generating without context")
            result = self.llm_client.generate(
                prompt=query,
                context="",
                system_prompt=system_prompt
            )
            return {
                **result,
                'retrieved_docs': [],
                'retrieval_scores': []
            }

        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)

        # Generate response with context
        logger.debug(f"Generating response with {len(retrieved_docs)} documents as context")
        result = self.llm_client.generate(
            prompt=query,
            context=context,
            system_prompt=system_prompt
        )

        # Add retrieval information
        result['retrieved_docs'] = retrieved_docs
        result['retrieval_scores'] = [doc['score'] for doc in retrieved_docs]

        return result

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract document text
            text = doc.get('text', '')

            # Add document with numbering
            context_parts.append(f"Document {i}:\n{text}")

        context = "\n\n".join(context_parts)
        logger.debug(f"Built context with {len(documents)} documents, {len(context)} chars")
        return context

    def evaluate(
        self,
        query: str,
        expected_answer: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline on a query

        Args:
            query: Query text
            expected_answer: Expected answer
            top_k: Override default top_k

        Returns:
            Dictionary with:
            - response: str - generated response
            - correct: bool - whether answer is correct
            - retrieved_docs: List[Dict] - retrieved documents
            - relevant_retrieved: bool - whether relevant doc was retrieved
        """
        # Generate response
        result = self.generate(query, top_k=top_k)

        # Check if answer is correct (simple substring match)
        response = result['response'].lower()
        expected = expected_answer.lower()
        correct = expected in response

        # Check if relevant document was retrieved
        relevant_retrieved = any(
            expected in doc['text'].lower()
            for doc in result['retrieved_docs']
        )

        return {
            'response': result['response'],
            'correct': correct,
            'retrieved_docs': result['retrieved_docs'],
            'retrieval_scores': result['retrieval_scores'],
            'relevant_retrieved': relevant_retrieved,
            'generation_time_ms': result['generation_time_ms'],
            'tokens': result['tokens']
        }
