"""
Ollama API Client

Wrapper for Ollama REST API providing text generation and embeddings.
Handles errors, retries, and response validation.
"""

import requests
import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OllamaAPIError(Exception):
    """Raised when Ollama API returns an error"""
    pass


class OllamaClient:
    """
    Ollama API client for LLM operations

    Input Data (generate):
    - prompt: str - text prompt for generation
    - context: Optional[str] - additional context

    Input Data (embed):
    - text: str - text to embed

    Setup Data:
    - model_name: str - Ollama model (e.g., 'llama2')
    - base_url: str - API endpoint
    - temperature: float - generation temperature
    - max_tokens: int - maximum tokens to generate
    - timeout: int - request timeout in seconds

    Output Data (generate):
    - Dict with: response, tokens, generation_time_ms

    Output Data (embed):
    - List[float] - embedding vector
    """

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama client

        Args:
            model_name: Ollama model name
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        self._validate_config()
        logger.info(f"Initialized OllamaClient with model: {model_name}")

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if not 0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2.0], got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text response using Ollama

        Args:
            prompt: User prompt
            context: Optional context to prepend
            system_prompt: Optional system prompt

        Returns:
            Dictionary containing:
            - response: str - generated text
            - tokens: int - number of tokens generated
            - generation_time_ms: float - generation time
            - model: str - model used

        Raises:
            ValueError: If prompt is empty
            OllamaAPIError: If API call fails
        """
        # Input validation
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Build full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\n{prompt}"

        # Prepare request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        # Call API with retries
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling Ollama API (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    generation_time = (time.time() - start_time) * 1000  # ms

                    return {
                        "response": result.get("response", ""),
                        "tokens": result.get("eval_count", 0),
                        "generation_time_ms": generation_time,
                        "model": self.model_name,
                        "done": result.get("done", False)
                    }
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)

                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise OllamaAPIError(error_msg)

            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise OllamaAPIError("Request timed out after all retries")

            except requests.exceptions.ConnectionError:
                logger.error(f"Connection error (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise OllamaAPIError(
                        "Could not connect to Ollama. Is the Ollama service running?"
                    )

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise OllamaAPIError(f"Unexpected error: {str(e)}")

        raise OllamaAPIError("Failed after all retry attempts")

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text

        Args:
            text: Text to embed

        Returns:
            Embedding vector (typically 768 dimensions for nomic-embed-text)

        Raises:
            ValueError: If text is empty
            OllamaAPIError: If API call fails
        """
        # Input validation
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Prepare request
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": "nomic-embed-text",  # Use embedding model
            "prompt": text
        }

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling Ollama embeddings API (attempt {attempt + 1})")
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])

                    if not embedding:
                        raise OllamaAPIError("Empty embedding returned")

                    logger.debug(f"Generated embedding of dimension {len(embedding)}")
                    return embedding
                else:
                    error_msg = f"Ollama API error: {response.status_code}"
                    logger.error(error_msg)

                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        raise OllamaAPIError(error_msg)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise OllamaAPIError("Embedding request timed out")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise OllamaAPIError(f"Embedding error: {str(e)}")

        raise OllamaAPIError("Failed to generate embeddings")

    def check_availability(self) -> bool:
        """
        Check if Ollama service is available

        Returns:
            True if available, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models

        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
