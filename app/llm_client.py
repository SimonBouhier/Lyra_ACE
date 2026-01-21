"""
LYRA CLEAN - ASYNC OLLAMA CLIENT
=================================

Non-blocking HTTP client for Ollama API.

Replaces:
- lyra_core/ollama_wrapper.py (blocking requests)

Key improvements:
- Async I/O (httpx instead of requests)
- Connection pooling
- Timeout handling
- Retry logic
"""
from __future__ import annotations

import asyncio
import httpx
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from core.physics import PhysicsState, map_tau_to_temperature, map_rho_to_penalties

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async client for Ollama API.

    Features:
    - Non-blocking HTTP requests
    - Connection pooling (reuse TCP connections)
    - Automatic retries on network errors
    - Timeout protection
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: float = 180.0,
        max_retries: int = 3,
        num_ctx: int = 8192
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Default model name
            timeout: Request timeout (seconds)
            max_retries: Max retry attempts on failure
            num_ctx: Context window size in tokens (default 8192, max ~128k)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.num_ctx = num_ctx

        # HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """
        Initialize HTTP client with connection pooling.

        Must be called before making requests.
        """
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            )
        )

    async def close(self) -> None:
        """Close HTTP client and release connections."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def session(self):
        """
        Context manager for client lifecycle.

        Usage:
            async with client.session():
                response = await client.chat(...)
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama server is reachable.

        Returns:
            Dict with keys: connected (bool), models (list), error (str)

        Example:
            health = await client.health_check()
            if health["connected"]:
                print(f"Models: {health['models']}")
        """
        try:
            if not self._client:
                await self.initialize()

            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]

            return {
                "connected": True,
                "models": models,
                "model": self.model,
                "available": self.model in models
            }

        except Exception as e:
            return {
                "connected": False,
                "models": [],
                "error": str(e)
            }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        physics_state: PhysicsState,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat completion request to Ollama.

        Args:
            messages: List of {role, content} dicts
            physics_state: Current physics state (for parameter mapping)
            model: Model override (uses default if None)
            stream: Enable streaming (not implemented yet)

        Returns:
            Dict with keys:
            - text: Generated response
            - model: Model used
            - latency_ms: Generation time
            - tokens: Token counts (approximate)

        Example:
            response = await client.chat(
                messages=[
                    {"role": "system", "content": "You are Lyra"},
                    {"role": "user", "content": "Hello"}
                ],
                physics_state=state
            )
            print(response["text"])
        """
        if not self._client:
            await self.initialize()

        # Map physics parameters to Ollama options
        temperature = map_tau_to_temperature(physics_state.tau_c)
        penalties = map_rho_to_penalties(physics_state.rho)

        # Build request payload
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,  # Max tokens to generate
                "num_ctx": self.num_ctx,  # Context window size (total input + output)
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.0 + penalties["frequency_penalty"],
                # Note: Ollama doesn't have presence_penalty, use repeat_penalty
            }
        }

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                # Extract response text
                text = data.get("message", {}).get("content", "")

                if not text:
                    raise ValueError("Empty response from Ollama")

                # Estimate tokens (rough)
                prompt_text = " ".join(m["content"] for m in messages)
                prompt_tokens = len(prompt_text) // 4
                completion_tokens = len(text) // 4

                return {
                    "text": text,
                    "model": payload["model"],
                    "latency_ms": latency_ms,
                    "tokens": {
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": prompt_tokens + completion_tokens
                    }
                }

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code == 404:
                    # Model not found, don't retry
                    break

            except httpx.RequestError as e:
                last_error = f"Network error: {str(e)}"

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)

        # All retries failed
        raise RuntimeError(f"Ollama request failed after {self.max_retries} attempts: {last_error}")

    async def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names

        Example:
            models = await client.list_models()
            # ["gpt-oss:20b", "llama3:latest", ...]
        """
        if not self._client:
            await self.initialize()

        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            return [m["name"] for m in data.get("models", [])]

        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_client_instance: Optional[OllamaClient] = None


async def get_ollama_client(
    base_url: str = None,
    model: str = None,
    num_ctx: int = None
) -> OllamaClient:
    """
    Get or create Ollama client instance (singleton).

    Environment variables:
        LYRA_MODEL: Model name (default: gpt-oss:20b)
        LYRA_OLLAMA_URL: Ollama base URL (default: http://localhost:11434)
        LYRA_NUM_CTX: Context window size in tokens (default: 8192)

    Usage:
        client = await get_ollama_client()
        response = await client.chat(messages, physics_state)
    """
    global _client_instance

    if _client_instance is None:
        import os
        actual_base_url = base_url or os.environ.get("LYRA_OLLAMA_URL", "http://localhost:11434")
        actual_model = model or os.environ.get("LYRA_MODEL", "gpt-oss:20b")
        actual_num_ctx = num_ctx or int(os.environ.get("LYRA_NUM_CTX", "8192"))

        logger.info(f"Initializing Ollama client with model: {actual_model}, context: {actual_num_ctx} tokens")
        _client_instance = OllamaClient(
            base_url=actual_base_url,
            model=actual_model,
            num_ctx=actual_num_ctx
        )
        await _client_instance.initialize()

    return _client_instance


async def close_ollama_client() -> None:
    """
    Close Ollama client (cleanup on shutdown).
    """
    global _client_instance

    if _client_instance:
        await _client_instance.close()
        _client_instance = None


# ============================================================================
# MULTI-MODEL WRAPPER (Lyra-ACE)
# ============================================================================

@dataclass
class ModelResponse:
    """Réponse d'un modèle unique."""
    model: str
    text: str
    latency_ms: float
    tokens: Dict[str, int]
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "success": self.success,
            "error": self.error
        }


@dataclass
class ConsensusMetrics:
    """Métriques de consensus inter-modèles."""
    num_models: int
    num_successful: int
    response_lengths: List[int]
    length_variance: float
    avg_latency_ms: float
    model_weights: Dict[str, float]  # Poids suggérés basés sur la performance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_models": self.num_models,
            "num_successful": self.num_successful,
            "response_lengths": self.response_lengths,
            "length_variance": self.length_variance,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "model_weights": {k: round(v, 3) for k, v in self.model_weights.items()}
        }


class MultiModelClient:
    """
    Wrapper pour générations multi-modèles séquentielles.

    Permet de comparer les réponses de plusieurs modèles Ollama
    et de calculer des métriques de consensus.

    Usage:
        client = MultiModelClient()
        await client.initialize()

        responses = await client.generate_sequential(
            messages=[{"role": "user", "content": "Hello"}],
            models=["llama3.1:8b", "mistral", "gemma3"],
            physics_state=state
        )

        consensus = client.compute_consensus(responses)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        num_ctx: int = 8192
    ):
        """
        Args:
            base_url: URL du serveur Ollama
            timeout: Timeout par requête
            num_ctx: Taille du contexte
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._client: Optional[httpx.AsyncClient] = None

        # Cache des modèles disponibles
        self._available_models: Optional[List[str]] = None

    async def initialize(self) -> None:
        """Initialise le client HTTP."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
        )

    async def close(self) -> None:
        """Ferme le client HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def list_available_models(self, refresh: bool = False) -> List[str]:
        """
        Liste les modèles disponibles sur Ollama.

        Args:
            refresh: Force le rafraîchissement du cache

        Returns:
            Liste des noms de modèles
        """
        if self._available_models and not refresh:
            return self._available_models

        if not self._client:
            await self.initialize()

        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            self._available_models = [m["name"] for m in data.get("models", [])]
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def generate_single(
        self,
        messages: List[Dict[str, str]],
        model: str,
        physics_state: PhysicsState,
        temperature_override: Optional[float] = None
    ) -> ModelResponse:
        """
        Génère une réponse avec un seul modèle.

        Args:
            messages: Messages de conversation
            model: Nom du modèle
            physics_state: État physique pour les paramètres
            temperature_override: Override de température (optionnel)

        Returns:
            ModelResponse avec le résultat ou l'erreur
        """
        if not self._client:
            await self.initialize()

        # Calculer température
        if temperature_override is not None:
            temperature = temperature_override
        else:
            temperature = map_tau_to_temperature(physics_state.tau_c)

        penalties = map_rho_to_penalties(physics_state.rho)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
                "num_ctx": self.num_ctx,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.0 + penalties["frequency_penalty"]
            }
        }

        start_time = time.time()

        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            text = data.get("message", {}).get("content", "")

            # Estimation tokens
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(text) // 4

            return ModelResponse(
                model=model,
                text=text,
                latency_ms=latency_ms,
                tokens={
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": prompt_tokens + completion_tokens
                },
                success=True
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return ModelResponse(
                model=model,
                text="",
                latency_ms=latency_ms,
                tokens={"prompt": 0, "completion": 0, "total": 0},
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ModelResponse(
                model=model,
                text="",
                latency_ms=latency_ms,
                tokens={"prompt": 0, "completion": 0, "total": 0},
                success=False,
                error=str(e)
            )

    async def generate_sequential(
        self,
        messages: List[Dict[str, str]],
        models: List[str],
        physics_state: PhysicsState,
        stop_on_first_success: bool = False
    ) -> Dict[str, ModelResponse]:
        """
        Génère des réponses séquentiellement avec plusieurs modèles.

        Args:
            messages: Messages de conversation
            models: Liste des modèles à utiliser
            physics_state: État physique pour les paramètres
            stop_on_first_success: Arrêter dès qu'un modèle répond avec succès

        Returns:
            Dict[model_name, ModelResponse]
        """
        responses = {}

        for model in models:
            logger.info(f"[MultiModel] Generating with {model}...")
            response = await self.generate_single(
                messages=messages,
                model=model,
                physics_state=physics_state
            )
            responses[model] = response

            if stop_on_first_success and response.success:
                logger.info(f"[MultiModel] Success with {model}, stopping")
                break

            if not response.success:
                logger.warning(f"[MultiModel] {model} failed: {response.error}")

        return responses

    def compute_consensus(
        self,
        responses: Dict[str, ModelResponse]
    ) -> ConsensusMetrics:
        """
        Calcule les métriques de consensus entre les réponses.

        Args:
            responses: Dict des réponses par modèle

        Returns:
            ConsensusMetrics avec variance, poids suggérés, etc.
        """
        successful = {k: v for k, v in responses.items() if v.success}

        if not successful:
            return ConsensusMetrics(
                num_models=len(responses),
                num_successful=0,
                response_lengths=[],
                length_variance=0.0,
                avg_latency_ms=0.0,
                model_weights={}
            )

        # Longueurs des réponses
        lengths = [len(r.text) for r in successful.values()]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)

        # Latences
        latencies = [r.latency_ms for r in successful.values()]
        avg_latency = sum(latencies) / len(latencies)

        # Calcul des poids basé sur la latence et la longueur
        # Plus rapide = meilleur, longueur proche de la moyenne = meilleur
        weights = {}
        for model, response in successful.items():
            # Score latence (inverse normalisé)
            latency_score = 1.0 / (1.0 + response.latency_ms / 1000)

            # Score longueur (pénalise les extrêmes)
            length_diff = abs(len(response.text) - avg_length)
            length_score = 1.0 / (1.0 + length_diff / avg_length) if avg_length > 0 else 1.0

            weights[model] = latency_score * 0.4 + length_score * 0.6

        # Normaliser les poids
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return ConsensusMetrics(
            num_models=len(responses),
            num_successful=len(successful),
            response_lengths=lengths,
            length_variance=length_variance,
            avg_latency_ms=avg_latency,
            model_weights=weights
        )

    def select_best_response(
        self,
        responses: Dict[str, ModelResponse],
        consensus: ConsensusMetrics
    ) -> Optional[ModelResponse]:
        """
        Sélectionne la meilleure réponse basée sur le consensus.

        Args:
            responses: Dict des réponses
            consensus: Métriques de consensus calculées

        Returns:
            Meilleure réponse ou None si aucune
        """
        if not consensus.model_weights:
            # Fallback : première réponse réussie
            for response in responses.values():
                if response.success:
                    return response
            return None

        # Sélectionner le modèle avec le poids le plus élevé
        best_model = max(consensus.model_weights.keys(),
                         key=lambda m: consensus.model_weights[m])
        return responses.get(best_model)


# ============================================================================
# SINGLETON MULTI-MODEL INSTANCE
# ============================================================================

_multi_model_instance: Optional[MultiModelClient] = None


async def get_multi_model_client(
    base_url: str = None,
    num_ctx: int = None
) -> MultiModelClient:
    """
    Get or create MultiModelClient instance (singleton).

    Usage:
        client = await get_multi_model_client()
        responses = await client.generate_sequential(...)
    """
    global _multi_model_instance

    if _multi_model_instance is None:
        import os
        actual_base_url = base_url or os.environ.get("LYRA_OLLAMA_URL", "http://localhost:11434")
        actual_num_ctx = num_ctx or int(os.environ.get("LYRA_NUM_CTX", "8192"))

        logger.info(f"Initializing MultiModelClient, context: {actual_num_ctx} tokens")
        _multi_model_instance = MultiModelClient(
            base_url=actual_base_url,
            num_ctx=actual_num_ctx
        )
        await _multi_model_instance.initialize()

    return _multi_model_instance


async def close_multi_model_client() -> None:
    """Close MultiModelClient (cleanup on shutdown)."""
    global _multi_model_instance

    if _multi_model_instance:
        await _multi_model_instance.close()
        _multi_model_instance = None
