"""
ESMM - MODEL ROTATOR FOR VRAM MANAGEMENT
=========================================

Sequential model rotation to prevent VRAM exhaustion on multi-model queries.

Strategy:
- Load one model at a time
- Process complete batch with that model
- Unload via keep_alive=0 before loading next
- Optional preload/warmup for latency-sensitive scenarios

Usage:
    rotator = ModelRotator(base_url="http://localhost:11434")
    await rotator.initialize()

    results = await rotator.rotate_and_process(
        models=["llama3.1:8b", "mistral", "gemma3"],
        question="What is entropy?",
        physics_state=state
    )

Author: Lyra-ACE ESMM Protocol
"""
from __future__ import annotations

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RotatedResponse:
    """Response from a single model in rotation."""
    model: str
    text: str
    latency_ms: float
    tokens: Dict[str, int] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    vram_freed: bool = False


@dataclass
class RotationResult:
    """Result of a complete rotation cycle."""
    responses: Dict[str, RotatedResponse]
    total_duration_ms: float
    models_processed: int
    models_failed: int
    vram_managed: bool


@dataclass
class BatchModelResult:
    """Result of batch sequential multi-model processing."""
    results: Dict[str, List[RotatedResponse]]  # model -> [responses]
    total_duration_ms: float
    models_processed: int
    questions_per_model: int
    vram_managed: bool


class ModelRotator:
    """
    Manages sequential model rotation with explicit VRAM control.

    Key Features:
    - Sequential loading prevents VRAM overflow
    - keep_alive=0 unloads model after use
    - Optional preload for warmup
    - Graceful degradation on failures

    VRAM Strategy:
    1. Check if model is loaded (optional)
    2. Generate response
    3. Send keep_alive=0 to unload
    4. Wait brief period for VRAM release
    5. Proceed to next model
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_ctx: int = 8192,
        unload_delay_ms: float = 100.0
    ):
        """
        Args:
            base_url: Ollama API base URL
            timeout: HTTP timeout for generation
            num_ctx: Context window size
            unload_delay_ms: Delay after unload before next model
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.unload_delay_ms = unload_delay_ms
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
            logger.info(f"[ModelRotator] Initialized, base_url={self.base_url}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def preload_model(self, model: str) -> bool:
        """
        Preload a model into VRAM without generating.

        Uses a minimal prompt to trigger model loading.

        Args:
            model: Model name to preload

        Returns:
            True if preload successful
        """
        if not self._client:
            await self.initialize()

        try:
            # Send minimal request with keep_alive to load model
            payload = {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "5m",  # Keep loaded for 5 minutes
                "options": {
                    "num_predict": 0  # Don't generate anything
                }
            }

            logger.info(f"[ModelRotator] Preloading {model}...")
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"[ModelRotator] {model} preloaded")
            return True

        except Exception as e:
            logger.warning(f"[ModelRotator] Preload failed for {model}: {e}")
            return False

    async def unload_model(self, model: str) -> bool:
        """
        Unload a model from VRAM.

        Sets keep_alive=0 to immediately free VRAM.

        Args:
            model: Model name to unload

        Returns:
            True if unload successful
        """
        if not self._client:
            return False

        try:
            # Send request with keep_alive=0 to unload
            payload = {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,  # Unload immediately
                "options": {
                    "num_predict": 0
                }
            }

            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            logger.debug(f"[ModelRotator] {model} unloaded")

            # Brief delay for VRAM release
            if self.unload_delay_ms > 0:
                import asyncio
                await asyncio.sleep(self.unload_delay_ms / 1000)

            return True

        except Exception as e:
            logger.warning(f"[ModelRotator] Unload failed for {model}: {e}")
            return False

    async def generate_single(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        unload_after: bool = True
    ) -> RotatedResponse:
        """
        Generate with a single model and optionally unload after.

        Args:
            model: Model name
            messages: Chat messages
            temperature: Generation temperature
            unload_after: If True, unload model after generation

        Returns:
            RotatedResponse with result
        """
        if not self._client:
            await self.initialize()

        start_time = time.time()

        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": 0 if unload_after else "5m",
                "options": {
                    "temperature": temperature,
                    "num_predict": 4096,
                    "num_ctx": self.num_ctx,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }

            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract tokens
            tokens = {}
            if "eval_count" in data:
                tokens["completion"] = data["eval_count"]
            if "prompt_eval_count" in data:
                tokens["prompt"] = data["prompt_eval_count"]
            tokens["total"] = tokens.get("prompt", 0) + tokens.get("completion", 0)

            text = data.get("message", {}).get("content", "")

            result = RotatedResponse(
                model=model,
                text=text,
                latency_ms=latency_ms,
                tokens=tokens,
                success=True,
                vram_freed=unload_after
            )

            logger.info(
                f"[ModelRotator] {model} generated {len(text)} chars "
                f"in {latency_ms:.0f}ms (unload={unload_after})"
            )

            # Brief delay if unloading
            if unload_after and self.unload_delay_ms > 0:
                import asyncio
                await asyncio.sleep(self.unload_delay_ms / 1000)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[ModelRotator] {model} failed: {e}")

            return RotatedResponse(
                model=model,
                text="",
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                vram_freed=False
            )

    async def rotate_and_process(
        self,
        models: List[str],
        question: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stop_on_first_success: bool = False
    ) -> RotationResult:
        """
        Process question with multiple models sequentially.

        Loads each model, generates, unloads, then moves to next.
        This prevents VRAM exhaustion when using multiple large models.

        Args:
            models: List of model names to use
            question: User question
            system_prompt: Optional system prompt
            temperature: Generation temperature
            stop_on_first_success: Stop after first successful generation

        Returns:
            RotationResult with all responses
        """
        start_time = time.time()
        responses: Dict[str, RotatedResponse] = {}
        models_failed = 0

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        logger.info(f"[ModelRotator] Starting rotation with {len(models)} models")

        for i, model in enumerate(models):
            logger.info(f"[ModelRotator] Processing model {i+1}/{len(models)}: {model}")

            # Generate with this model (unload after unless it's the last and succeeded)
            is_last = (i == len(models) - 1)
            response = await self.generate_single(
                model=model,
                messages=messages,
                temperature=temperature,
                unload_after=True  # Always unload to free VRAM
            )

            responses[model] = response

            if not response.success:
                models_failed += 1

            if stop_on_first_success and response.success:
                logger.info(f"[ModelRotator] Success with {model}, stopping rotation")
                break

        total_duration_ms = (time.time() - start_time) * 1000

        result = RotationResult(
            responses=responses,
            total_duration_ms=total_duration_ms,
            models_processed=len(responses),
            models_failed=models_failed,
            vram_managed=True
        )

        logger.info(
            f"[ModelRotator] Rotation complete: {result.models_processed} models, "
            f"{models_failed} failed, {total_duration_ms:.0f}ms total"
        )

        return result

    async def batch_process(
        self,
        model: str,
        questions: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        unload_when_done: bool = True
    ) -> List[RotatedResponse]:
        """
        Process multiple questions with a single model.

        Keeps model loaded for the entire batch, then unloads.

        Args:
            model: Model name
            questions: List of questions
            system_prompt: Optional system prompt
            temperature: Generation temperature
            unload_when_done: Unload model after batch

        Returns:
            List of RotatedResponse for each question
        """
        logger.info(f"[ModelRotator] Batch processing {len(questions)} questions with {model}")

        responses = []
        for i, question in enumerate(questions):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})

            # Keep model loaded until last question
            is_last = (i == len(questions) - 1)
            unload = unload_when_done and is_last

            response = await self.generate_single(
                model=model,
                messages=messages,
                temperature=temperature,
                unload_after=unload
            )
            responses.append(response)

        return responses

    async def batch_sequential_models(
        self,
        models: List[str],
        questions: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> BatchModelResult:
        """
        Process multiple questions with multiple models, one model at a time.

        VRAM-OPTIMAL STRATEGY:
        1. For each model:
           a. Preload the model (keep_alive="5m")
           b. Process ALL questions (model stays in VRAM)
           c. Unload the model (keep_alive=0)
        2. Move to next model

        This minimizes VRAM usage and model switching overhead.

        Args:
            models: List of models to use
            questions: List of questions to process
            system_prompt: Optional system prompt
            temperature: Generation temperature

        Returns:
            BatchModelResult with all responses organized by model
        """
        start_time = time.time()
        results: Dict[str, List[RotatedResponse]] = {}

        logger.info(
            f"[ModelRotator] Batch sequential: {len(models)} models × "
            f"{len(questions)} questions"
        )

        for model_idx, model in enumerate(models):
            logger.info(
                f"[ModelRotator] Loading model {model_idx+1}/{len(models)}: {model}"
            )

            # Preload the model BEFORE processing the batch
            await self.preload_model(model)

            # Process ALL questions with this model
            # unload_when_done=True frees VRAM after the complete batch
            model_responses = await self.batch_process(
                model=model,
                questions=questions,
                system_prompt=system_prompt,
                temperature=temperature,
                unload_when_done=True  # Unload after this batch
            )

            results[model] = model_responses

            logger.info(
                f"[ModelRotator] Model {model} done: "
                f"{len(model_responses)} responses, VRAM freed"
            )

        total_duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[ModelRotator] Batch sequential complete: "
            f"{len(models)} models × {len(questions)} questions "
            f"in {total_duration_ms:.0f}ms"
        )

        return BatchModelResult(
            results=results,
            total_duration_ms=total_duration_ms,
            models_processed=len(models),
            questions_per_model=len(questions),
            vram_managed=True
        )


# Singleton instance
_rotator_instance: Optional[ModelRotator] = None


async def get_model_rotator(
    base_url: str = "http://localhost:11434",
    num_ctx: int = 8192
) -> ModelRotator:
    """Get or create singleton ModelRotator instance."""
    global _rotator_instance

    if _rotator_instance is None:
        import os
        actual_url = base_url or os.environ.get("LYRA_OLLAMA_URL", "http://localhost:11434")
        actual_ctx = num_ctx or int(os.environ.get("LYRA_NUM_CTX", "8192"))

        _rotator_instance = ModelRotator(
            base_url=actual_url,
            num_ctx=actual_ctx
        )
        await _rotator_instance.initialize()

    return _rotator_instance


async def close_model_rotator() -> None:
    """Close the singleton ModelRotator."""
    global _rotator_instance
    if _rotator_instance:
        await _rotator_instance.close()
        _rotator_instance = None
