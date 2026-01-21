"""
LYRA CLEAN - API MODELS
=======================

Pydantic models for request/response validation.

All models are immutable (frozen=True equivalent via Config).
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime


# ============================================================================
# CHAT MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """
    User chat message request.

    Example:
        {
            "text": "What is entropy?",
            "session_id": "abc-123",
            "profile": "creative",
            "enable_context": true,
            "consciousness_level": 1
        }
    """
    text: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(None, description="Session UUID (auto-created if None)")
    profile: str = Field("balanced", description="Bezier profile name")
    enable_context: bool = Field(True, description="Enable semantic context injection")
    max_history: int = Field(50, ge=0, le=500, description="Max conversation history messages")
    consciousness_level: int = Field(0, ge=0, le=3, description="Consciousness level: 0=off, 1=passive, 2=adaptive, 3=full")

    model_config = ConfigDict(frozen=False)  # Allow mutation for backward compat
    
    @field_validator('consciousness_level')
    @classmethod
    def validate_consciousness_level(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("consciousness_level must be 0, 1, 2, or 3")
        return v


class ChatResponse(BaseModel):
    """
    LLM response with metadata.

    Example:
        {
            "text": "Entropy is a measure...",
            "session_id": "abc-123",
            "physics_state": {...},
            "context": {...},
            "latency": {...},
            "consciousness": {...}
        }
    """
    text: str = Field(..., description="LLM generated response")
    session_id: str = Field(..., description="Session UUID")
    model: Optional[str] = Field(None, description="LLM model name used for generation")

    # Physics state at generation time
    physics_state: Dict[str, float] = Field(..., description="τc, ρ, δr, κ at time t")

    # Context metadata
    context: Optional[Dict[str, Any]] = Field(None, description="Semantic context metadata")

    # Performance metrics
    latency: Dict[str, float] = Field(
        ...,
        description="Latency breakdown (ms): context_extraction, llm_generation, total"
    )

    # Token estimates
    tokens: Dict[str, int] = Field(
        default_factory=dict,
        description="Token counts: prompt, completion, total (approximate)"
    )

    # Consciousness metrics (Phase 1+)
    consciousness: Optional[Dict[str, float]] = Field(
        None,
        description="Epistemological metrics if consciousness_level >= 1: coherence, tension, fit, pressure, stability_score"
    )

    # Semantic memory echo (Phase 3)
    memory_echo: Optional[str] = Field(
        None,
        description="Formatted memory recall from semantic memory if consciousness_level >= 3"
    )

    model_config = ConfigDict(frozen=True)


# ============================================================================
# SESSION MODELS
# ============================================================================

class SessionCreateRequest(BaseModel):
    """
    Create new session request.

    Example:
        {
            "profile": "creative",
            "metadata": {"user_id": "user-123"}
        }
    """
    profile: str = Field("balanced", description="Initial Bezier profile")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")

    model_config = ConfigDict(frozen=False)


class SessionResponse(BaseModel):
    """
    Session information.

    Example:
        {
            "session_id": "abc-123",
            "profile": "creative",
            "created_at": "2025-01-01T12:00:00Z",
            "message_count": 15
        }
    """
    session_id: str
    profile: str
    created_at: str  # ISO format
    last_activity: str  # ISO format
    message_count: int
    total_tokens: int = 0

    model_config = ConfigDict(frozen=True)


class SessionHistoryResponse(BaseModel):
    """
    Session conversation history.

    Example:
        {
            "session_id": "abc-123",
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": "..."},
                {"role": "assistant", "content": "Hi!", "timestamp": "..."}
            ]
        }
    """
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int

    model_config = ConfigDict(frozen=True)


# ============================================================================
# PROFILE MODELS
# ============================================================================

class ProfileResponse(BaseModel):
    """
    Bezier profile configuration.

    Example:
        {
            "profile_name": "creative",
            "description": "High exploration...",
            "tau_c_curve": [[0, 1.3], [0.3, 1.5], ...],
            "preview": [
                {"t": 0.0, "tau_c": 1.30, "rho": 0.50},
                {"t": 0.5, "tau_c": 1.45, "rho": 0.65}
            ]
        }
    """
    profile_name: str
    description: str
    tau_c_curve: List[List[float]]
    rho_curve: List[List[float]]
    delta_r_curve: List[List[float]]
    kappa_curve: Optional[List[List[float]]] = None
    is_default: bool = False

    # Optional: trajectory preview
    preview: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Sampled trajectory points for visualization"
    )

    model_config = ConfigDict(frozen=True)


class ProfileListResponse(BaseModel):
    """
    List of available profiles.

    Example:
        {
            "profiles": [
                {"name": "balanced", "description": "...", "is_default": true},
                {"name": "creative", "description": "...", "is_default": false}
            ]
        }
    """
    profiles: List[Dict[str, Any]]

    model_config = ConfigDict(frozen=True)


# ============================================================================
# SYSTEM MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """
    System health check.

    Example:
        {
            "status": "healthy",
            "database": {"connected": true, "concepts": 15234},
            "ollama": {"connected": true, "model": "gpt-oss:20b"},
            "version": "1.0.0"
        }
    """
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    database: Dict[str, Any]
    ollama: Dict[str, Any]
    version: str
    uptime_seconds: float = 0.0

    model_config = ConfigDict(frozen=True)


class StatsResponse(BaseModel):
    """
    System statistics.

    Example:
        {
            "database": {"concepts": 15234, "relations": 245678, "sessions": 42},
            "performance": {"avg_response_time_ms": 850, "requests_total": 1523}
        }
    """
    database: Dict[str, Any]
    performance: Dict[str, float]
    cache: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(frozen=True)


# ============================================================================
# ERROR MODELS
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Standard error response.

    Example:
        {
            "error": "session_not_found",
            "message": "Session abc-123 does not exist",
            "details": {...}
        }
    """
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    model_config = ConfigDict(frozen=True)


# ============================================================================
# MULTI-MODEL MODELS (Lyra-ACE)
# ============================================================================

class MultiModelRequest(BaseModel):
    """
    Requête de génération multi-modèles.

    Example:
        {
            "text": "Explain quantum entanglement",
            "models": ["llama3.1:8b", "mistral"],
            "session_id": "abc-123",
            "profile": "analytical",
            "stop_on_first_success": false
        }
    """
    text: str = Field(..., min_length=1, max_length=10000)
    models: List[str] = Field(..., min_length=1, max_length=5)
    session_id: Optional[str] = None
    profile: str = Field("balanced")
    stop_on_first_success: bool = Field(False)

    model_config = ConfigDict(frozen=False)


class MultiModelResponseItem(BaseModel):
    """Réponse d'un modèle individuel."""
    model: str
    text: str
    latency_ms: float
    tokens: Dict[str, int]
    success: bool
    error: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class ConsensusMetricsModel(BaseModel):
    """Métriques de consensus."""
    num_models: int
    num_successful: int
    response_lengths: List[int]
    length_variance: float
    avg_latency_ms: float
    model_weights: Dict[str, float]

    model_config = ConfigDict(frozen=True)


class MultiModelResponse(BaseModel):
    """
    Réponse agrégée multi-modèles.

    Example:
        {
            "best_response": "Quantum entanglement is...",
            "best_model": "llama3.1:8b",
            "responses": {...},
            "consensus": {...},
            "session_id": "abc-123"
        }
    """
    best_response: str
    best_model: str
    responses: Dict[str, MultiModelResponseItem]
    consensus: ConsensusMetricsModel
    session_id: str
    physics_state: Dict[str, float]

    model_config = ConfigDict(frozen=True)


# ============================================================================
# GRAPH DELTA MODELS (Lyra-ACE)
# ============================================================================

class GraphDeltaRequest(BaseModel):
    """
    Requête d'application d'un delta.

    Example:
        {
            "operation": "add_edge",
            "source": "entropy",
            "target": "information",
            "weight": 0.85,
            "confidence": 0.9,
            "reason": "Strong semantic relationship"
        }
    """
    operation: str = Field(..., pattern="^(add_node|add_edge|update_edge|delete_edge|delete_node)$")
    source: str = Field(..., min_length=1)
    target: Optional[str] = None
    weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    model_source: str = Field("api")
    reason: Optional[str] = None

    model_config = ConfigDict(frozen=False)


class GraphDeltaResponse(BaseModel):
    """Réponse après application d'un delta."""
    delta_id: int
    operation: str
    source: str
    target: Optional[str]
    old_weight: Optional[float]
    new_weight: Optional[float]
    old_kappa: Optional[float]
    new_kappa: Optional[float]
    applied_at: float

    model_config = ConfigDict(frozen=True)


class KappaResponse(BaseModel):
    """Réponse du calcul de κ."""
    source: str
    target: str
    kappa_ollivier: float
    kappa_jaccard: float
    kappa_hybrid: float
    alpha: float

    model_config = ConfigDict(frozen=True)


# ============================================================================
# UTILITIES
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (4 chars ≈ 1 token).

    Args:
        text: Input text

    Returns:
        Approximate token count
    """
    return len(text) // 4


def format_timestamp(unix_time: float) -> str:
    """
    Format Unix timestamp as ISO 8601.

    Args:
        unix_time: Unix timestamp (seconds)

    Returns:
        ISO format string (e.g., "2025-01-01T12:00:00Z")
    """
    return datetime.fromtimestamp(unix_time).isoformat() + "Z"
