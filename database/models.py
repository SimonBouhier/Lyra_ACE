"""
LYRA-ACE - DATABASE PYDANTIC MODELS
===================================

Modeles Pydantic pour validation stricte des donnees.
Assure l'integrite des donnees entrant/sortant de la base.

Usage:
    concept = ConceptModel(id="entropy", rho_static=0.5)
    relation = RelationModel(source="entropy", target="information", weight=0.85)
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# ENUMS
# ============================================================================

class ConceptSource(str, Enum):
    """Sources possibles pour un concept."""
    MANUAL = "manual"
    SEED = "seed"
    EXTRACTED = "extracted"
    MERGED = "merged"


class RelationType(str, Enum):
    """Types de relations canoniques."""
    CAUSE = "cause"
    CAUSED_BY = "caused_by"
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DIFFERENT_FROM = "different_from"
    HAS_PROPERTY = "has_property"
    USED_FOR = "used_for"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    COOCCURS_WITH = "cooccurs_with"
    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    REQUIRES = "requires"
    TRANSFORMS_INTO = "transforms_into"
    PRODUCES = "produces"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUIVALENT_TO = "equivalent_to"


class KappaMethod(str, Enum):
    """Methodes de calcul kappa."""
    JACCARD = "jaccard"
    OLLIVIER = "ollivier"
    HYBRID = "hybrid"


class EpistemicType(str, Enum):
    """Types epistemiques pour la cochaine."""
    GENERALIST = "generalist"
    SPECIALIZED = "specialized"
    HYBRID = "hybrid"


class GapType(str, Enum):
    """Types de lacunes de connaissances."""
    ISOLATED = "isolated"
    UNSTABLE = "unstable"
    BRIDGE = "bridge"


class ESMMStatus(str, Enum):
    """Statuts d'un run ESMM."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class CycleType(str, Enum):
    """Types de cycles ESMM."""
    DIVERGENT = "divergent"
    DEBATE = "debate"
    META = "meta"


# ============================================================================
# BASE MODELS
# ============================================================================

class LyraBaseModel(BaseModel):
    """Modele de base avec configuration commune."""

    model_config = {
        "from_attributes": True,
        "validate_assignment": True,
        "extra": "forbid"
    }


# ============================================================================
# CONCEPT MODELS
# ============================================================================

class ConceptCreate(LyraBaseModel):
    """Modele pour creer un concept."""
    id: str = Field(..., min_length=1, max_length=255, description="Identifiant unique")
    rho_static: float = Field(default=0.0, ge=0.0, le=1.0, description="Densite statique")
    source: ConceptSource = Field(default=ConceptSource.MANUAL)
    first_seen_model: Optional[str] = Field(default=None, max_length=100)

    @field_validator('id')
    @classmethod
    def normalize_id(cls, v: str) -> str:
        return v.lower().strip()


class ConceptModel(LyraBaseModel):
    """Modele complet d'un concept."""
    id: str
    rho_static: float = Field(ge=0.0, le=1.0)
    degree: int = Field(ge=0)
    embedding: Optional[bytes] = None
    embedding_model: Optional[str] = None
    embedding_updated_at: Optional[float] = None
    source: ConceptSource = ConceptSource.MANUAL
    first_seen_model: Optional[str] = None
    created_at: float
    last_accessed: Optional[float] = None
    access_count: int = Field(ge=0, default=0)
    aliases: List[str] = Field(default_factory=list)


class ConceptSummary(LyraBaseModel):
    """Modele leger pour les listes."""
    id: str
    rho_static: float
    degree: int
    access_count: int = 0


# ============================================================================
# RELATION MODELS
# ============================================================================

class RelationCreate(LyraBaseModel):
    """Modele pour creer une relation."""
    source: str = Field(..., min_length=1, max_length=255)
    target: str = Field(..., min_length=1, max_length=255)
    weight: float = Field(default=0.0, ge=0.0)
    relation_type: str = Field(default="related_to")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_source: str = Field(default="system", max_length=100)

    @field_validator('source', 'target')
    @classmethod
    def normalize_concepts(cls, v: str) -> str:
        return v.lower().strip()

    @model_validator(mode='after')
    def check_not_self_relation(self):
        if self.source == self.target:
            raise ValueError("Source and target cannot be the same")
        return self


class RelationModel(LyraBaseModel):
    """Modele complet d'une relation."""
    source: str
    target: str
    weight: float = Field(ge=0.0)
    kappa: float = Field(ge=0.0, le=1.0, default=0.5)
    kappa_method: KappaMethod = KappaMethod.JACCARD
    relation_type: str = "related_to"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    model_source: str = "system"
    extraction_count: int = Field(ge=1, default=1)
    created_at: float
    updated_at: Optional[float] = None


class NeighborModel(LyraBaseModel):
    """Modele pour un voisin semantique."""
    target: str
    weight: float
    kappa: float


# ============================================================================
# ALIAS MODELS
# ============================================================================

class AliasCreate(LyraBaseModel):
    """Modele pour creer un alias."""
    alias: str = Field(..., min_length=1, max_length=255)
    canonical_id: str = Field(..., min_length=1, max_length=255)
    similarity: float = Field(ge=0.0, le=1.0)
    fusion_method: str = Field(default="embedding")

    @field_validator('alias', 'canonical_id')
    @classmethod
    def normalize(cls, v: str) -> str:
        return v.lower().strip()


class AliasModel(LyraBaseModel):
    """Modele complet d'un alias."""
    alias: str
    canonical_id: str
    similarity: float
    fusion_method: str
    created_at: float
    created_by: str = "system"


# ============================================================================
# SESSION MODELS
# ============================================================================

class SessionCreate(LyraBaseModel):
    """Modele pour creer une session."""
    session_id: str = Field(..., min_length=36, max_length=36)  # UUID format
    profile: str = Field(default="balanced", max_length=50)
    params_snapshot: Optional[Dict[str, Any]] = None


class SessionModel(LyraBaseModel):
    """Modele complet d'une session."""
    session_id: str
    profile: str
    params_snapshot: Optional[Dict[str, Any]] = None
    message_count: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0, default=0)
    created_at: float
    last_activity: float


class SessionSummary(LyraBaseModel):
    """Resume de session pour les listes."""
    session_id: str
    profile: str
    message_count: int
    last_activity: float


# ============================================================================
# ESMM MODELS
# ============================================================================

class ESMMRunCreate(LyraBaseModel):
    """Modele pour creer un run ESMM."""
    config: Dict[str, Any]
    models_used: List[str] = Field(min_length=1)
    seed_type: str = Field(default="standard")


class ESMMRunModel(LyraBaseModel):
    """Modele complet d'un run ESMM."""
    run_id: int
    config: Dict[str, Any]
    models_used: List[str]
    seed_type: str
    status: ESMMStatus
    current_cycle: Optional[CycleType] = None
    current_iteration: int = 0
    cycles_completed: int = 0
    total_questions: int = 0
    total_triplets: int = 0
    triplets_injected: int = 0
    concepts_created: int = 0
    relations_created: int = 0
    final_cochain_size: Optional[int] = None
    coverage_score: Optional[float] = None
    consensus_density: Optional[float] = None
    epistemic_diversity: Optional[float] = None
    structural_stability: Optional[float] = None
    started_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None


class TripletCreate(LyraBaseModel):
    """Modele pour un triplet extrait."""
    subject: str = Field(..., min_length=1, max_length=255)
    relation: str = Field(..., min_length=1, max_length=100)
    object: str = Field(..., min_length=1, max_length=255)
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str
    model_source: str
    cycle_id: Optional[int] = None
    event_id: Optional[int] = None
    source_text: Optional[str] = Field(default=None, max_length=100)


class TripletModel(LyraBaseModel):
    """Modele complet d'un triplet."""
    extraction_id: int
    cycle_id: Optional[int]
    event_id: Optional[int]
    subject: str
    subject_canonical: Optional[str]
    relation: str
    relation_canonical: Optional[str]
    object: str
    object_canonical: Optional[str]
    confidence: float
    extraction_method: str
    model_source: str
    source_text: Optional[str]
    injected_to_graph: bool = False
    delta_id: Optional[int] = None
    injection_skipped_reason: Optional[str] = None
    extracted_at: float


class CochainEntryCreate(LyraBaseModel):
    """Modele pour une entree de cochaine."""
    concept_id: str
    consensus_score: float = Field(ge=0.0, le=1.0)
    model_agreement: float = Field(ge=0.0, le=1.0)
    semantic_consistency: float = Field(ge=0.0, le=1.0)
    structural_centrality: float = Field(ge=0.0, le=1.0)
    stability_score: float = Field(ge=0.0, le=1.0)
    signature_vector: List[float] = Field(min_length=5, max_length=5)
    epistemic_type: EpistemicType
    contributing_models: Dict[str, float]
    triplet_count: int = Field(ge=0)
    run_id: Optional[int] = None


class CochainEntryModel(CochainEntryCreate):
    """Modele complet avec timestamp."""
    computed_at: float
    protocol_version: str = "v2"


class KnowledgeGapCreate(LyraBaseModel):
    """Modele pour une lacune de connaissance."""
    gap_type: GapType
    details: Dict[str, Any]
    priority: float = Field(ge=0.0)
    run_id: Optional[int] = None


class KnowledgeGapModel(LyraBaseModel):
    """Modele complet d'une lacune."""
    gap_id: int
    run_id: Optional[int]
    gap_type: GapType
    details: Dict[str, Any]
    priority: float
    addressed: bool = False
    addressed_by_cycle_id: Optional[int] = None
    detected_at: float
    addressed_at: Optional[float] = None


# ============================================================================
# STATISTICS MODELS
# ============================================================================

class DatabaseStats(LyraBaseModel):
    """Statistiques de la base de donnees."""
    concepts: int = Field(ge=0)
    relations: int = Field(ge=0)
    sessions: int = Field(ge=0)
    events: int = Field(ge=0)
    esmm_runs: int = Field(ge=0)
    triplets: int = Field(ge=0)
    cochain_entries: int = Field(ge=0)
    knowledge_gaps: int = Field(ge=0)
    db_size_mb: float = Field(ge=0.0)


class PoolStats(LyraBaseModel):
    """Statistiques du pool de connexions."""
    total: int
    in_use: int
    available: int
    overflow: int
    initialized: bool


class CacheStats(LyraBaseModel):
    """Statistiques du cache."""
    size: int
    hits: int
    misses: int
    hit_rate: str


class PerformanceStats(LyraBaseModel):
    """Statistiques de performance globales."""
    pool: PoolStats
    cache: CacheStats
    uptime_seconds: float
