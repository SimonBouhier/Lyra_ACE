"""
Lyra Clean Database Package

Exports:
- ISpaceDB: Main database engine
- get_db: Singleton accessor
- Graph delta management (Lyra-ACE)
- Connection pool & performance utilities
- Pydantic models for validation
"""
from .engine import ISpaceDB, get_db, close_db
from .graph_delta import (
    GraphDelta,
    DeltaBatch,
    DeltaOperation,
    KappaCalculator,
    DeltaValidationError,
    MutationLimitExceededError
)
from .pool import (
    SQLiteConnectionPool,
    ConceptCache,
    ConcurrencyLimiter,
    SQLValidator,
    get_pool,
    get_concept_cache,
    get_concurrency_limiter,
    close_pool,
    get_logger,
    configure_logging
)
from .models import (
    # Enums
    ConceptSource,
    RelationType,
    KappaMethod,
    EpistemicType,
    GapType,
    ESMMStatus,
    CycleType,
    # Concept models
    ConceptCreate,
    ConceptModel,
    ConceptSummary,
    # Relation models
    RelationCreate,
    RelationModel,
    NeighborModel,
    # Alias models
    AliasCreate,
    AliasModel,
    # Session models
    SessionCreate,
    SessionModel,
    SessionSummary,
    # ESMM models
    ESMMRunCreate,
    ESMMRunModel,
    TripletCreate,
    TripletModel,
    CochainEntryCreate,
    CochainEntryModel,
    KnowledgeGapCreate,
    KnowledgeGapModel,
    # Stats models
    DatabaseStats,
    PoolStats,
    CacheStats,
    PerformanceStats
)

__all__ = [
    # Engine
    "ISpaceDB",
    "get_db",
    "close_db",
    # Graph Delta
    "GraphDelta",
    "DeltaBatch",
    "DeltaOperation",
    "KappaCalculator",
    "DeltaValidationError",
    "MutationLimitExceededError",
    # Pool & Performance
    "SQLiteConnectionPool",
    "ConceptCache",
    "ConcurrencyLimiter",
    "SQLValidator",
    "get_pool",
    "get_concept_cache",
    "get_concurrency_limiter",
    "close_pool",
    "get_logger",
    "configure_logging",
    # Enums
    "ConceptSource",
    "RelationType",
    "KappaMethod",
    "EpistemicType",
    "GapType",
    "ESMMStatus",
    "CycleType",
    # Models
    "ConceptCreate",
    "ConceptModel",
    "ConceptSummary",
    "RelationCreate",
    "RelationModel",
    "NeighborModel",
    "AliasCreate",
    "AliasModel",
    "SessionCreate",
    "SessionModel",
    "SessionSummary",
    "ESMMRunCreate",
    "ESMMRunModel",
    "TripletCreate",
    "TripletModel",
    "CochainEntryCreate",
    "CochainEntryModel",
    "KnowledgeGapCreate",
    "KnowledgeGapModel",
    "DatabaseStats",
    "PoolStats",
    "CacheStats",
    "PerformanceStats"
]
