"""
ESMM - Exploration Sémantique Multi-Modèles
============================================

Module principal pour le protocole ESMM de Lyra-ACE.

Phase 1: Fondations
- populate_graph: Chargement des concepts depuis topics.txt
- relation_generator: Génération de relations par similarité d'embeddings
- seed_injector: Injection de la graine sémantique dialectique
- model_rotator: Rotation séquentielle des modèles pour gestion VRAM

Phase 2: Extracteur de triplets (à venir)
Phase 3: Protocole ESMM complet (à venir)

Author: Lyra-ACE ESMM Protocol
Version: 1.1
"""

from .populate_graph import GraphPopulator
from .relation_generator import RelationGenerator
from .seed_injector import SeedInjector
from .model_rotator import (
    ModelRotator,
    get_model_rotator,
    close_model_rotator,
    RotatedResponse,
    RotationResult,
    BatchModelResult
)
from .prompts import (
    CANONICAL_RELATIONS,
    get_triplet_extraction_prompt,
    get_triplet_validation_prompt,
    get_relation_generation_prompt,
    get_concept_extraction_prompt,
    is_canonical_relation,
    normalize_relation
)
from .triplet_validator import (
    TripletValidator,
    ExtractedTriplet,
    ValidationResult,
    validate_triplet_quick,
    extract_and_validate
)

__all__ = [
    # Phase 1: Graph Population
    "GraphPopulator",
    "RelationGenerator",
    "SeedInjector",
    # VRAM Management
    "ModelRotator",
    "get_model_rotator",
    "close_model_rotator",
    "RotatedResponse",
    "RotationResult",
    "BatchModelResult",
    # Prompts
    "CANONICAL_RELATIONS",
    "get_triplet_extraction_prompt",
    "get_triplet_validation_prompt",
    "get_relation_generation_prompt",
    "get_concept_extraction_prompt",
    "is_canonical_relation",
    "normalize_relation",
    # Validation
    "TripletValidator",
    "ExtractedTriplet",
    "ValidationResult",
    "validate_triplet_quick",
    "extract_and_validate"
]
