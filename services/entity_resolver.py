"""
LYRA-ACE - ENTITY RESOLVER
==========================

Resout les entites (concepts) vers leur forme canonique.
Detecte les doublons semantiques via embeddings.

Seuils:
- SIMILARITY_THRESHOLD (0.92): Fusion automatique
- REVIEW_THRESHOLD (0.85): Candidat a la revue manuelle
"""
from __future__ import annotations

import struct
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from database.engine import ISpaceDB, get_db


# Seuils de similarite
SIMILARITY_THRESHOLD = 0.92  # Au-dessus: fusion automatique
REVIEW_THRESHOLD = 0.85      # Entre REVIEW et SIMILARITY: candidat revue


@dataclass
class ResolutionResult:
    """Resultat de resolution d'entite."""
    canonical_id: str           # Concept canonique
    is_new: bool               # True si nouveau concept cree
    is_alias: bool             # True si alias d'un existant
    similarity: Optional[float] # Score de similarite si alias
    method: str                # 'exact' | 'embedding' | 'new'


class EntityResolver:
    """
    Resolveur d'entites semantiques.

    Utilise les embeddings pour detecter les doublons
    et fusionner les concepts similaires.
    """

    def __init__(self, db: ISpaceDB):
        self.db = db

    async def resolve(
        self,
        concept: str,
        auto_create: bool = True
    ) -> ResolutionResult:
        """
        Resout un concept vers sa forme canonique.

        Args:
            concept: Concept a resoudre
            auto_create: Si True, cree le concept s'il n'existe pas

        Returns:
            ResolutionResult avec le concept canonique

        Strategie:
        1. Verifier les aliases existants (exact match)
        2. Verifier le concept directement (exact match)
        3. Chercher par similarite d'embedding
        4. Creer si nouveau (et auto_create=True)
        """
        concept_normalized = self._normalize(concept)

        # 1. Verifier les aliases existants
        canonical = await self.db.resolve_concept(concept_normalized)
        if canonical != concept_normalized:
            return ResolutionResult(
                canonical_id=canonical,
                is_new=False,
                is_alias=True,
                similarity=1.0,
                method="exact"
            )

        # 2. Verifier si le concept existe directement
        existing = await self.db.get_concept(concept_normalized)
        if existing:
            return ResolutionResult(
                canonical_id=concept_normalized,
                is_new=False,
                is_alias=False,
                similarity=1.0,
                method="exact"
            )

        # 3. Chercher par similarite d'embedding
        similar = await self._find_similar(concept_normalized, top_k=5)
        if similar:
            canonical_id, similarity = similar
            if similarity >= SIMILARITY_THRESHOLD:
                # Fusion automatique
                await self.db.add_alias(
                    alias=concept_normalized,
                    canonical_id=canonical_id,
                    similarity=similarity,
                    method="embedding"
                )
                return ResolutionResult(
                    canonical_id=canonical_id,
                    is_new=False,
                    is_alias=True,
                    similarity=similarity,
                    method="embedding"
                )

        # 4. Creer si nouveau
        if auto_create:
            await self._create_concept(concept_normalized)
            return ResolutionResult(
                canonical_id=concept_normalized,
                is_new=True,
                is_alias=False,
                similarity=None,
                method="new"
            )

        # Pas de creation, retourner quand meme
        return ResolutionResult(
            canonical_id=concept_normalized,
            is_new=False,
            is_alias=False,
            similarity=None,
            method="not_found"
        )

    async def _find_similar(
        self,
        concept: str,
        top_k: int = 5
    ) -> Optional[Tuple[str, float]]:
        """
        Trouve le concept le plus similaire par embedding.

        Returns:
            (canonical_id, similarity) ou None
        """
        # Obtenir l'embedding du concept cible
        from app.embeddings import get_embedding
        target_embedding = await get_embedding(concept)
        if not target_embedding:
            return None

        # Recuperer tous les concepts avec embeddings
        concepts_with_emb = await self.db.get_concepts_with_embeddings(limit=1000)
        if not concepts_with_emb:
            return None

        best_match = None
        best_similarity = 0.0

        for c in concepts_with_emb:
            if not c.get("embedding"):
                continue

            emb = self._deserialize_embedding(c["embedding"])
            if not emb:
                continue

            similarity = self._cosine_similarity(target_embedding, emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = c["id"]

        if best_match and best_similarity >= REVIEW_THRESHOLD:
            return (best_match, best_similarity)

        return None

    async def _create_concept(self, concept: str) -> None:
        """Cree un nouveau concept avec embedding."""
        from app.embeddings import get_embedding
        embedding = await get_embedding(concept)

        await self.db.add_concept(
            concept_id=concept,
            rho_static=0.0,
            embedding=self._serialize_embedding(embedding) if embedding else None,
            source="extracted"
        )

    def _normalize(self, concept: str) -> str:
        """Normalise un concept (lowercase, strip, etc.)."""
        return concept.lower().strip()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calcule la similarite cosinus entre deux vecteurs."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialise un embedding en bytes (float32)."""
        return struct.pack(f'{len(embedding)}f', *embedding)

    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialise un embedding depuis bytes."""
        n_floats = len(data) // 4
        return list(struct.unpack(f'{n_floats}f', data))


# ============================================================================
# SINGLETON
# ============================================================================

_resolver_instance: Optional[EntityResolver] = None


async def get_entity_resolver() -> EntityResolver:
    """
    Retourne l'instance singleton du resolveur.
    """
    global _resolver_instance
    if _resolver_instance is None:
        db = await get_db()
        _resolver_instance = EntityResolver(db)
    return _resolver_instance
