"""
LYRA-ACE - RELATION NORMALIZER
==============================

Normalise les types de relations vers les 20 formes canoniques.
Gere les inverses et la symetrie.
"""
from __future__ import annotations

import json
from typing import Optional, Dict, List

from database.engine import ISpaceDB, get_db


class RelationNormalizer:
    """
    Normaliseur de relations semantiques.

    Convertit les relations brutes (ex: "provoque", "engendre")
    vers leurs formes canoniques (ex: "cause").
    """

    def __init__(self, db: ISpaceDB):
        self.db = db
        self._cache: Dict[str, Dict] = {}  # canonical -> metadata
        self._alias_cache: Dict[str, str] = {}  # alias -> canonical
        self._loaded = False

    async def _load_cache(self) -> None:
        """Charge le cache des relations canoniques."""
        if self._loaded:
            return

        relations = await self.db.get_all_canonical_relations()
        for rel in relations:
            canonical = rel["canonical"]
            self._cache[canonical] = rel

            # Ajouter les aliases
            aliases = json.loads(rel["aliases"])
            for alias in aliases:
                self._alias_cache[alias.lower()] = canonical

            # Le canonique pointe vers lui-meme
            self._alias_cache[canonical] = canonical

        self._loaded = True

    async def normalize(self, relation: str) -> str:
        """
        Normalise une relation vers sa forme canonique.

        Args:
            relation: Relation brute

        Returns:
            Relation canonique ou 'related_to' si non trouvee
        """
        await self._load_cache()

        relation_lower = relation.lower().strip()

        # Chercher dans le cache des aliases
        if relation_lower in self._alias_cache:
            return self._alias_cache[relation_lower]

        # Non trouve -> default
        return "related_to"

    async def get_inverse(self, relation: str) -> Optional[str]:
        """
        Retourne la relation inverse si elle existe.

        Ex: "cause" -> "caused_by"
        """
        await self._load_cache()

        # Normaliser d'abord
        canonical = await self.normalize(relation)

        if canonical in self._cache:
            return self._cache[canonical].get("inverse")

        return None

    async def is_symmetric(self, relation: str) -> bool:
        """
        Verifie si une relation est symetrique.

        Ex: "related_to" est symetrique (A related_to B = B related_to A)
        """
        await self._load_cache()

        # Normaliser d'abord
        canonical = await self.normalize(relation)

        if canonical in self._cache:
            return bool(self._cache[canonical].get("symmetric", 0))

        return False

    async def get_category(self, relation: str) -> str:
        """
        Retourne la categorie d'une relation.

        Categories: causal, hierarchical, associative, property,
                   temporal, epistemic, transformational, comparative
        """
        await self._load_cache()

        canonical = await self.normalize(relation)

        if canonical in self._cache:
            return self._cache[canonical].get("category", "unknown")

        return "unknown"

    async def get_all_relations(self) -> List[Dict]:
        """
        Retourne toutes les relations canoniques avec metadonnees.
        """
        await self._load_cache()
        return list(self._cache.values())


# ============================================================================
# SINGLETON
# ============================================================================

_normalizer_instance: Optional[RelationNormalizer] = None


async def get_relation_normalizer() -> RelationNormalizer:
    """
    Retourne l'instance singleton du normaliseur.
    """
    global _normalizer_instance
    if _normalizer_instance is None:
        db = await get_db()
        _normalizer_instance = RelationNormalizer(db)
    return _normalizer_instance
