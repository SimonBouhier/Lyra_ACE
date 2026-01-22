"""
ESMM Phase 1 - L1.2: Relation Generator
========================================

Génère des relations initiales basées sur la similarité d'embeddings.

Usage:
    from services.esmm import RelationGenerator

    generator = RelationGenerator(db)
    result = await generator.generate_initial_relations(similarity_threshold=0.6)

Optimisation:
- Utilise numpy pour le calcul vectorisé des similarités
- Évite O(n²) via seuil de similarité et limite de voisins

Author: Lyra-ACE ESMM Protocol
"""

import time
import struct
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import math

from database.graph_delta import GraphDelta, DeltaOperation

logger = logging.getLogger(__name__)


@dataclass
class RelationGenerationResult:
    """Résultat de la génération de relations."""
    relations_created: int
    relations_skipped: int
    concepts_processed: int
    average_similarity: float
    duration_ms: float
    errors: List[str]


class RelationGenerator:
    """
    Génère des relations basées sur la similarité des embeddings.

    Stratégie:
    1. Charger tous les concepts avec embeddings
    2. Pour chaque concept, calculer la similarité avec tous les autres
    3. Créer des arêtes pour les paires au-dessus du seuil
    4. Utiliser le poids = similarité pour le PPMI initial

    Optimisations:
    - Calcul vectorisé avec numpy si disponible
    - Limite du nombre de voisins par concept (top-k)
    - Relations bidirectionnelles via une seule insertion
    """

    def __init__(self, db, max_neighbors: int = 20):
        """
        Args:
            db: Instance ISpaceDB
            max_neighbors: Nombre maximum de voisins par concept
        """
        self.db = db
        self.max_neighbors = max_neighbors
        self._numpy_available = self._check_numpy()

    def _check_numpy(self) -> bool:
        """Vérifie si numpy est disponible."""
        try:
            import numpy as np
            return True
        except ImportError:
            logger.warning("[RelationGenerator] numpy not available, using pure Python")
            return False

    def deserialize_embedding(self, data: bytes) -> List[float]:
        """Désérialise un embedding depuis bytes."""
        if data is None:
            return []
        count = len(data) // 4  # float32 = 4 bytes
        return list(struct.unpack(f'{count}f', data))

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.

        Args:
            v1, v2: Vecteurs de même dimension

        Returns:
            Similarité ∈ [-1, 1]
        """
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def cosine_similarity_batch(
        self,
        target: List[float],
        embeddings: List[Tuple[str, List[float]]]
    ) -> List[Tuple[str, float]]:
        """
        Calcule la similarité cosinus entre un vecteur cible et plusieurs vecteurs.

        Args:
            target: Vecteur cible
            embeddings: Liste de (concept_id, embedding)

        Returns:
            Liste de (concept_id, similarity) triée par similarité décroissante
        """
        if self._numpy_available:
            import numpy as np

            target_np = np.array(target)
            target_norm = np.linalg.norm(target_np)

            if target_norm == 0:
                return []

            similarities = []
            for concept_id, emb in embeddings:
                emb_np = np.array(emb)
                emb_norm = np.linalg.norm(emb_np)
                if emb_norm == 0:
                    continue
                sim = np.dot(target_np, emb_np) / (target_norm * emb_norm)
                similarities.append((concept_id, float(sim)))
        else:
            # Pure Python fallback
            similarities = [
                (cid, self.cosine_similarity(target, emb))
                for cid, emb in embeddings
            ]

        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    async def generate_initial_relations(
        self,
        similarity_threshold: float = 0.6,
        confidence: float = 0.7,
        relation_type: str = "similar_to",
        model_source: str = "embedding_similarity",
        limit_concepts: Optional[int] = None
    ) -> RelationGenerationResult:
        """
        Génère les relations initiales pour tous les concepts.

        Args:
            similarity_threshold: Seuil minimum de similarité [0, 1]
            confidence: Confiance attribuée aux relations auto-générées
            relation_type: Type de relation (canonique)
            model_source: Source pour traçabilité
            limit_concepts: Limite de concepts à traiter (None = tous)

        Returns:
            RelationGenerationResult avec statistiques
        """
        start_time = time.time()
        errors: List[str] = []

        # Charger tous les concepts avec embeddings
        logger.info("[RelationGenerator] Loading concepts with embeddings...")
        concepts_data = await self.db.get_concepts_with_embeddings(
            limit=limit_concepts or 10000
        )

        if not concepts_data:
            return RelationGenerationResult(
                relations_created=0,
                relations_skipped=0,
                concepts_processed=0,
                average_similarity=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                errors=["No concepts with embeddings found"]
            )

        # Désérialiser les embeddings
        embeddings: List[Tuple[str, List[float]]] = []
        for concept in concepts_data:
            if concept["embedding"]:
                emb = self.deserialize_embedding(concept["embedding"])
                if emb:
                    embeddings.append((concept["id"], emb))

        logger.info(f"[RelationGenerator] {len(embeddings)} concepts with valid embeddings")

        # Charger les relations existantes pour éviter les doublons
        existing_relations: Set[Tuple[str, str]] = set()
        async with self.db.connection() as conn:
            cursor = await conn.execute("SELECT source, target FROM relations")
            for row in await cursor.fetchall():
                existing_relations.add((row[0], row[1]))
                existing_relations.add((row[1], row[0]))  # Bidirectionnel

        logger.info(f"[RelationGenerator] {len(existing_relations)//2} existing relations")

        # Stats
        relations_created = 0
        relations_skipped = 0
        total_similarity = 0.0
        similarity_count = 0

        # Pour chaque concept, trouver les voisins similaires
        for i, (concept_id, embedding) in enumerate(embeddings):
            # Autres concepts (exclure soi-même)
            others = [(cid, emb) for cid, emb in embeddings if cid != concept_id]

            # Calculer les similarités
            similarities = self.cosine_similarity_batch(embedding, others)

            # Prendre les top-k au-dessus du seuil
            neighbors_added = 0
            for neighbor_id, sim in similarities:
                if sim < similarity_threshold:
                    break  # Les suivants sont encore plus bas

                if neighbors_added >= self.max_neighbors:
                    break

                # Vérifier si la relation existe déjà
                if (concept_id, neighbor_id) in existing_relations:
                    relations_skipped += 1
                    continue

                # Créer la relation via delta
                try:
                    # Assurer que les deux concepts existent
                    # (normalement oui car ils viennent de la même requête)
                    delta = GraphDelta(
                        operation=DeltaOperation.ADD_EDGE,
                        source=concept_id,
                        target=neighbor_id,
                        weight=sim,  # Utiliser la similarité comme poids
                        confidence=confidence,
                        model_source=model_source,
                        reason=f"Embedding similarity: {sim:.3f}"
                    )

                    await self.db.apply_delta(delta)
                    relations_created += 1
                    neighbors_added += 1

                    # Marquer comme existante pour éviter doublons inverses
                    existing_relations.add((concept_id, neighbor_id))
                    existing_relations.add((neighbor_id, concept_id))

                    # Stats
                    total_similarity += sim
                    similarity_count += 1

                except Exception as e:
                    logger.warning(
                        f"[RelationGenerator] Failed to create relation "
                        f"{concept_id} -> {neighbor_id}: {e}"
                    )
                    errors.append(f"{concept_id}->{neighbor_id}: {str(e)[:50]}")

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(
                    f"[RelationGenerator] Progress: {i+1}/{len(embeddings)} concepts, "
                    f"{relations_created} relations created"
                )

        duration_ms = (time.time() - start_time) * 1000
        avg_similarity = total_similarity / similarity_count if similarity_count > 0 else 0

        result = RelationGenerationResult(
            relations_created=relations_created,
            relations_skipped=relations_skipped,
            concepts_processed=len(embeddings),
            average_similarity=round(avg_similarity, 4),
            duration_ms=duration_ms,
            errors=errors[:50]
        )

        logger.info(
            f"[RelationGenerator] Complete: {relations_created} relations created, "
            f"{relations_skipped} skipped, avg sim={avg_similarity:.3f} in {duration_ms:.1f}ms"
        )

        return result

    async def find_similar_concepts(
        self,
        concept_id: str,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Trouve les concepts les plus similaires à un concept donné.

        Args:
            concept_id: ID du concept source
            top_k: Nombre de résultats
            min_similarity: Similarité minimum

        Returns:
            Liste de dicts avec concept_id, similarity
        """
        # Récupérer l'embedding du concept source
        concept = await self.db.get_concept_with_aliases(concept_id)
        if not concept or not concept.get("embedding"):
            return []

        source_embedding = self.deserialize_embedding(concept["embedding"])
        if not source_embedding:
            return []

        # Charger tous les autres embeddings
        concepts_data = await self.db.get_concepts_with_embeddings(limit=5000)

        embeddings = [
            (c["id"], self.deserialize_embedding(c["embedding"]))
            for c in concepts_data
            if c["id"] != concept_id and c["embedding"]
        ]

        # Calculer similarités
        similarities = self.cosine_similarity_batch(source_embedding, embeddings)

        # Filtrer et limiter
        results = [
            {"concept_id": cid, "similarity": round(sim, 4)}
            for cid, sim in similarities[:top_k]
            if sim >= min_similarity
        ]

        return results

    async def get_generation_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de génération de relations.
        """
        async with self.db.connection() as conn:
            # Total relations
            cursor = await conn.execute("SELECT COUNT(*) FROM relations")
            total = (await cursor.fetchone())[0]

            # Par source
            cursor = await conn.execute(
                "SELECT model_source, COUNT(*) FROM relations GROUP BY model_source"
            )
            by_source = {row[0] or "unknown": row[1] for row in await cursor.fetchall()}

            # Distribution des poids
            cursor = await conn.execute("""
                SELECT
                    CASE
                        WHEN weight >= 0.9 THEN 'very_high'
                        WHEN weight >= 0.7 THEN 'high'
                        WHEN weight >= 0.5 THEN 'medium'
                        ELSE 'low'
                    END as category,
                    COUNT(*)
                FROM relations
                GROUP BY category
            """)
            weight_distribution = {row[0]: row[1] for row in await cursor.fetchall()}

            # Kappa moyen
            cursor = await conn.execute("SELECT AVG(kappa) FROM relations")
            avg_kappa = (await cursor.fetchone())[0] or 0

            return {
                "total_relations": total,
                "by_model_source": by_source,
                "weight_distribution": weight_distribution,
                "average_kappa": round(avg_kappa, 4)
            }
