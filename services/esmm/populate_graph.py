"""
ESMM Phase 1 - L1.1: Graph Populator
=====================================

Charge les concepts depuis topics.txt et génère leurs embeddings.

Usage:
    from services.esmm import GraphPopulator

    populator = GraphPopulator(db)
    result = await populator.populate_from_file("data/topics.txt")

Author: Lyra-ACE ESMM Protocol
"""

import re
import time
import logging
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from app.embeddings import get_embeddings, EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class PopulationResult:
    """Résultat de la population du graphe."""
    concepts_loaded: int
    concepts_skipped: int
    embeddings_generated: int
    embeddings_failed: int
    duplicates_found: int
    duration_ms: float
    errors: List[str]


class GraphPopulator:
    """
    Charge les concepts depuis un fichier texte et les insère dans le graphe.

    Responsabilités:
    - Lecture et parsing du fichier topics.txt
    - Normalisation des noms de concepts
    - Génération des embeddings 1024D via Ollama
    - Insertion dans la base de données
    - Gestion des doublons et erreurs
    """

    def __init__(self, db, batch_size: int = 50):
        """
        Args:
            db: Instance ISpaceDB
            batch_size: Nombre de concepts par batch pour les embeddings
        """
        self.db = db
        self.batch_size = batch_size
        self._seen_concepts: Set[str] = set()

    def normalize_concept_name(self, raw: str) -> Optional[str]:
        """
        Normalise un nom de concept.

        Règles:
        - Trim whitespace
        - Lowercase
        - Remplace espaces multiples par underscore
        - Supprime caractères spéciaux sauf tirets et underscores
        - Minimum 2 caractères

        Args:
            raw: Nom brut du concept

        Returns:
            Nom normalisé ou None si invalide
        """
        if not raw or not isinstance(raw, str):
            return None

        # Trim et lowercase
        name = raw.strip().lower()

        # Ignorer les lignes vides ou commentaires
        if not name or name.startswith('#') or name.startswith('//'):
            return None

        # Ignorer les lignes qui semblent être des instructions/métadonnées
        skip_patterns = [
            r'^let\'s',
            r'^final list',
            r'^produce',
            r'^\d+\.',  # Lignes numérotées
            r'^\.',
        ]
        for pattern in skip_patterns:
            if re.match(pattern, name):
                return None

        # Supprimer le point final s'il existe
        name = name.rstrip('.')

        # Remplacer caractères spéciaux par espace
        name = re.sub(r'[^\w\s\-àâäéèêëïîôùûüçœæ]', ' ', name)

        # Remplacer espaces multiples par un seul
        name = re.sub(r'\s+', ' ', name)

        # Trim à nouveau
        name = name.strip()

        # Minimum 2 caractères
        if len(name) < 2:
            return None

        # Maximum 100 caractères
        if len(name) > 100:
            name = name[:100].rsplit(' ', 1)[0]

        return name

    def load_topics_file(self, file_path: str) -> List[str]:
        """
        Charge et parse le fichier topics.txt.

        Args:
            file_path: Chemin vers le fichier

        Returns:
            Liste des noms de concepts normalisés (uniques)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Topics file not found: {file_path}")

        concepts = []
        seen = set()

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                normalized = self.normalize_concept_name(line)
                if normalized and normalized not in seen:
                    concepts.append(normalized)
                    seen.add(normalized)

        logger.info(f"[GraphPopulator] Loaded {len(concepts)} unique concepts from {file_path}")
        return concepts

    def serialize_embedding(self, embedding: List[float]) -> bytes:
        """
        Sérialise un embedding en bytes pour stockage SQLite.

        Format: 1024 float32 = 4096 bytes
        """
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(f"Embedding dimension mismatch: {len(embedding)} != {EMBEDDING_DIM}")

        return struct.pack(f'{len(embedding)}f', *embedding)

    def deserialize_embedding(self, data: bytes) -> List[float]:
        """
        Désérialise un embedding depuis bytes.
        """
        count = len(data) // 4  # float32 = 4 bytes
        return list(struct.unpack(f'{count}f', data))

    async def populate_from_file(
        self,
        file_path: str = "data/topics.txt",
        generate_embeddings: bool = True,
        skip_existing: bool = True
    ) -> PopulationResult:
        """
        Population complète depuis un fichier.

        Args:
            file_path: Chemin vers topics.txt
            generate_embeddings: Si True, génère les embeddings via Ollama
            skip_existing: Si True, ignore les concepts déjà en base

        Returns:
            PopulationResult avec statistiques
        """
        start_time = time.time()
        errors: List[str] = []

        # Charger les concepts
        try:
            concepts = self.load_topics_file(file_path)
        except FileNotFoundError as e:
            return PopulationResult(
                concepts_loaded=0,
                concepts_skipped=0,
                embeddings_generated=0,
                embeddings_failed=0,
                duplicates_found=0,
                duration_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

        # Récupérer les concepts existants si skip_existing
        existing_concepts: Set[str] = set()
        if skip_existing:
            async with self.db.connection() as conn:
                cursor = await conn.execute("SELECT id FROM concepts")
                existing_concepts = {row[0] for row in await cursor.fetchall()}
            logger.info(f"[GraphPopulator] {len(existing_concepts)} concepts already in database")

        # Stats
        loaded = 0
        skipped = 0
        embeddings_generated = 0
        embeddings_failed = 0
        duplicates = 0

        # Traiter par batches
        for i in range(0, len(concepts), self.batch_size):
            batch = concepts[i:i + self.batch_size]

            for concept in batch:
                # Vérifier si existe déjà
                if concept in existing_concepts:
                    skipped += 1
                    continue

                # Vérifier doublons dans ce run
                if concept in self._seen_concepts:
                    duplicates += 1
                    continue

                self._seen_concepts.add(concept)

                # Générer embedding
                embedding_bytes: Optional[bytes] = None
                if generate_embeddings:
                    try:
                        embedding = await get_embeddings(concept)
                        embedding_bytes = self.serialize_embedding(embedding)
                        embeddings_generated += 1
                    except Exception as e:
                        logger.warning(f"[GraphPopulator] Embedding failed for '{concept}': {e}")
                        errors.append(f"Embedding failed: {concept}")
                        embeddings_failed += 1

                # Insérer le concept
                try:
                    await self.db.add_concept(
                        concept_id=concept,
                        rho_static=0.0,
                        embedding=embedding_bytes,
                        source="topics_file",
                        first_seen_model=None
                    )
                    loaded += 1
                except Exception as e:
                    logger.error(f"[GraphPopulator] Insert failed for '{concept}': {e}")
                    errors.append(f"Insert failed: {concept}")

            # Log progress
            progress = min(i + self.batch_size, len(concepts))
            logger.info(f"[GraphPopulator] Progress: {progress}/{len(concepts)} concepts processed")

        duration_ms = (time.time() - start_time) * 1000

        result = PopulationResult(
            concepts_loaded=loaded,
            concepts_skipped=skipped,
            embeddings_generated=embeddings_generated,
            embeddings_failed=embeddings_failed,
            duplicates_found=duplicates,
            duration_ms=duration_ms,
            errors=errors[:50]  # Limiter les erreurs
        )

        logger.info(
            f"[GraphPopulator] Complete: {loaded} loaded, {skipped} skipped, "
            f"{embeddings_generated} embeddings in {duration_ms:.1f}ms"
        )

        return result

    async def populate_from_list(
        self,
        concepts: List[str],
        source: str = "manual",
        generate_embeddings: bool = True
    ) -> PopulationResult:
        """
        Population depuis une liste de concepts.

        Args:
            concepts: Liste de noms de concepts
            source: Source de provenance
            generate_embeddings: Si True, génère les embeddings

        Returns:
            PopulationResult
        """
        start_time = time.time()
        errors: List[str] = []

        loaded = 0
        skipped = 0
        embeddings_generated = 0
        embeddings_failed = 0

        for raw_concept in concepts:
            concept = self.normalize_concept_name(raw_concept)
            if not concept:
                skipped += 1
                continue

            # Générer embedding
            embedding_bytes: Optional[bytes] = None
            if generate_embeddings:
                try:
                    embedding = await get_embeddings(concept)
                    embedding_bytes = self.serialize_embedding(embedding)
                    embeddings_generated += 1
                except Exception as e:
                    logger.warning(f"[GraphPopulator] Embedding failed: {e}")
                    embeddings_failed += 1

            # Insérer
            try:
                await self.db.add_concept(
                    concept_id=concept,
                    rho_static=0.0,
                    embedding=embedding_bytes,
                    source=source
                )
                loaded += 1
            except Exception as e:
                logger.error(f"[GraphPopulator] Insert failed: {e}")
                errors.append(str(e))

        return PopulationResult(
            concepts_loaded=loaded,
            concepts_skipped=skipped,
            embeddings_generated=embeddings_generated,
            embeddings_failed=embeddings_failed,
            duplicates_found=0,
            duration_ms=(time.time() - start_time) * 1000,
            errors=errors
        )

    async def get_population_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques actuelles de population.
        """
        async with self.db.connection() as conn:
            # Total concepts
            cursor = await conn.execute("SELECT COUNT(*) FROM concepts")
            total = (await cursor.fetchone())[0]

            # Avec embeddings
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
            )
            with_embeddings = (await cursor.fetchone())[0]

            # Par source
            cursor = await conn.execute(
                "SELECT source, COUNT(*) FROM concepts GROUP BY source"
            )
            by_source = {row[0]: row[1] for row in await cursor.fetchall()}

            # Degré moyen
            cursor = await conn.execute("SELECT AVG(degree) FROM concepts")
            avg_degree = (await cursor.fetchone())[0] or 0

            return {
                "total_concepts": total,
                "with_embeddings": with_embeddings,
                "without_embeddings": total - with_embeddings,
                "embedding_coverage": with_embeddings / total if total > 0 else 0,
                "by_source": by_source,
                "average_degree": round(avg_degree, 2)
            }
