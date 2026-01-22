"""
ESMM Phase 1 - L1.3: Seed Injector
===================================

Injecte la graine sémantique dialectique dans le graphe.

La graine ESMM contient:
- Oppositions dialectiques fondamentales (cause/effet, théorie/pratique)
- Domaines de tension épistémiques (quantique/classique, objectif/subjectif)
- Concepts fondamentaux pour l'ancrage sémantique

Usage:
    from services.esmm import SeedInjector

    injector = SeedInjector(db)
    result = await injector.inject_seed(seed_type="standard")

Author: Lyra-ACE ESMM Protocol
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from database.graph_delta import GraphDelta, DeltaOperation
from app.embeddings import get_embeddings
from .populate_graph import GraphPopulator

logger = logging.getLogger(__name__)


@dataclass
class SeedInjectionResult:
    """Résultat de l'injection de la graine."""
    concepts_created: int
    relations_created: int
    concepts_existed: int
    duration_ms: float
    seed_type: str
    errors: List[str]


# ============================================================================
# GRAINES SÉMANTIQUES
# ============================================================================

SEED_STANDARD = {
    "concepts_fondamentaux": [
        # Oppositions dialectiques de base
        ("cause", "effet"),
        ("principe", "application"),
        ("théorie", "pratique"),
        ("abstrait", "concret"),
        ("simple", "complexe"),
        ("local", "global"),
        ("statique", "dynamique"),
        ("continu", "discret"),
        ("ordre", "chaos"),
        ("unité", "multiplicité"),
    ],
    "domaines_tension": [
        # Tensions épistémiques
        ("quantique", "classique"),
        ("déterministe", "probabiliste"),
        ("objectif", "subjectif"),
        ("empirique", "théorique"),
        ("analytique", "synthétique"),
        ("réductionniste", "holiste"),
        ("linéaire", "non-linéaire"),
        ("synchrone", "asynchrone"),
    ],
    "hierarchies_cognitives": [
        # Relations hiérarchiques
        ("information", "donnée"),
        ("connaissance", "information"),
        ("sagesse", "connaissance"),
        ("concept", "instance"),
        ("catégorie", "élément"),
        ("système", "composant"),
    ],
    "processus_fondamentaux": [
        # Relations processus
        ("entrée", "sortie"),
        ("stimulus", "réponse"),
        ("question", "réponse"),
        ("problème", "solution"),
        ("hypothèse", "vérification"),
        ("observation", "interprétation"),
    ]
}

SEED_MINIMAL = {
    "concepts_fondamentaux": [
        ("cause", "effet"),
        ("théorie", "pratique"),
        ("abstrait", "concret"),
    ],
    "domaines_tension": [
        ("quantique", "classique"),
        ("objectif", "subjectif"),
    ]
}

SEED_EXTENDED = {
    **SEED_STANDARD,
    "sciences_naturelles": [
        ("énergie", "matière"),
        ("espace", "temps"),
        ("force", "mouvement"),
        ("onde", "particule"),
        ("champ", "source"),
        ("potentiel", "actuel"),
        ("équilibre", "déséquilibre"),
        ("conservation", "transformation"),
    ],
    "sciences_formelles": [
        ("axiome", "théorème"),
        ("définition", "démonstration"),
        ("ensemble", "élément"),
        ("fonction", "argument"),
        ("variable", "constante"),
        ("fini", "infini"),
        ("nécessaire", "contingent"),
        ("vrai", "faux"),
    ],
    "cognition": [
        ("perception", "conception"),
        ("mémoire", "apprentissage"),
        ("attention", "conscience"),
        ("raison", "intuition"),
        ("analyse", "synthèse"),
        ("induction", "déduction"),
        ("analogie", "différence"),
    ],
    "language": [
        ("signifiant", "signifié"),
        ("syntaxe", "sémantique"),
        ("dénotation", "connotation"),
        ("explicite", "implicite"),
        ("littéral", "métaphorique"),
    ]
}

# Mapping des types de seed
SEED_TYPES = {
    "minimal": SEED_MINIMAL,
    "standard": SEED_STANDARD,
    "extended": SEED_EXTENDED,
}


class SeedInjector:
    """
    Injecte la graine sémantique dialectique dans le graphe.

    La graine fournit:
    1. Des points d'ancrage stables pour le graphe
    2. Des oppositions dialectiques pour structurer l'espace sémantique
    3. Des relations de haute confiance (seed = source fiable)
    """

    def __init__(self, db):
        """
        Args:
            db: Instance ISpaceDB
        """
        self.db = db
        self._populator = GraphPopulator(db)

    async def inject_seed(
        self,
        seed_type: str = "standard",
        custom_seed: Optional[Dict[str, List[Tuple[str, str]]]] = None,
        generate_embeddings: bool = True,
        skip_existing_concepts: bool = True
    ) -> SeedInjectionResult:
        """
        Injecte la graine sémantique complète.

        Args:
            seed_type: Type de graine ('minimal', 'standard', 'extended')
            custom_seed: Graine personnalisée (override seed_type)
            generate_embeddings: Si True, génère les embeddings pour nouveaux concepts
            skip_existing_concepts: Si True, n'écrase pas les concepts existants

        Returns:
            SeedInjectionResult avec statistiques
        """
        start_time = time.time()
        errors: List[str] = []

        # Sélectionner la graine
        if custom_seed:
            seed = custom_seed
            seed_type = "custom"
        elif seed_type in SEED_TYPES:
            seed = SEED_TYPES[seed_type]
        else:
            return SeedInjectionResult(
                concepts_created=0,
                relations_created=0,
                concepts_existed=0,
                duration_ms=0,
                seed_type=seed_type,
                errors=[f"Unknown seed type: {seed_type}"]
            )

        # Collecter tous les concepts uniques
        all_concepts = set()
        for category, pairs in seed.items():
            for c1, c2 in pairs:
                all_concepts.add(c1)
                all_concepts.add(c2)

        logger.info(f"[SeedInjector] Injecting {seed_type} seed: {len(all_concepts)} concepts")

        # Vérifier les concepts existants
        existing_concepts = set()
        async with self.db.connection() as conn:
            placeholders = ','.join('?' * len(all_concepts))
            cursor = await conn.execute(
                f"SELECT id FROM concepts WHERE id IN ({placeholders})",
                list(all_concepts)
            )
            existing_concepts = {row[0] for row in await cursor.fetchall()}

        concepts_existed = len(existing_concepts)
        concepts_to_create = all_concepts - existing_concepts if skip_existing_concepts else all_concepts

        logger.info(
            f"[SeedInjector] {len(existing_concepts)} concepts exist, "
            f"{len(concepts_to_create)} to create"
        )

        # Créer les nouveaux concepts
        concepts_created = 0
        for concept in concepts_to_create:
            embedding_bytes = None
            if generate_embeddings:
                try:
                    embedding = await get_embeddings(concept)
                    embedding_bytes = self._populator.serialize_embedding(embedding)
                except Exception as e:
                    logger.warning(f"[SeedInjector] Embedding failed for '{concept}': {e}")
                    errors.append(f"Embedding: {concept}")

            try:
                await self.db.add_concept(
                    concept_id=concept,
                    rho_static=0.5,  # Concepts seed ont une densité moyenne
                    embedding=embedding_bytes,
                    source="seed",
                    first_seen_model=None
                )
                concepts_created += 1
            except Exception as e:
                logger.warning(f"[SeedInjector] Concept creation failed for '{concept}': {e}")
                errors.append(f"Concept: {concept}")

        # Créer les relations avec types appropriés
        relations_created = 0
        relation_type_mapping = {
            "concepts_fondamentaux": "opposite_of",
            "domaines_tension": "opposite_of",
            "hierarchies_cognitives": "is_a",
            "processus_fondamentaux": "follows",
            "sciences_naturelles": "related_to",
            "sciences_formelles": "related_to",
            "cognition": "related_to",
            "language": "related_to",
        }

        for category, pairs in seed.items():
            relation_type = relation_type_mapping.get(category, "related_to")

            for c1, c2 in pairs:
                try:
                    # Créer la relation c1 -> c2
                    delta = GraphDelta(
                        operation=DeltaOperation.ADD_EDGE,
                        source=c1,
                        target=c2,
                        weight=0.9,  # Haute confiance pour seed
                        confidence=0.95,
                        model_source="seed",
                        reason=f"ESMM seed: {category}"
                    )

                    # Vérifier si existe déjà
                    existing = await self.db.get_relation(c1, c2)
                    if existing:
                        logger.debug(f"[SeedInjector] Relation exists: {c1} -> {c2}")
                        continue

                    await self.db.apply_delta(delta)
                    relations_created += 1

                    # Pour les oppositions, créer aussi la relation inverse
                    if relation_type == "opposite_of":
                        existing_inv = await self.db.get_relation(c2, c1)
                        if not existing_inv:
                            delta_inv = GraphDelta(
                                operation=DeltaOperation.ADD_EDGE,
                                source=c2,
                                target=c1,
                                weight=0.9,
                                confidence=0.95,
                                model_source="seed",
                                reason=f"ESMM seed: {category} (inverse)"
                            )
                            await self.db.apply_delta(delta_inv)
                            relations_created += 1

                except Exception as e:
                    logger.warning(f"[SeedInjector] Relation failed {c1}->{c2}: {e}")
                    errors.append(f"Relation: {c1}->{c2}")

        duration_ms = (time.time() - start_time) * 1000

        result = SeedInjectionResult(
            concepts_created=concepts_created,
            relations_created=relations_created,
            concepts_existed=concepts_existed,
            duration_ms=duration_ms,
            seed_type=seed_type,
            errors=errors[:50]
        )

        logger.info(
            f"[SeedInjector] Complete: {concepts_created} concepts, "
            f"{relations_created} relations in {duration_ms:.1f}ms"
        )

        return result

    async def inject_dialectical_pairs(
        self,
        pairs: List[Tuple[str, str]],
        relation_type: str = "opposite_of",
        confidence: float = 0.9
    ) -> Dict[str, int]:
        """
        Injecte des paires dialectiques spécifiques.

        Args:
            pairs: Liste de tuples (concept1, concept2)
            relation_type: Type de relation
            confidence: Confiance des relations

        Returns:
            Dict avec counts
        """
        created = 0
        skipped = 0
        errors = 0

        for c1, c2 in pairs:
            try:
                # Assurer que les concepts existent
                await self._ensure_concept_exists(c1)
                await self._ensure_concept_exists(c2)

                # Créer la relation bidirectionnelle
                for source, target in [(c1, c2), (c2, c1)]:
                    existing = await self.db.get_relation(source, target)
                    if existing:
                        skipped += 1
                        continue

                    delta = GraphDelta(
                        operation=DeltaOperation.ADD_EDGE,
                        source=source,
                        target=target,
                        weight=confidence,
                        confidence=confidence,
                        model_source="seed",
                        reason=f"Dialectical pair: {relation_type}"
                    )
                    await self.db.apply_delta(delta)
                    created += 1

            except Exception as e:
                logger.warning(f"[SeedInjector] Pair failed {c1}/{c2}: {e}")
                errors += 1

        return {"created": created, "skipped": skipped, "errors": errors}

    async def _ensure_concept_exists(
        self,
        concept_id: str,
        generate_embedding: bool = True
    ) -> bool:
        """
        S'assure qu'un concept existe, le crée sinon.

        Returns:
            True si créé, False si existait déjà
        """
        existing = await self.db.get_concept(concept_id)
        if existing:
            return False

        embedding_bytes = None
        if generate_embedding:
            try:
                embedding = await get_embeddings(concept_id)
                embedding_bytes = self._populator.serialize_embedding(embedding)
            except Exception as e:
                logger.warning(f"[SeedInjector] Embedding failed: {e}")

        await self.db.add_concept(
            concept_id=concept_id,
            rho_static=0.5,
            embedding=embedding_bytes,
            source="seed"
        )
        return True

    def get_available_seeds(self) -> Dict[str, Dict[str, int]]:
        """
        Retourne les types de graines disponibles avec leurs statistiques.
        """
        result = {}
        for seed_type, seed in SEED_TYPES.items():
            concepts = set()
            relations = 0
            for category, pairs in seed.items():
                relations += len(pairs)
                for c1, c2 in pairs:
                    concepts.add(c1)
                    concepts.add(c2)

            result[seed_type] = {
                "concepts": len(concepts),
                "relations": relations,
                "categories": list(seed.keys())
            }
        return result

    async def get_seed_status(self) -> Dict[str, Any]:
        """
        Vérifie le statut de la graine dans le graphe actuel.
        """
        async with self.db.connection() as conn:
            # Concepts seed
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE source = 'seed'"
            )
            seed_concepts = (await cursor.fetchone())[0]

            # Relations seed
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM relations WHERE model_source = 'seed'"
            )
            seed_relations = (await cursor.fetchone())[0]

            # Kappa moyen des relations seed
            cursor = await conn.execute(
                "SELECT AVG(kappa) FROM relations WHERE model_source = 'seed'"
            )
            avg_kappa = (await cursor.fetchone())[0] or 0

            # Degré moyen des concepts seed
            cursor = await conn.execute(
                "SELECT AVG(degree) FROM concepts WHERE source = 'seed'"
            )
            avg_degree = (await cursor.fetchone())[0] or 0

            return {
                "seed_concepts": seed_concepts,
                "seed_relations": seed_relations,
                "average_kappa": round(avg_kappa, 4),
                "average_degree": round(avg_degree, 2),
                "seed_coverage": seed_concepts > 0
            }
