"""
LYRA-ACE - KAPPA WORKER
=======================

Worker asynchrone pour le calcul differe de la courbure kappa Ollivier.
Les insertions utilisent Jaccard (rapide), Ollivier est calcule en batch.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from database.engine import ISpaceDB, get_db
from database.graph_delta import KappaCalculator


class KappaWorker:
    """
    Worker pour le recalcul kappa Ollivier en batch.

    Les aretes sont inserees avec kappa_jaccard (O(1)).
    Ce worker recalcule kappa_ollivier en arriere-plan.
    """

    def __init__(self, db: ISpaceDB, alpha: float = 0.5):
        """
        Args:
            db: Instance de la base de donnees
            alpha: Coefficient de melange Ollivier/Jaccard pour kappa hybride
        """
        self.db = db
        self.calculator = KappaCalculator(alpha=alpha)
        self._running = False

    async def process_batch(
        self,
        limit: int = 100,
        max_attempts: int = 3
    ) -> int:
        """
        Traite un batch d'aretes en attente.

        Args:
            limit: Nombre maximum d'aretes a traiter
            max_attempts: Nombre maximum de tentatives par arete

        Returns:
            Nombre d'aretes traitees avec succes
        """
        batch = await self.db.get_pending_kappa_batch(limit=limit)
        processed = 0

        for item in batch:
            source = item["source"]
            target = item["target"]
            attempts = item.get("attempts", 0)

            if attempts >= max_attempts:
                # Trop de tentatives, abandonner
                await self.db.mark_kappa_recalc_done(source, target)
                continue

            try:
                # Recuperer les donnees pour le calcul
                relation = await self.db.get_relation(source, target)
                if not relation:
                    await self.db.mark_kappa_recalc_done(source, target)
                    continue

                source_concept = await self.db.get_concept(source)
                target_concept = await self.db.get_concept(target)

                if not source_concept or not target_concept:
                    await self.db.mark_kappa_recalc_done(source, target)
                    continue

                # Calculer les voisinages pour Jaccard
                source_neighbors = set(
                    n["target"] for n in await self.db.get_neighbors(source, limit=50)
                )
                target_neighbors = set(
                    n["target"] for n in await self.db.get_neighbors(target, limit=50)
                )

                # Calculer kappa hybride
                kappa_result = self.calculator.compute_hybrid(
                    degree_u=source_concept.get("degree", 1),
                    degree_v=target_concept.get("degree", 1),
                    weight=relation.get("weight", 1.0),
                    neighbors_u=source_neighbors,
                    neighbors_v=target_neighbors
                )

                # Mettre a jour la relation
                await self.db.update_edge_kappa(
                    source=source,
                    target=target,
                    kappa=kappa_result["kappa_hybrid"],
                    method="hybrid"
                )

                # Enregistrer dans l'historique
                await self.db.log_kappa_history(
                    source=source,
                    target=target,
                    kappa_ollivier=kappa_result["kappa_ollivier"],
                    kappa_jaccard=kappa_result["kappa_jaccard"],
                    kappa_hybrid=kappa_result["kappa_hybrid"],
                    alpha=kappa_result["alpha"]
                )

                # Marquer comme traite
                await self.db.mark_kappa_recalc_done(source, target)
                processed += 1

            except Exception as e:
                # Marquer l'echec
                await self.db.mark_kappa_recalc_failed(source, target, str(e))

        return processed

    async def run_continuous(
        self,
        interval_seconds: float = 5.0,
        batch_size: int = 50
    ) -> None:
        """
        Execute le worker en continu.

        Args:
            interval_seconds: Intervalle entre les batches
            batch_size: Taille des batches
        """
        self._running = True
        print(f"[KappaWorker] Demarrage (interval={interval_seconds}s, batch={batch_size})")

        while self._running:
            try:
                processed = await self.process_batch(limit=batch_size)
                if processed > 0:
                    print(f"[KappaWorker] {processed} aretes traitees")
            except Exception as e:
                print(f"[KappaWorker] Erreur: {e}")

            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        """Arrete le worker."""
        self._running = False
        print("[KappaWorker] Arret demande")


# ============================================================================
# FONCTION UTILITAIRE
# ============================================================================

async def run_kappa_worker_once(limit: int = 100) -> int:
    """
    Execute un seul batch de recalcul kappa.

    Utile pour les appels ponctuels (cron, API, etc.)

    Returns:
        Nombre d'aretes traitees
    """
    db = await get_db()
    worker = KappaWorker(db)
    return await worker.process_batch(limit=limit)
