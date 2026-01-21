"""
LYRA CLEAN - DATABASE ENGINE
============================

Async SQLite engine with connection pooling and performance optimizations.

Key Features:
- Zero CSV loading (all data in SQLite)
- O(1) concept lookups via indexes
- WAL mode for concurrent reads
- Context managers for safe transactions

Author: Refactored from Lyra_Uni_3 legacy
"""
from __future__ import annotations

import aiosqlite
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

from database.graph_delta import (
    GraphDelta, DeltaBatch, DeltaOperation,
    KappaCalculator, DeltaValidationError, MutationLimitExceededError
)


class ISpaceDB:
    """
    Unified database engine for Lyra Clean.

    Replaces:
    - lyra_core/graph_loader.py (CSV loading)
    - lyra_core/memory_store.py (RAM cache)
    - ispacenav/graph_store.py (separate SQLite)

    Performance:
    - Indexed queries: O(log N)
    - Connection pooling via aiosqlite
    - WAL mode: concurrent reads + single writer
    """

    def __init__(self, db_path: str = "data/ispace.db"):
        """
        Initialize database engine.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """
        Initialize database with schema and performance optimizations.

        Must be called before any queries.
        """
        # Read schema from file
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Create database and apply schema
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema_sql)

            # Performance optimizations
            await db.execute("PRAGMA journal_mode=WAL")        # Write-Ahead Logging
            await db.execute("PRAGMA synchronous=NORMAL")      # Balance safety/speed
            await db.execute("PRAGMA cache_size=-64000")       # 64MB cache
            await db.execute("PRAGMA temp_store=MEMORY")       # Temp tables in RAM
            await db.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped I/O

            await db.commit()

        print(f"[ISpaceDB] Initialized at {self.db_path}")

    @asynccontextmanager
    async def connection(self):
        """
        Context manager for database connections.

        Usage:
            async with db.connection() as conn:
                cursor = await conn.execute("SELECT ...")
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row  # Dict-like rows
            yield conn

    # ========================================================================
    # CONCEPT QUERIES (replaces graph_loader.py)
    # ========================================================================

    async def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve concept metadata.

        Args:
            concept_id: Concept identifier (e.g., "entropy")

        Returns:
            Dict with keys: id, rho_static, degree, access_count
            None if concept not found

        Performance: O(1) via primary key index
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, rho_static, degree, access_count, last_accessed
                FROM concepts
                WHERE id = ?
                """,
                (concept_id,)
            )
            row = await cursor.fetchone()

            if row:
                # Update access tracking
                await conn.execute(
                    """
                    UPDATE concepts
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                    """,
                    (time.time(), concept_id)
                )
                await conn.commit()

                return dict(row)
            return None

    async def get_neighbors(
        self,
        concept_id: str,
        limit: int = 20,
        min_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get semantic neighbors of a concept (most frequent operation).

        Args:
            concept_id: Source concept
            limit: Maximum neighbors to return
            min_weight: Minimum PPMI weight threshold

        Returns:
            List of dicts with keys: target, weight, kappa

        Performance: O(log N) via idx_relations_source

        Example:
            neighbors = await db.get_neighbors("entropy", limit=10)
            # [{"target": "information", "weight": 0.85, "kappa": 0.62}, ...]
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT target, weight, kappa
                FROM relations
                WHERE source = ? AND weight >= ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (concept_id, min_weight, limit)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_multi_neighbors(
        self,
        concept_ids: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get neighbors for multiple concepts (batch query).

        Args:
            concept_ids: List of concept identifiers
            limit: Total neighbors to return (top weighted)

        Returns:
            Aggregated list of neighbors, sorted by weight

        Performance: Single query vs N queries = ~10x faster
        """
        if not concept_ids:
            return []

        placeholders = ','.join('?' * len(concept_ids))

        async with self.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT target, weight, kappa, source
                FROM relations
                WHERE source IN ({placeholders})
                ORDER BY weight DESC
                LIMIT ?
                """,
                (*concept_ids, limit)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def search_concepts(
        self,
        pattern: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search concepts by pattern (case-insensitive).

        Args:
            pattern: SQL LIKE pattern (e.g., "entr%")
            limit: Maximum results

        Returns:
            List of matching concepts with metadata
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, rho_static, degree, access_count
                FROM concepts
                WHERE id LIKE ?
                ORDER BY degree DESC, rho_static DESC
                LIMIT ?
                """,
                (pattern, limit)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def add_concept(
        self,
        concept_id: str,
        rho_static: float = 0.0,
        embedding: bytes = None,
        source: str = "manual",
        first_seen_model: str = None
    ) -> None:
        """
        Ajoute un nouveau concept au graphe.

        Args:
            concept_id: Identifiant canonique
            rho_static: Densite pre-calculee
            embedding: Vecteur d'embedding (bytes)
            source: Source du concept
            first_seen_model: Modele ayant introduit le concept
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO concepts
                (id, rho_static, degree, embedding, embedding_updated_at, source, first_seen_model, created_at)
                VALUES (?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    concept_id, rho_static, embedding,
                    time.time() if embedding else None,
                    source, first_seen_model, time.time()
                )
            )
            await conn.commit()

    async def get_concepts_with_embeddings(self, limit: int = 1000) -> List[Dict]:
        """
        Recupere les concepts avec leurs embeddings pour recherche de similarite.

        Args:
            limit: Nombre maximum de concepts

        Returns:
            Liste de dicts avec id et embedding
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, embedding
                FROM concepts
                WHERE embedding IS NOT NULL
                ORDER BY degree DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def get_relation(self, source: str, target: str) -> Optional[Dict]:
        """
        Recupere une relation specifique.

        Args:
            source: Concept source
            target: Concept cible

        Returns:
            Dict avec les donnees de la relation ou None
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT source, target, weight, kappa, kappa_method, relation_type,
                       confidence, model_source, extraction_count
                FROM relations
                WHERE source = ? AND target = ?
                """,
                (source, target)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_edge_kappa(
        self,
        source: str,
        target: str,
        kappa: float,
        method: str = "hybrid"
    ) -> None:
        """
        Met a jour la courbure kappa d'une arete.

        Args:
            source: Concept source
            target: Concept cible
            kappa: Nouvelle valeur de kappa
            method: Methode de calcul ('jaccard', 'ollivier', 'hybrid')
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE relations
                SET kappa = ?, kappa_method = ?, updated_at = ?
                WHERE source = ? AND target = ?
                """,
                (kappa, method, time.time(), source, target)
            )
            await conn.commit()

    async def log_kappa_history(
        self,
        source: str,
        target: str,
        kappa_ollivier: float,
        kappa_jaccard: float,
        kappa_hybrid: float,
        alpha: float = 0.5
    ) -> None:
        """
        Enregistre un calcul de kappa dans l'historique.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO kappa_history
                (source, target, kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source, target, kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha, time.time())
            )
            await conn.commit()

    # ========================================================================
    # SESSION MANAGEMENT (replaces memory_store.py)
    # ========================================================================

    async def create_session(
        self,
        session_id: str,
        profile: str = "balanced",
        params_snapshot: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Create new conversation session.

        Args:
            session_id: UUID v4
            profile: Bezier profile name
            params_snapshot: Initial parameters (optional)
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_activity, profile, params_snapshot, message_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (
                    session_id,
                    time.time(),
                    time.time(),
                    profile,
                    json.dumps(params_snapshot) if params_snapshot else None
                )
            )
            await conn.commit()

    async def append_event(
        self,
        session_id: str,
        event_type: str,
        role: Optional[str] = None,
        content: Optional[str] = None,
        injected_concepts: Optional[List[str]] = None,
        graph_weight: float = 0.0,
        latency_ms: Optional[float] = None
    ) -> int:
        """
        Append event to session (immutable log).

        Args:
            session_id: Session UUID
            event_type: 'user_message', 'assistant_message', 'system_event'
            role: 'user', 'assistant', 'system'
            content: Message text or JSON payload
            injected_concepts: Concepts used in context injection
            graph_weight: Contextual weight from graph
            latency_ms: Processing time

        Returns:
            event_id: Auto-incremented ID
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO events (
                    session_id, event_type, role, content,
                    injected_concepts, graph_weight, timestamp, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_type,
                    role,
                    content,
                    json.dumps(injected_concepts) if injected_concepts else None,
                    graph_weight,
                    time.time(),
                    latency_ms
                )
            )

            # Update session metadata
            await conn.execute(
                """
                UPDATE sessions
                SET last_activity = ?, message_count = message_count + 1
                WHERE session_id = ?
                """,
                (time.time(), session_id)
            )

            await conn.commit()
            return cursor.lastrowid

    async def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve session event history.

        Args:
            session_id: Session UUID
            limit: Maximum events to return
            event_types: Filter by event types (e.g., ['user_message', 'assistant_message'])

        Returns:
            List of events, ordered by timestamp ASC

        Performance: O(log N) via idx_events_session
        """
        event_type_filter = ""
        params: Tuple = (session_id,)

        if event_types:
            placeholders = ','.join('?' * len(event_types))
            event_type_filter = f"AND event_type IN ({placeholders})"
            params = (session_id, *event_types)

        async with self.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT event_id, event_type, role, content, timestamp,
                       injected_concepts, graph_weight, latency_ms
                FROM events
                WHERE session_id = ? {event_type_filter}
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (*params, limit)
            )
            rows = await cursor.fetchall()

            # Parse JSON fields
            events = []
            for row in rows:
                event = dict(row)
                if event['injected_concepts']:
                    event['injected_concepts'] = json.loads(event['injected_concepts'])
                events.append(event)

            return events

    async def get_conversation_messages(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, str]]:
        """
        Get formatted conversation history (user/assistant messages only).

        Args:
            session_id: Session UUID
            limit: Maximum messages

        Returns:
            List of dicts with keys: role, content
            Format compatible with Ollama API
        """
        events = await self.get_session_history(
            session_id,
            limit=limit,
            event_types=['user_message', 'assistant_message']
        )

        return [
            {"role": e['role'], "content": e['content']}
            for e in events
            if e['role'] and e['content']
        ]

    async def cleanup_old_sessions(
        self,
        max_age_days: int = 30,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Nettoie les sessions inactives depuis plus de max_age_days.

        Args:
            max_age_days: Age maximum en jours (defaut: 30)
            dry_run: Si True, retourne uniquement le compte sans supprimer

        Returns:
            Dict avec:
            - sessions_deleted: Nombre de sessions supprimees
            - events_deleted: Nombre d'evenements supprimes
            - trajectories_deleted: Nombre de trajectoires supprimees
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        async with self.connection() as conn:
            # Compter les sessions a supprimer
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE last_activity < ?",
                (cutoff_time,)
            )
            sessions_count = (await cursor.fetchone())[0]

            if dry_run:
                # Mode simulation: compter sans supprimer
                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM events e
                    JOIN sessions s ON e.session_id = s.session_id
                    WHERE s.last_activity < ?
                    """,
                    (cutoff_time,)
                )
                events_count = (await cursor.fetchone())[0]

                cursor = await conn.execute(
                    """
                    SELECT COUNT(*) FROM trajectories t
                    JOIN sessions s ON t.session_id = s.session_id
                    WHERE s.last_activity < ?
                    """,
                    (cutoff_time,)
                )
                trajectories_count = (await cursor.fetchone())[0]

                return {
                    "sessions_deleted": sessions_count,
                    "events_deleted": events_count,
                    "trajectories_deleted": trajectories_count,
                    "dry_run": True
                }

            # Supprimer (CASCADE supprime events et trajectories)
            await conn.execute(
                "DELETE FROM sessions WHERE last_activity < ?",
                (cutoff_time,)
            )
            await conn.commit()

            # Nettoyer les graph_deltas orphelins
            await conn.execute(
                """
                UPDATE graph_deltas SET session_id = NULL
                WHERE session_id NOT IN (SELECT session_id FROM sessions)
                """
            )
            await conn.commit()

            return {
                "sessions_deleted": sessions_count,
                "events_deleted": 0,  # CASCADE, pas compte exact
                "trajectories_deleted": 0,  # CASCADE
                "dry_run": False
            }

    async def get_inactive_sessions(
        self,
        min_age_days: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Liste les sessions inactives.

        Args:
            min_age_days: Age minimum d'inactivite
            limit: Nombre maximum de resultats

        Returns:
            Liste de sessions avec leur derniere activite
        """
        cutoff_time = time.time() - (min_age_days * 24 * 60 * 60)

        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT session_id, profile, message_count, last_activity, created_at
                FROM sessions
                WHERE last_activity < ?
                ORDER BY last_activity ASC
                LIMIT ?
                """,
                (cutoff_time, limit)
            )
            return [dict(row) for row in await cursor.fetchall()]

    # ========================================================================
    # TRAJECTORY LOGGING (Physics engine state)
    # ========================================================================

    async def log_trajectory_point(
        self,
        session_id: str,
        t_param: float,
        tau_c: float,
        rho: float,
        delta_r: float,
        kappa: Optional[float] = None,
        event_id: Optional[int] = None
    ) -> None:
        """
        Log Bezier trajectory point.

        Args:
            session_id: Session UUID
            t_param: Time parameter t ∈ [0, 1]
            tau_c, rho, delta_r: Physics parameters at t
            kappa: Optional curvature
            event_id: Associated event (optional)
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO trajectories (
                    session_id, event_id, t_param,
                    tau_c, rho, delta_r, kappa, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, event_id, t_param, tau_c, rho, delta_r, kappa, time.time())
            )
            await conn.commit()

    # ========================================================================
    # PROFILE MANAGEMENT (Bezier curves)
    # ========================================================================

    async def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve Bezier profile configuration.

        Args:
            profile_name: Profile identifier (e.g., "creative", "safe")

        Returns:
            Dict with keys: profile_name, tau_c_curve, rho_curve, delta_r_curve
            Curves are parsed JSON lists
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT profile_name, description,
                       tau_c_curve, rho_curve, delta_r_curve, kappa_curve
                FROM profiles
                WHERE profile_name = ?
                """,
                (profile_name,)
            )
            row = await cursor.fetchone()

            if row:
                profile = dict(row)
                # Parse JSON curves
                for key in ['tau_c_curve', 'rho_curve', 'delta_r_curve', 'kappa_curve']:
                    if profile[key]:
                        profile[key] = json.loads(profile[key])
                return profile
            return None

    async def list_profiles(self) -> List[Dict[str, str]]:
        """
        List all available Bezier profiles.

        Returns:
            List of dicts with keys: profile_name, description
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT profile_name, description, is_default
                FROM profiles
                ORDER BY is_default DESC, profile_name ASC
                """
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # CONSCIOUSNESS ADJUSTMENTS (Phase 2)
    # ========================================================================

    async def get_last_consciousness_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent consciousness metrics for a session.

        Args:
            session_id: Session UUID

        Returns:
            ConsciousnessMetrics dict or None if no metrics exist
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT metrics FROM session_adjustments
                WHERE session_id = ?
                ORDER BY turn_number DESC
                LIMIT 1
                """,
                (session_id,)
            )
            row = await cursor.fetchone()

            if row and row[0]:
                import json
                try:
                    metrics_dict = json.loads(row[0])
                    # Convert to ConsciousnessMetrics-like object
                    from services.consciousness.metrics import ConsciousnessMetrics
                    return ConsciousnessMetrics(
                        coherence=metrics_dict.get("coherence", 0.0),
                        tension=metrics_dict.get("tension", 0.0),
                        fit=metrics_dict.get("fit", 0.0),
                        pressure=metrics_dict.get("pressure", 0.0)
                    )
                except (json.JSONDecodeError, KeyError):
                    return None

            return None

    async def store_consciousness_metrics(
        self,
        session_id: str,
        turn_number: int,
        metrics: Dict[str, Any],
        adjustments: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store consciousness metrics and adjustments for a session turn.

        Args:
            session_id: Session UUID
            turn_number: Turn/message number in the session
            metrics: ConsciousnessMetrics dict (coherence, tension, fit, pressure, stability_score)
            adjustments: Optional adjustment suggestions (tau_c_multiplier, rho_shift, etc.)
        """
        import json

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO session_adjustments (session_id, turn_number, metrics, adjustments, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_number,
                    json.dumps(metrics),
                    json.dumps(adjustments) if adjustments else "{}",
                    time.time()
                )
            )
            await conn.commit()

    # ========================================================================
    # UTILITIES
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with counts for all tables
        """
        async with self.connection() as conn:
            stats = {}

            for table in ['concepts', 'relations', 'sessions', 'events', 'trajectories', 'profiles']:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = await cursor.fetchone()
                stats[table] = count[0]

            # Database file size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

            return stats

    async def vacuum(self) -> None:
        """
        Optimize database (reclaim space, rebuild indexes).

        Run periodically (e.g., weekly) for maintenance.
        """
        async with self.connection() as conn:
            await conn.execute("VACUUM")
            await conn.execute("ANALYZE")
            await conn.commit()

        print("[ISpaceDB] Database optimized (VACUUM + ANALYZE)")

    # ========================================================================
    # GRAPH DELTA OPERATIONS (Lyra-ACE)
    # ========================================================================

    async def apply_delta(
        self,
        delta: GraphDelta,
        session_id: Optional[str] = None,
        kappa_alpha: float = 0.5
    ) -> GraphDelta:
        """
        Applique un delta atomique au graphe.

        Args:
            delta: Delta à appliquer
            session_id: Session associée (pour audit)
            kappa_alpha: Coefficient α pour calcul κ hybride

        Returns:
            Delta enrichi avec old_values et new_kappa

        Raises:
            DeltaValidationError: Si le delta est invalide
            ValueError: Si l'opération échoue (ex: nœud inexistant)
        """
        if not delta.validate():
            raise DeltaValidationError(f"Invalid delta: {delta}")

        calculator = KappaCalculator(alpha=kappa_alpha)

        async with self.connection() as conn:
            try:
                # Récupérer les anciennes valeurs si modification
                if delta.operation in {DeltaOperation.UPDATE_EDGE, DeltaOperation.DELETE_EDGE}:
                    cursor = await conn.execute(
                        "SELECT weight, kappa FROM relations WHERE source = ? AND target = ?",
                        (delta.source, delta.target)
                    )
                    row = await cursor.fetchone()
                    if row:
                        delta.old_weight = row[0]
                        delta.old_kappa = row[1]
                    elif delta.operation == DeltaOperation.UPDATE_EDGE:
                        raise ValueError(f"Edge {delta.source} -> {delta.target} not found for update")

                # Appliquer l'opération
                if delta.operation == DeltaOperation.ADD_NODE:
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO concepts (id, rho_static, degree, created_at)
                        VALUES (?, 0.0, 0, ?)
                        """,
                        (delta.source, time.time())
                    )

                elif delta.operation == DeltaOperation.ADD_EDGE:
                    # Calculer κ pour la nouvelle arête
                    kappa_data = await self._compute_kappa_for_edge(
                        conn, delta.source, delta.target, delta.weight, calculator
                    )
                    delta.new_kappa = kappa_data["kappa_hybrid"]

                    await conn.execute(
                        """
                        INSERT INTO relations (source, target, weight, kappa, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (delta.source, delta.target, delta.weight, delta.new_kappa, time.time())
                    )

                    # Mettre à jour les degrés
                    await self._update_degrees(conn, delta.source, delta.target, increment=True)

                elif delta.operation == DeltaOperation.UPDATE_EDGE:
                    kappa_data = await self._compute_kappa_for_edge(
                        conn, delta.source, delta.target, delta.weight, calculator
                    )
                    delta.new_kappa = kappa_data["kappa_hybrid"]

                    await conn.execute(
                        """
                        UPDATE relations SET weight = ?, kappa = ?
                        WHERE source = ? AND target = ?
                        """,
                        (delta.weight, delta.new_kappa, delta.source, delta.target)
                    )

                elif delta.operation == DeltaOperation.DELETE_EDGE:
                    await conn.execute(
                        "DELETE FROM relations WHERE source = ? AND target = ?",
                        (delta.source, delta.target)
                    )
                    await self._update_degrees(conn, delta.source, delta.target, increment=False)

                elif delta.operation == DeltaOperation.DELETE_NODE:
                    # Supprimer le nœud (CASCADE supprime les arêtes)
                    await conn.execute(
                        "DELETE FROM concepts WHERE id = ?",
                        (delta.source,)
                    )

                # Enregistrer le delta dans l'historique
                delta.applied_at = time.time()
                cursor = await conn.execute(
                    """
                    INSERT INTO graph_deltas (
                        session_id, operation, source, target,
                        old_weight, new_weight, old_kappa, new_kappa,
                        confidence, model_source, reason, timestamp, applied_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, delta.operation.value, delta.source, delta.target,
                        delta.old_weight, delta.weight, delta.old_kappa, delta.new_kappa,
                        delta.confidence, delta.model_source, delta.reason,
                        delta.timestamp, delta.applied_at
                    )
                )
                delta.delta_id = cursor.lastrowid

                await conn.commit()
                return delta

            except Exception as e:
                await conn.rollback()
                raise

    async def apply_delta_batch(
        self,
        batch: DeltaBatch,
        kappa_alpha: float = 0.5
    ) -> List[GraphDelta]:
        """
        Applique un lot de deltas avec vérification de limite.

        Args:
            batch: Lot de deltas
            kappa_alpha: Coefficient α pour κ hybride

        Returns:
            Liste des deltas appliqués (enrichis)

        Raises:
            MutationLimitExceededError: Si le batch dépasse 5% du graphe
        """
        # Vérifier la taille du graphe
        stats = await self.get_stats()
        graph_size = stats["concepts"] + stats["relations"]

        if not batch.validate_batch_size(graph_size):
            max_allowed = int(graph_size * batch.max_mutation_ratio)
            raise MutationLimitExceededError(
                f"Batch size {len(batch.deltas)} exceeds limit {max_allowed} "
                f"({batch.max_mutation_ratio*100}% of {graph_size} elements)"
            )

        applied = []
        for delta in batch.deltas:
            result = await self.apply_delta(delta, batch.session_id, kappa_alpha)
            applied.append(result)

        return applied

    async def _compute_kappa_for_edge(
        self,
        conn,
        source: str,
        target: str,
        weight: float,
        calculator: KappaCalculator
    ) -> Dict[str, float]:
        """Calcule κ hybride pour une arête."""
        # Récupérer les degrés
        cursor = await conn.execute(
            "SELECT degree FROM concepts WHERE id = ?", (source,)
        )
        row = await cursor.fetchone()
        degree_u = row[0] if row else 0

        cursor = await conn.execute(
            "SELECT degree FROM concepts WHERE id = ?", (target,)
        )
        row = await cursor.fetchone()
        degree_v = row[0] if row else 0

        # Récupérer les voisins pour Jaccard
        cursor = await conn.execute(
            "SELECT target FROM relations WHERE source = ?", (source,)
        )
        neighbors_u = {row[0] for row in await cursor.fetchall()}

        cursor = await conn.execute(
            "SELECT target FROM relations WHERE source = ?", (target,)
        )
        neighbors_v = {row[0] for row in await cursor.fetchall()}

        return calculator.compute_hybrid(
            degree_u, degree_v, weight, neighbors_u, neighbors_v
        )

    async def _update_degrees(
        self,
        conn,
        source: str,
        target: str,
        increment: bool
    ):
        """Met à jour les degrés des nœuds après ajout/suppression d'arête."""
        delta = 1 if increment else -1
        await conn.execute(
            "UPDATE concepts SET degree = MAX(0, degree + ?) WHERE id = ?",
            (delta, source)
        )
        await conn.execute(
            "UPDATE concepts SET degree = MAX(0, degree + ?) WHERE id = ?",
            (delta, target)
        )

    async def compute_kappa_live(
        self,
        source: str,
        target: str,
        kappa_alpha: float = 0.5,
        store_history: bool = False
    ) -> Optional[Dict[str, float]]:
        """
        Calcule κ en temps réel pour une arête existante.

        Args:
            source: Concept source
            target: Concept cible
            kappa_alpha: Coefficient α hybride
            store_history: Si True, enregistre dans kappa_history

        Returns:
            Dict avec kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha
            None si l'arête n'existe pas
        """
        calculator = KappaCalculator(alpha=kappa_alpha)

        async with self.connection() as conn:
            # Vérifier que l'arête existe
            cursor = await conn.execute(
                "SELECT weight FROM relations WHERE source = ? AND target = ?",
                (source, target)
            )
            row = await cursor.fetchone()
            if not row:
                return None

            weight = row[0]
            kappa_data = await self._compute_kappa_for_edge(
                conn, source, target, weight, calculator
            )

            if store_history:
                await conn.execute(
                    """
                    INSERT INTO kappa_history
                    (source, target, kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source, target,
                        kappa_data["kappa_ollivier"],
                        kappa_data["kappa_jaccard"],
                        kappa_data["kappa_hybrid"],
                        kappa_data["alpha"],
                        time.time()
                    )
                )
                await conn.commit()

            return kappa_data

    async def get_delta_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
        include_rolled_back: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des deltas.

        Args:
            session_id: Filtrer par session (None = tous)
            limit: Nombre max de résultats
            include_rolled_back: Inclure les deltas annulés

        Returns:
            Liste de deltas (plus récents en premier)
        """
        async with self.connection() as conn:
            where_clauses = []
            params = []

            if session_id:
                where_clauses.append("session_id = ?")
                params.append(session_id)

            if not include_rolled_back:
                where_clauses.append("rolled_back_at IS NULL")

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cursor = await conn.execute(
                f"""
                SELECT delta_id, session_id, operation, source, target,
                       old_weight, new_weight, old_kappa, new_kappa,
                       confidence, model_source, reason, timestamp,
                       applied_at, rolled_back_at
                FROM graph_deltas
                {where_sql}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (*params, limit)
            )

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def rollback_deltas(
        self,
        session_id: str,
        to_timestamp: Optional[float] = None,
        delta_ids: Optional[List[int]] = None
    ) -> int:
        """
        Annule des deltas (restore état précédent).

        Args:
            session_id: Session dont annuler les deltas
            to_timestamp: Annuler tous les deltas après ce timestamp
            delta_ids: Ou liste explicite de delta_ids à annuler

        Returns:
            Nombre de deltas annulés

        Note:
            Les deltas sont annulés en ordre inverse (LIFO)
        """
        async with self.connection() as conn:
            # Récupérer les deltas à annuler
            if delta_ids:
                placeholders = ','.join('?' * len(delta_ids))
                cursor = await conn.execute(
                    f"""
                    SELECT delta_id, operation, source, target, old_weight, old_kappa
                    FROM graph_deltas
                    WHERE delta_id IN ({placeholders})
                      AND session_id = ?
                      AND rolled_back_at IS NULL
                    ORDER BY timestamp DESC
                    """,
                    (*delta_ids, session_id)
                )
            elif to_timestamp:
                cursor = await conn.execute(
                    """
                    SELECT delta_id, operation, source, target, old_weight, old_kappa
                    FROM graph_deltas
                    WHERE session_id = ?
                      AND timestamp > ?
                      AND rolled_back_at IS NULL
                    ORDER BY timestamp DESC
                    """,
                    (session_id, to_timestamp)
                )
            else:
                return 0

            deltas_to_rollback = await cursor.fetchall()
            rollback_count = 0
            rollback_time = time.time()

            for row in deltas_to_rollback:
                delta_id, operation, source, target, old_weight, old_kappa = row

                # Inverser l'opération
                if operation == DeltaOperation.ADD_EDGE.value:
                    await conn.execute(
                        "DELETE FROM relations WHERE source = ? AND target = ?",
                        (source, target)
                    )
                    await self._update_degrees(conn, source, target, increment=False)

                elif operation == DeltaOperation.DELETE_EDGE.value and old_weight is not None:
                    await conn.execute(
                        """
                        INSERT INTO relations (source, target, weight, kappa, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (source, target, old_weight, old_kappa or 0.5, time.time())
                    )
                    await self._update_degrees(conn, source, target, increment=True)

                elif operation == DeltaOperation.UPDATE_EDGE.value and old_weight is not None:
                    await conn.execute(
                        "UPDATE relations SET weight = ?, kappa = ? WHERE source = ? AND target = ?",
                        (old_weight, old_kappa or 0.5, source, target)
                    )

                elif operation == DeltaOperation.ADD_NODE.value:
                    await conn.execute(
                        "DELETE FROM concepts WHERE id = ?",
                        (source,)
                    )

                # Marquer comme rolled back
                await conn.execute(
                    "UPDATE graph_deltas SET rolled_back_at = ? WHERE delta_id = ?",
                    (rollback_time, delta_id)
                )
                rollback_count += 1

            await conn.commit()
            return rollback_count

    async def get_graph_mutation_stats(self) -> Dict[str, Any]:
        """
        Statistiques sur les mutations du graphe.

        Returns:
            Dict avec counts par opération, taux de rollback, etc.
        """
        async with self.connection() as conn:
            stats = {}

            # Comptage par opération
            cursor = await conn.execute(
                """
                SELECT operation, COUNT(*) as count,
                       SUM(CASE WHEN rolled_back_at IS NOT NULL THEN 1 ELSE 0 END) as rolled_back
                FROM graph_deltas
                GROUP BY operation
                """
            )
            rows = await cursor.fetchall()
            stats["by_operation"] = {row[0]: {"total": row[1], "rolled_back": row[2]} for row in rows}

            # Comptage par modèle source
            cursor = await conn.execute(
                """
                SELECT model_source, COUNT(*) as count
                FROM graph_deltas
                WHERE rolled_back_at IS NULL
                GROUP BY model_source
                """
            )
            rows = await cursor.fetchall()
            stats["by_model"] = {row[0]: row[1] for row in rows}

            # Stats temporelles
            cursor = await conn.execute(
                """
                SELECT COUNT(*), AVG(confidence)
                FROM graph_deltas
                WHERE rolled_back_at IS NULL
                  AND timestamp > ?
                """,
                (time.time() - 86400,)  # Dernières 24h
            )
            row = await cursor.fetchone()
            stats["last_24h"] = {
                "count": row[0],
                "avg_confidence": round(row[1], 3) if row[1] else 0
            }

            return stats

    # ========================================================================
    # CANONICALISATION (Aliases)
    # ========================================================================

    async def resolve_concept(self, concept: str) -> str:
        """
        Resout un concept vers sa forme canonique.

        Args:
            concept: Concept brut (ex: "Intelligence Artificielle")

        Returns:
            Concept canonique (ex: "ia") ou le concept original si pas d'alias
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                "SELECT canonical_id FROM concept_aliases WHERE alias = ?",
                (concept.lower().strip(),)
            )
            row = await cursor.fetchone()
            return row[0] if row else concept.lower().strip()

    async def add_alias(
        self,
        alias: str,
        canonical_id: str,
        similarity: float,
        method: str = "embedding"
    ) -> None:
        """
        Ajoute un alias pour un concept canonique.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO concept_aliases
                (alias, canonical_id, similarity, fusion_method, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alias.lower().strip(), canonical_id, similarity, method, time.time())
            )
            await conn.commit()

    async def get_concept_with_aliases(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupere un concept avec tous ses aliases.
        """
        async with self.connection() as conn:
            # Concept principal
            cursor = await conn.execute(
                "SELECT * FROM concepts WHERE id = ?", (concept_id,)
            )
            row = await cursor.fetchone()
            concept = dict(row) if row else None

            if not concept:
                return None

            # Aliases
            cursor = await conn.execute(
                "SELECT alias, similarity FROM concept_aliases WHERE canonical_id = ?",
                (concept_id,)
            )
            aliases = [{"alias": row[0], "similarity": row[1]} for row in await cursor.fetchall()]
            concept["aliases"] = aliases

            return concept

    # ========================================================================
    # CALCUL KAPPA DIFFERE
    # ========================================================================

    async def queue_kappa_recalc(
        self,
        source: str,
        target: str,
        priority: int = 0
    ) -> None:
        """
        Ajoute une arete a la queue de recalcul kappa Ollivier.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO pending_kappa_recalc
                (source, target, priority, queued_at)
                VALUES (?, ?, ?, ?)
                """,
                (source, target, priority, time.time())
            )
            await conn.commit()

    async def get_pending_kappa_batch(self, limit: int = 100) -> List[Dict]:
        """
        Recupere un batch d'aretes en attente de recalcul kappa.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT source, target, priority, queued_at, attempts
                FROM pending_kappa_recalc
                ORDER BY priority DESC, queued_at ASC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def mark_kappa_recalc_done(self, source: str, target: str) -> None:
        """
        Supprime une arete de la queue apres recalcul reussi.
        """
        async with self.connection() as conn:
            await conn.execute(
                "DELETE FROM pending_kappa_recalc WHERE source = ? AND target = ?",
                (source, target)
            )
            await conn.commit()

    async def mark_kappa_recalc_failed(
        self,
        source: str,
        target: str,
        error: str
    ) -> None:
        """
        Marque un echec de recalcul (incremente attempts).
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE pending_kappa_recalc
                SET attempts = attempts + 1, last_error = ?
                WHERE source = ? AND target = ?
                """,
                (error, source, target)
            )
            await conn.commit()

    # ========================================================================
    # RELATIONS CANONIQUES
    # ========================================================================

    async def get_canonical_relation(self, relation: str) -> Optional[str]:
        """
        Normalise un type de relation vers sa forme canonique.

        Args:
            relation: Relation brute (ex: "provoque", "engendre")

        Returns:
            Relation canonique (ex: "cause") ou None si non trouvee
        """
        relation_lower = relation.lower().strip()

        async with self.connection() as conn:
            # Chercher directement
            cursor = await conn.execute(
                "SELECT canonical FROM canonical_relations WHERE canonical = ?",
                (relation_lower,)
            )
            if await cursor.fetchone():
                return relation_lower

            # Chercher dans les aliases (JSON array)
            cursor = await conn.execute(
                "SELECT canonical, aliases FROM canonical_relations"
            )
            for row in await cursor.fetchall():
                aliases = json.loads(row[1])
                if relation_lower in [a.lower() for a in aliases]:
                    return row[0]

            return None  # Relation inconnue

    async def get_all_canonical_relations(self) -> List[Dict]:
        """
        Recupere toutes les relations canoniques avec leurs metadonnees.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM canonical_relations"
            )
            return [dict(row) for row in await cursor.fetchall()]

    # ========================================================================
    # ESMM: RUNS
    # ========================================================================

    async def create_esmm_run(
        self,
        config: Dict,
        models: List[str],
        seed_type: str = "standard"
    ) -> int:
        """
        Cree un nouveau run ESMM.

        Returns:
            run_id
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO esmm_runs (config, models_used, seed_type, status, started_at)
                VALUES (?, ?, ?, 'initializing', ?)
                """,
                (json.dumps(config), json.dumps(models), seed_type, time.time())
            )
            await conn.commit()
            return cursor.lastrowid

    async def update_esmm_run_status(
        self,
        run_id: int,
        status: str,
        current_cycle: str = None,
        current_iteration: int = None,
        error_message: str = None
    ) -> None:
        """
        Met a jour le statut d'un run ESMM.
        """
        async with self.connection() as conn:
            updates = ["status = ?"]
            params = [status]

            if current_cycle is not None:
                updates.append("current_cycle = ?")
                params.append(current_cycle)
            if current_iteration is not None:
                updates.append("current_iteration = ?")
                params.append(current_iteration)
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            if status == "completed":
                updates.append("completed_at = ?")
                params.append(time.time())

            params.append(run_id)

            await conn.execute(
                f"UPDATE esmm_runs SET {', '.join(updates)} WHERE run_id = ?",
                params
            )
            await conn.commit()

    async def finalize_esmm_run(
        self,
        run_id: int,
        stats: Dict[str, Any]
    ) -> None:
        """
        Finalise un run ESMM avec les statistiques finales.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE esmm_runs SET
                    status = 'completed',
                    completed_at = ?,
                    cycles_completed = ?,
                    total_questions = ?,
                    total_triplets = ?,
                    triplets_injected = ?,
                    concepts_created = ?,
                    relations_created = ?,
                    final_cochain_size = ?,
                    coverage_score = ?,
                    consensus_density = ?,
                    epistemic_diversity = ?,
                    structural_stability = ?
                WHERE run_id = ?
                """,
                (
                    time.time(),
                    stats.get("cycles_completed", 0),
                    stats.get("total_questions", 0),
                    stats.get("total_triplets", 0),
                    stats.get("triplets_injected", 0),
                    stats.get("concepts_created", 0),
                    stats.get("relations_created", 0),
                    stats.get("final_cochain_size"),
                    stats.get("coverage_score"),
                    stats.get("consensus_density"),
                    stats.get("epistemic_diversity"),
                    stats.get("structural_stability"),
                    run_id
                )
            )
            await conn.commit()

    # ========================================================================
    # ESMM: CYCLES
    # ========================================================================

    async def log_exploration_cycle(
        self,
        run_id: int,
        cycle_type: str,
        iteration: int,
        question_template: str,
        question_rendered: str,
        responses: Dict[str, str],
        target_concepts: List[str] = None,
        response_latencies: Dict[str, float] = None
    ) -> int:
        """
        Enregistre un cycle d'exploration.

        Returns:
            cycle_id
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO exploration_cycles (
                    run_id, cycle_type, iteration, question_template, question_rendered,
                    target_concepts, responses, response_latencies, started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, cycle_type, iteration, question_template, question_rendered,
                    json.dumps(target_concepts) if target_concepts else None,
                    json.dumps(responses),
                    json.dumps(response_latencies) if response_latencies else None,
                    time.time()
                )
            )
            await conn.commit()
            return cursor.lastrowid

    async def update_cycle_extraction(
        self,
        cycle_id: int,
        triplets_extracted: int,
        triplets_data: List[Dict],
        consensus_map: Dict[str, float],
        exploration_metrics: Dict[str, float]
    ) -> None:
        """
        Met a jour un cycle avec les resultats d'extraction.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE exploration_cycles SET
                    triplets_extracted = ?,
                    triplets_data = ?,
                    consensus_map = ?,
                    exploration_metrics = ?,
                    completed_at = ?
                WHERE cycle_id = ?
                """,
                (
                    triplets_extracted,
                    json.dumps(triplets_data),
                    json.dumps(consensus_map),
                    json.dumps(exploration_metrics),
                    time.time(),
                    cycle_id
                )
            )
            await conn.commit()

    # ========================================================================
    # ESMM: TRIPLETS
    # ========================================================================

    async def store_triplet_extraction(
        self,
        subject: str,
        relation: str,
        object_: str,
        confidence: float,
        extraction_method: str,
        model_source: str,
        cycle_id: int = None,
        event_id: int = None,
        source_text: str = None
    ) -> int:
        """
        Stocke un triplet extrait (avant injection dans le graphe).

        Returns:
            extraction_id
        """
        async with self.connection() as conn:
            # Canonicaliser
            subject_canonical = await self.resolve_concept(subject)
            object_canonical = await self.resolve_concept(object_)
            relation_canonical = await self.get_canonical_relation(relation) or relation.lower()

            cursor = await conn.execute(
                """
                INSERT INTO triplet_extractions (
                    cycle_id, event_id, subject, subject_canonical,
                    relation, relation_canonical, object, object_canonical,
                    confidence, extraction_method, model_source, source_text,
                    extracted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id, event_id, subject, subject_canonical,
                    relation, relation_canonical, object_, object_canonical,
                    confidence, extraction_method, model_source,
                    source_text[:100] if source_text else None,
                    time.time()
                )
            )
            await conn.commit()
            return cursor.lastrowid

    async def mark_triplet_injected(
        self,
        extraction_id: int,
        delta_id: int
    ) -> None:
        """
        Marque un triplet comme injecte dans le graphe.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE triplet_extractions
                SET injected_to_graph = 1, delta_id = ?
                WHERE extraction_id = ?
                """,
                (delta_id, extraction_id)
            )
            await conn.commit()

    async def skip_triplet_injection(
        self,
        extraction_id: int,
        reason: str
    ) -> None:
        """
        Marque un triplet comme non-injecte avec raison.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE triplet_extractions
                SET injection_skipped_reason = ?
                WHERE extraction_id = ?
                """,
                (reason, extraction_id)
            )
            await conn.commit()

    # ========================================================================
    # ESMM: COCHAIN (0-Cochaine)
    # ========================================================================

    async def upsert_cochain_entry(
        self,
        concept_id: str,
        consensus_score: float,
        model_agreement: float,
        semantic_consistency: float,
        structural_centrality: float,
        stability_score: float,
        signature_vector: List[float],
        epistemic_type: str,
        contributing_models: Dict[str, float],
        triplet_count: int,
        run_id: int = None
    ) -> None:
        """
        Insere ou met a jour une entree de la 0-cochaine.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO cochain_entries (
                    concept_id, consensus_score, model_agreement, semantic_consistency,
                    structural_centrality, stability_score, signature_vector,
                    epistemic_type, contributing_models, triplet_count, run_id, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(concept_id) DO UPDATE SET
                    consensus_score = excluded.consensus_score,
                    model_agreement = excluded.model_agreement,
                    semantic_consistency = excluded.semantic_consistency,
                    structural_centrality = excluded.structural_centrality,
                    stability_score = excluded.stability_score,
                    signature_vector = excluded.signature_vector,
                    epistemic_type = excluded.epistemic_type,
                    contributing_models = excluded.contributing_models,
                    triplet_count = excluded.triplet_count,
                    run_id = excluded.run_id,
                    computed_at = excluded.computed_at
                """,
                (
                    concept_id, consensus_score, model_agreement, semantic_consistency,
                    structural_centrality, stability_score, json.dumps(signature_vector),
                    epistemic_type, json.dumps(contributing_models), triplet_count,
                    run_id, time.time()
                )
            )
            await conn.commit()

    async def get_cochain_entry(self, concept_id: str) -> Optional[Dict]:
        """
        Recupere une entree de la cochaine.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM cochain_entries WHERE concept_id = ?",
                (concept_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            entry = dict(row)
            entry["signature_vector"] = json.loads(entry["signature_vector"])
            entry["contributing_models"] = json.loads(entry["contributing_models"])
            return entry

    async def get_cochain_by_type(
        self,
        epistemic_type: str,
        min_consensus: float = 0.0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Recupere les entrees de cochaine par type epistemique.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM cochain_entries
                WHERE epistemic_type = ? AND consensus_score >= ?
                ORDER BY consensus_score DESC
                LIMIT ?
                """,
                (epistemic_type, min_consensus, limit)
            )
            entries = []
            for row in await cursor.fetchall():
                entry = dict(row)
                entry["signature_vector"] = json.loads(entry["signature_vector"])
                entry["contributing_models"] = json.loads(entry["contributing_models"])
                entries.append(entry)
            return entries

    async def export_cochain_for_viz(self) -> List[Dict]:
        """
        Exporte la cochaine pour visualisation externe.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT concept_id, consensus_score, epistemic_type, signature_vector
                FROM cochain_entries
                ORDER BY consensus_score DESC
                """
            )
            points = []
            for row in await cursor.fetchall():
                sig = json.loads(row[3])
                points.append({
                    "id": row[0],
                    "consensus": row[1],
                    "type": row[2],
                    "x": sig[0] if len(sig) > 0 else 0,
                    "y": sig[1] if len(sig) > 1 else 0,
                    "z": sig[2] if len(sig) > 2 else 0
                })
            return points

    # ========================================================================
    # ESMM: KNOWLEDGE GAPS
    # ========================================================================

    async def add_knowledge_gap(
        self,
        gap_type: str,
        details: Dict,
        priority: float,
        run_id: int = None
    ) -> int:
        """
        Ajoute une lacune de connaissance identifiee.

        Returns:
            gap_id
        """
        async with self.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO knowledge_gaps (run_id, gap_type, details, priority, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, gap_type, json.dumps(details), priority, time.time())
            )
            await conn.commit()
            return cursor.lastrowid

    async def get_active_gaps(
        self,
        gap_type: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Recupere les lacunes non-adressees.
        """
        async with self.connection() as conn:
            if gap_type:
                cursor = await conn.execute(
                    """
                    SELECT * FROM knowledge_gaps
                    WHERE addressed = 0 AND gap_type = ?
                    ORDER BY priority DESC LIMIT ?
                    """,
                    (gap_type, limit)
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM knowledge_gaps
                    WHERE addressed = 0
                    ORDER BY priority DESC LIMIT ?
                    """,
                    (limit,)
                )

            gaps = []
            for row in await cursor.fetchall():
                gap = dict(row)
                gap["details"] = json.loads(gap["details"])
                gaps.append(gap)
            return gaps

    async def mark_gap_addressed(
        self,
        gap_id: int,
        cycle_id: int
    ) -> None:
        """
        Marque une lacune comme adressee.
        """
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE knowledge_gaps
                SET addressed = 1, addressed_at = ?, addressed_by_cycle_id = ?
                WHERE gap_id = ?
                """,
                (time.time(), cycle_id, gap_id)
            )
            await conn.commit()


# ============================================================================
# SINGLETON INSTANCE (Dependency Injection ready)
# ============================================================================

_db_instance: Optional[ISpaceDB] = None


async def get_db(db_path: str = "data/ispace.db") -> ISpaceDB:
    """
    Get or create database instance (singleton pattern).

    Usage:
        db = await get_db()
        concepts = await db.get_neighbors("entropy")
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = ISpaceDB(db_path)
        await _db_instance.initialize()

    return _db_instance


async def close_db() -> None:
    """
    Close database connection (cleanup).

    Call on application shutdown.
    """
    global _db_instance
    _db_instance = None
    print("[ISpaceDB] Connection closed")
