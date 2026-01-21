"""
LYRA-ACE - CONNECTION POOL & PERFORMANCE UTILITIES
===================================================

Optimisations de performance pour SQLite:
- Pool de connexions asynchrones
- Cache LRU pour les concepts
- Semaphore pour le controle de concurrence
- Logging structure

Auteur: Optimisation Lyra-ACE
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any, List, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

# Import conditionnel pour cachetools et structlog
try:
    from cachetools import TTLCache
    HAS_CACHETOOLS = True
except ImportError:
    HAS_CACHETOOLS = False
    TTLCache = None

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None


# ============================================================================
# STRUCTURED LOGGING
# ============================================================================

def get_logger(name: str = "lyra"):
    """
    Retourne un logger structure ou un fallback simple.
    """
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        # Fallback simple
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


def configure_logging():
    """
    Configure structlog pour un logging structure JSON-compatible.
    """
    if not HAS_STRUCTLOG:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# ============================================================================
# CONNECTION POOL
# ============================================================================

@dataclass
class PooledConnection:
    """Wrapper pour une connexion poolee."""
    connection: aiosqlite.Connection
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    in_use: bool = False


class SQLiteConnectionPool:
    """
    Pool de connexions SQLite asynchrones.

    Caracteristiques:
    - Pre-allocation de connexions
    - Reutilisation pour eviter overhead open/close
    - Timeout sur acquisition
    - Nettoyage des connexions inactives

    Usage:
        pool = SQLiteConnectionPool("data/ispace.db", pool_size=10)
        await pool.initialize()

        async with pool.acquire() as conn:
            cursor = await conn.execute("SELECT ...")
    """

    def __init__(
        self,
        db_path: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0  # 5 minutes
    ):
        """
        Args:
            db_path: Chemin vers la base SQLite
            pool_size: Nombre de connexions dans le pool
            max_overflow: Connexions supplementaires en cas de surcharge
            connection_timeout: Timeout pour acquerir une connexion
            idle_timeout: Temps avant fermeture d'une connexion inactive
        """
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout

        self._pool: List[PooledConnection] = []
        self._overflow_count: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False
        self._logger = get_logger("pool")

        # Pragmas de performance
        self._pragmas = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",
            "PRAGMA temp_store=MEMORY",
            "PRAGMA mmap_size=268435456",
            "PRAGMA busy_timeout=30000",
        ]

    async def initialize(self) -> None:
        """Initialise le pool avec les connexions de base."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Creer les connexions initiales
        for _ in range(self.pool_size):
            conn = await self._create_connection()
            self._pool.append(PooledConnection(connection=conn))

        self._initialized = True
        self._logger.info("pool_initialized", size=self.pool_size, path=str(self.db_path))

    async def _create_connection(self) -> aiosqlite.Connection:
        """Cree une nouvelle connexion avec les pragmas."""
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row

        # Appliquer les pragmas
        for pragma in self._pragmas:
            await conn.execute(pragma)

        return conn

    @asynccontextmanager
    async def acquire(self):
        """
        Acquiert une connexion du pool.

        Raises:
            asyncio.TimeoutError: Si timeout depasse
        """
        start_time = time.time()
        pooled_conn = None

        while True:
            async with self._lock:
                # Chercher une connexion disponible
                for pc in self._pool:
                    if not pc.in_use:
                        pc.in_use = True
                        pc.last_used = time.time()
                        pooled_conn = pc
                        break

                # Si pas de connexion disponible, creer overflow si possible
                if pooled_conn is None and self._overflow_count < self.max_overflow:
                    conn = await self._create_connection()
                    pooled_conn = PooledConnection(connection=conn, in_use=True)
                    self._pool.append(pooled_conn)
                    self._overflow_count += 1
                    self._logger.debug("overflow_connection_created", overflow_count=self._overflow_count)

            if pooled_conn:
                break

            # Verifier timeout
            if time.time() - start_time > self.connection_timeout:
                self._logger.error("connection_timeout", waited=time.time() - start_time)
                raise asyncio.TimeoutError("Could not acquire connection from pool")

            # Attendre un peu avant de reessayer
            await asyncio.sleep(0.01)

        try:
            yield pooled_conn.connection
        finally:
            async with self._lock:
                pooled_conn.in_use = False
                pooled_conn.last_used = time.time()

    async def close(self) -> None:
        """Ferme toutes les connexions du pool."""
        async with self._lock:
            for pc in self._pool:
                try:
                    await pc.connection.close()
                except Exception:
                    pass
            self._pool.clear()
            self._overflow_count = 0
            self._initialized = False

        self._logger.info("pool_closed")

    async def cleanup_idle(self) -> int:
        """
        Ferme les connexions inactives depuis idle_timeout.

        Returns:
            Nombre de connexions fermees
        """
        now = time.time()
        closed = 0

        async with self._lock:
            # Garder au moins pool_size connexions
            to_remove = []
            for pc in self._pool:
                if not pc.in_use and len(self._pool) - len(to_remove) > self.pool_size:
                    if now - pc.last_used > self.idle_timeout:
                        to_remove.append(pc)

            for pc in to_remove:
                try:
                    await pc.connection.close()
                except Exception:
                    pass
                self._pool.remove(pc)
                closed += 1

        if closed > 0:
            self._logger.info("idle_connections_closed", count=closed)

        return closed

    @property
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du pool."""
        in_use = sum(1 for pc in self._pool if pc.in_use)
        return {
            "total": len(self._pool),
            "in_use": in_use,
            "available": len(self._pool) - in_use,
            "overflow": self._overflow_count,
            "initialized": self._initialized
        }


# ============================================================================
# LRU CACHE FOR CONCEPTS
# ============================================================================

class ConceptCache:
    """
    Cache LRU pour les concepts avec TTL.

    Reduit la charge sur SQLite pour les requetes frequentes.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl: int = 3600  # 1 heure
    ):
        """
        Args:
            maxsize: Nombre maximum d'entrees
            ttl: Time-to-live en secondes
        """
        self._logger = get_logger("cache")

        if HAS_CACHETOOLS:
            self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            # Fallback: dict simple sans TTL (moins optimal)
            self._cache: Dict[str, Any] = {}
            self._maxsize = maxsize
            self._logger.warning("cachetools_not_available", fallback="simple_dict")

        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Recupere une valeur du cache."""
        value = self._cache.get(key)
        if value is not None:
            self._hits += 1
            return value
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Ajoute une valeur au cache."""
        if not HAS_CACHETOOLS and len(self._cache) >= self._maxsize:
            # Eviction simple: supprimer le premier
            try:
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            except StopIteration:
                pass
        self._cache[key] = value

    def delete(self, key: str) -> None:
        """Supprime une entree du cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Vide le cache."""
        self._cache.clear()
        self._logger.info("cache_cleared")

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalide les entrees matchant un pattern.

        Args:
            pattern: Prefix des cles a invalider

        Returns:
            Nombre d'entrees invalidees
        """
        to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
        for k in to_delete:
            del self._cache[k]
        return len(to_delete)

    @property
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


# ============================================================================
# CONCURRENCY CONTROL
# ============================================================================

class ConcurrencyLimiter:
    """
    Limiteur de concurrence via Semaphore.

    Evite la surcharge de la base avec trop de requetes simultanees.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: float = 30.0
    ):
        """
        Args:
            max_concurrent: Nombre maximum de requetes simultanees
            timeout: Timeout pour acquerir le semaphore
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self._logger = get_logger("limiter")
        self._active = 0
        self._total = 0
        self._timeouts = 0

    @asynccontextmanager
    async def acquire(self):
        """Acquiert un slot d'execution."""
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._timeout
            )
            if acquired:
                self._active += 1
                self._total += 1
        except asyncio.TimeoutError:
            self._timeouts += 1
            self._logger.warning("concurrency_timeout", active=self._active)
            raise

        try:
            yield
        finally:
            self._semaphore.release()
            self._active -= 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        return {
            "active": self._active,
            "total_requests": self._total,
            "timeouts": self._timeouts
        }


# ============================================================================
# SQL VALIDATION
# ============================================================================

class SQLValidator:
    """
    Validateur d'entrees SQL pour prevenir les injections.
    """

    # Patterns dangereux
    DANGEROUS_PATTERNS = [
        "';",
        "--",
        "/*",
        "*/",
        "DROP ",
        "DELETE ",
        "TRUNCATE ",
        "ALTER ",
        "EXEC ",
        "EXECUTE ",
        "xp_",
        "sp_",
    ]

    @classmethod
    def validate_identifier(cls, value: str) -> bool:
        """
        Valide un identifiant (table, colonne).

        Returns:
            True si valide, False sinon
        """
        if not value:
            return False

        # Caracteres autorises: alphanumerique + underscore
        if not value.replace('_', '').isalnum():
            return False

        # Ne doit pas commencer par un chiffre
        if value[0].isdigit():
            return False

        return True

    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """
        Sanitize une chaine pour utilisation dans une requete.

        Echappe les apostrophes et supprime les patterns dangereux.
        """
        if not isinstance(value, str):
            return str(value)

        # Echapper les apostrophes
        sanitized = value.replace("'", "''")

        # Verifier les patterns dangereux
        upper = sanitized.upper()
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.upper() in upper:
                raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")

        return sanitized

    @classmethod
    def validate_concept_id(cls, concept_id: str) -> str:
        """
        Valide et normalise un ID de concept.

        Returns:
            Concept ID normalise

        Raises:
            ValueError: Si l'ID est invalide
        """
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")

        # Normaliser
        normalized = concept_id.lower().strip()

        # Longueur max
        if len(normalized) > 255:
            raise ValueError("Concept ID too long (max 255)")

        # Caracteres autorises: alphanumerique, underscore, tiret, espace
        import re
        if not re.match(r'^[\w\s\-]+$', normalized, re.UNICODE):
            raise ValueError(f"Invalid characters in concept ID: {concept_id}")

        return normalized


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_pool_instance: Optional[SQLiteConnectionPool] = None
_concept_cache: Optional[ConceptCache] = None
_concurrency_limiter: Optional[ConcurrencyLimiter] = None


async def get_pool(
    db_path: str = "data/ispace.db",
    pool_size: int = 10
) -> SQLiteConnectionPool:
    """Retourne l'instance singleton du pool."""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = SQLiteConnectionPool(db_path, pool_size=pool_size)
        await _pool_instance.initialize()
    return _pool_instance


def get_concept_cache(maxsize: int = 1000, ttl: int = 3600) -> ConceptCache:
    """Retourne l'instance singleton du cache."""
    global _concept_cache
    if _concept_cache is None:
        _concept_cache = ConceptCache(maxsize=maxsize, ttl=ttl)
    return _concept_cache


def get_concurrency_limiter(max_concurrent: int = 10) -> ConcurrencyLimiter:
    """Retourne l'instance singleton du limiteur."""
    global _concurrency_limiter
    if _concurrency_limiter is None:
        _concurrency_limiter = ConcurrencyLimiter(max_concurrent=max_concurrent)
    return _concurrency_limiter


async def close_pool() -> None:
    """Ferme le pool de connexions."""
    global _pool_instance
    if _pool_instance:
        await _pool_instance.close()
        _pool_instance = None
