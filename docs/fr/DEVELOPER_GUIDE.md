# Guide Développeur Lyra Clean

Guide complet pour comprendre, modifier et contribuer au framework Lyra Clean.

## Table des matières

1. [Architecture](#architecture)
2. [Structure du code](#structure-du-code)
3. [Composants principaux](#composants-principaux)
4. [Flux de données](#flux-de-données)
5. [Moteur physique Bézier](#moteur-physique-bézier)
6. [Système de conscience](#système-de-conscience)
7. [Base de données](#base-de-données)
8. [Tests et benchmarks](#tests-et-benchmarks)
9. [Contribution](#contribution)
10. [Roadmap](#roadmap)

---

## Architecture

### Vue d'ensemble

Lyra Clean suit une architecture en couches (layered architecture) avec séparation claire des responsabilités :

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│                   (app/main.py, app/api/)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Services Layer                           │
│         (Context Injection, Consciousness, Memory)          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   Database Layer                            │
│            (SQLite Engine, Schema Management)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                  Core Physics Engine                        │
│         (Bézier Trajectories, Parameter Mapping)            │
└─────────────────────────────────────────────────────────────┘
```

### Principes de conception

1. **Deterministic Physics**
   - Paramètres LLM contrôlés par trajectoires mathématiques
   - Comportement prévisible et reproductible
   - Pas de feedback loops réactifs

2. **Separation of Concerns**
   - API layer : Validation, routing
   - Services : Business logic, context injection
   - Database : Persistence, queries
   - Core : Pure math, no side effects

3. **Async-First**
   - Tout est async/await (aiosqlite, httpx)
   - Non-blocking I/O
   - Scalabilité horizontale

4. **Dependency Injection**
   - FastAPI `Depends()` pour DI
   - Singleton pattern pour DB et LLM client
   - Testable et modulaire

5. **Type Safety**
   - Pydantic models pour validation
   - Type hints partout
   - Mypy-friendly (mostly)

---

## Structure du code

```
lyra_clean_bis/
│
├── app/                          # Application FastAPI
│   ├── __init__.py
│   ├── main.py                   # Entry point (369 LOC)
│   ├── models.py                 # Pydantic models (307 LOC)
│   ├── llm_client.py             # Ollama async client (308 LOC)
│   ├── embeddings.py             # Embedding wrapper (91 LOC)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py               # Chat endpoint (350 LOC)
│   │   ├── sessions.py           # Session management (332 LOC)
│   │   ├── graph.py              # [NOUVEAU] API mutations graphe (Lyra-ACE)
│   │   └── multimodel.py         # [NOUVEAU] API multi-modèles
│   └── static/
│       └── index.html            # Web UI
│
├── services/                     # Business logic
│   ├── __init__.py
│   ├── injector.py               # Context injection (442 LOC)
│   ├── entity_resolver.py        # [NOUVEAU] Déduplication sémantique
│   ├── relation_normalizer.py    # [NOUVEAU] Canonicalisation relations
│   ├── kappa_worker.py           # [NOUVEAU] Calcul courbure asynchrone
│   ├── session_storage.py        # [NOUVEAU] Export/import sessions
│   └── consciousness/
│       ├── __init__.py
│       ├── metrics.py            # Phase 1: Passive monitoring
│       ├── adaptation.py         # Phase 2: Active adaptation
│       └── memory.py             # Phase 3: Semantic memory
│
├── database/                     # Data layer
│   ├── __init__.py
│   ├── engine.py                 # ISpaceDB (571 LOC)
│   ├── schema.sql                # Database schema
│   ├── graph_delta.py            # [NOUVEAU] Suivi mutations graphe
│   ├── models.py                 # [NOUVEAU] Modèles Pydantic
│   └── pool.py                   # [NOUVEAU] Pool connexions & cache
│
├── core/                         # Pure logic
│   ├── __init__.py
│   ├── security.py               # [NOUVEAU] Gestion secrets
│   └── physics/
│       ├── __init__.py
│       └── bezier.py             # Bézier engine (471 LOC)
│
├── data/                         # Runtime data
│   ├── ispace.db                 # SQLite database
│   ├── embeddings_cache.json    # Embeddings cache
│   └── weaver.log                # Application logs
│
├── saves/                        # [NOUVEAU] Exports sessions
│   └── {nom_modele}/             # Organisé par modèle LLM
│       └── {timestamp}_{id}.json
│
├── scripts/                      # Utilities
│   ├── build_global_map.py      # Import knowledge graph
│   ├── test_api.py               # API tests
│   └── test_brain.py             # Physics tests
│
├── tests/                        # Unit tests
│   └── test_ab_metrics.py
│
├── config.yaml                   # Configuration
├── requirements.txt              # Dependencies
├── docker-compose.yml            # Docker setup
├── Dockerfile
└── README.md
```

### Statistiques du codebase

| Composant | Fichiers | LOC | Complexité |
|-----------|----------|-----|------------|
| **App** | 8 | ~1,800 | Moyenne |
| **Services** | 8 | ~1,200 | Élevée |
| **Database** | 5 | ~1,100 | Moyenne |
| **Core** | 2 | ~600 | Élevée |
| **Total** | 23 | ~4,700 | - |

---

## Composants principaux

### 1. Application Layer (app/)

#### main.py

Point d'entrée FastAPI avec lifecycle management.

**Responsabilités :**
- Initialisation database et LLM client
- Configuration CORS
- Mounting static files
- Health checks

**Hooks de lifecycle :**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await get_database()
    await get_ollama_client(...)
    yield
    # Shutdown
    await close_ollama_client()
```

**Endpoints montés :**
- `/` : Web UI
- `/api` : Root API endpoint
- `/health` : Health checks
- `/stats` : System stats
- `/chat/*` : Chat router
- `/sessions/*` : Sessions router
- `/profiles/*` : Profiles router

#### models.py

Pydantic models pour validation de requêtes/réponses.

**Modèles principaux :**
```python
# Requests
ChatRequest(message, session_id?, consciousness_level?, ...)
SessionCreateRequest(profile_name?, max_messages?, ...)

# Responses
ChatResponse(response, session_id, physics_state, ...)
SessionResponse(session_id, created_at, ...)
ProfileResponse(name, description, curves)
HealthResponse(status, database, ollama)

# Domain models
PhysicsState(t, tau_c, rho, delta_r, kappa?)
ConsciousnessMetrics(coherence, tension, fit, ...)
```

**Validators :**
```python
@field_validator('consciousness_level')
def validate_consciousness_level(cls, v):
    if not 0 <= v <= 3:
        raise ValueError("Must be 0-3")
    return v
```

#### llm_client.py

Client async pour Ollama API.

**Features :**
- Connection pooling (httpx)
- Retry logic avec backoff exponentiel
- Timeout handling
- Physics parameter mapping

**Utilisation :**
```python
client = await get_ollama_client()
response = await client.chat(
    messages=[{"role": "user", "content": "..."}],
    physics_state=state
)
```

**Mapping physique → Ollama :**
```python
temperature = map_tau_to_temperature(tau_c)  # [0.1, 1.5]
repeat_penalty = 1.0 + map_rho_to_penalties(rho)["frequency_penalty"]
```

#### embeddings.py

Wrapper pour génération d'embeddings (mxbai-embed-large, 1024D).

```python
# Single text
emb = await get_embeddings("Hello world")  # shape: (1024,)

# Batch (sequential pour l'instant)
embs = await get_embeddings_batch(["text1", "text2", ...])
```

### 2. Services Layer (services/)

#### injector.py

Injection de contexte sémantique depuis le graphe de connaissances.

**Workflow :**
```
User message
    ↓
Extract keywords (TF-IDF, stop words)
    ↓
Query semantic neighbors (SQLite, PPMI weights)
    ↓
Schedule context based on δ_r
    ↓
Inject into system prompt
```

**Classes principales :**
```python
class ContextInjector:
    async def inject_context(self, message, physics_state, db) -> GraphContext
        # Returns GraphContext(neighbors, keywords, total_weight)

class ConversationMemory:
    async def format_history(self, session_id, max_messages, max_tokens) -> List[Dict]
        # Returns conversation history with token budget
```

**Keyword extraction :**
```python
def extract_keywords(text: str, max_keywords: int) -> List[str]:
    # TF-IDF-like scoring
    # Stop words filtering (English + French)
    # Returns top N keywords
```

#### consciousness/metrics.py

Phase 1 : Calcul de métriques épistemologiques (passive, no side effects).

**Métriques :**
```python
class ConsciousnessMonitor:
    async def compute_metrics(
        self,
        context: GraphContext,
        response: str,
        physics_state: PhysicsState,
        message_index: int
    ) -> ConsciousnessMetrics
```

**Formules :**
- **Coherence** : `min(1.0, total_weight / 10.0)`
- **Tension** : `tau_c * log(1 + len(response) / 500)`
- **Fit** : `1.0 - abs(expected_len - actual_len) / max(expected_len, actual_len)`
- **Pressure** : `(tau_c + delta_r) / 2.0`
- **Stability** : Composite score basé sur coherence, tension, fit

#### consciousness/adaptation.py

Phase 2 : Adaptation active (suggère ajustements).

**Règles d'adaptation :**
```python
def suggest_adaptation(self, metrics, state, message_index):
    if metrics.tension > 0.75:
        # Reduce tau_c by 5%
        return Suggestion(reason="High tension", adjustments={"tau_c": -0.05})

    if metrics.coherence < 0.3:
        # Adjust rho towards focus
        return Suggestion(...)

    if metrics.fit > 0.8 and metrics.stability_score > 0.7:
        # Encourage exploration
        return Suggestion(...)

    # ... etc
```

**Caractéristiques :**
- Ajustements graduels (5-7.5% par tour)
- Règles non-conflictuelles
- Convergence garantie (long sessions)

#### consciousness/memory.py

Phase 3 : Mémoire sémantique avec rappel par similarité.

**Architecture :**
```python
@dataclass
class MemoryEntry:
    content: str
    embedding: np.ndarray  # 1024D
    timestamp: datetime
    message_index: int

class SemanticMemory:
    _memories: Dict[str, List[MemoryEntry]]  # session_id -> entries

    async def record(self, session_id, content, message_index):
        # Generate embedding, store in dict

    async def recall(self, session_id, query_text, threshold=0.7, max_results=3):
        # Cosine similarity search
        # Temporal decay: max(0.5, 1.0 - turns_ago * 0.01)
        # Return top matches
```

**Injection dans contexte :**
```python
# Ajouté au system prompt
[MEMORY ECHO] (similarity=0.89, 12 turns ago):
{recalled_content}
```

#### Services Lyra-ACE (Nouveau)

**entity_resolver.py** - Déduplication sémantique d'entités

Résout les concepts vers leur forme canonique via similarité d'embeddings.

```python
class EntityResolver:
    async def resolve(self, concept: str, auto_create: bool = True) -> ResolutionResult

# Stratégie de résolution :
# 1. Vérifier les aliases existants (match exact)
# 2. Vérifier le concept directement (match exact)
# 3. Chercher par similarité d'embedding
# 4. Créer si nouveau (auto_create=True)

# Seuils :
SIMILARITY_THRESHOLD = 0.92  # Fusion automatique
REVIEW_THRESHOLD = 0.85      # Candidat à la revue
```

**relation_normalizer.py** - Canonicalisation des relations

Mappe les relations brutes vers 20 formes canoniques avec gestion des inverses et symétrie.

```python
class RelationNormalizer:
    async def normalize(self, relation: str) -> str
    async def get_inverse(self, relation: str) -> Optional[str]
    async def is_symmetric(self, relation: str) -> bool
    async def get_category(self, relation: str) -> str

# Catégories : causal, hierarchical, associative, property,
#             temporal, epistemic, transformational, comparative

# Exemples de mappings :
# "provoque" -> "cause"
# "est un" -> "is_a"
# "cause" <-> "caused_by" (paire inverse)
```

**kappa_worker.py** - Calcul de courbure asynchrone

Worker en arrière-plan pour calcul batch de courbure Ollivier.

```python
class KappaWorker:
    def __init__(self, db: ISpaceDB, alpha: float = 0.5):
        self.calculator = KappaCalculator(alpha=alpha)

    async def process_batch(self, limit: int = 100) -> int:
        # Traite les arêtes en attente, retourne le nombre traité

    async def run_continuous(self, interval: float = 5.0):
        # Exécute comme worker en arrière-plan

# Stratégie :
# - Insérer arêtes avec kappa Jaccard (rapide, O(1))
# - Calculer kappa Ollivier en arrière-plan (différé)
# - Mettre à jour avec kappa hybride quand prêt
```

**session_storage.py** - Export/Import de sessions

Persiste les sessions vers fichiers JSON organisés par modèle.

```python
class SessionStorage:
    def __init__(self, base_dir: str = "saves"):
        # Organisation : saves/{nom_modele}/{timestamp}_{session_id}.json

    async def export_session(self, db, session_id: str, model: str) -> Dict:
        # Exporte : messages, trajectoires, ajustements conscience

    async def import_session(self, db, filepath: str, new_session_id: Optional[str]) -> Dict:
        # Restaure session avec ID nouveau ou spécifié

    def list_saves(self, model: Optional[str] = None) -> List[Dict]:
        # Liste les sauvegardes, optionnellement filtrées par modèle
```

### 3. Database Layer (database/)

#### engine.py

Unified async SQLite engine.

**Classe principale :**
```python
class ISpaceDB:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._pool = None  # aiosqlite connection pool

    async def initialize(self):
        # Create tables if not exists
        # Enable WAL mode
        # Create indexes
        # Optimize PRAGMA settings

    # Concept queries
    async def get_concept(self, concept: str) -> Dict
    async def get_neighbors(self, concept: str, limit: int) -> List[Dict]
    async def search_concepts(self, keyword: str) -> List[str]

    # Session management
    async def create_session(self, profile_name: str) -> str
    async def get_session(self, session_id: str) -> Dict
    async def append_event(self, session_id, role, content, ...)

    # Profile management
    async def get_profile(self, name: str) -> Dict
    async def list_profiles(self) -> List[Dict]

    # Utilities
    async def get_stats(self) -> Dict
    async def vacuum(self)  # VACUUM + ANALYZE
```

**Optimisations :**
```sql
-- WAL mode for concurrent reads
PRAGMA journal_mode=WAL;

-- Large cache
PRAGMA cache_size=-64000;  -- 64MB

-- Memory-mapped I/O
PRAGMA mmap_size=268435456;  -- 256MB
```

**Indexes :**
```sql
-- O(log N) lookups
CREATE INDEX idx_relations_source ON semantic_relations(source);
CREATE INDEX idx_events_session ON events(session_id, timestamp);
CREATE INDEX idx_sessions_created ON sessions(created_at);
```

#### schema.sql

Schéma de base de données (13KB).

**Tables principales :**

```sql
-- Knowledge graph
concepts (
    concept TEXT PRIMARY KEY,
    embedding BLOB  -- 1024D float32 array (optionnel)
)

semantic_relations (
    source TEXT,
    target TEXT,
    weight REAL,  -- PPMI score
    PRIMARY KEY (source, target)
)

-- Sessions
sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    profile_name TEXT,
    max_messages INTEGER,
    time_mapping TEXT
)

events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp TEXT,
    role TEXT,  -- user, assistant, system
    content TEXT,
    message_index INTEGER,
    physics_state TEXT  -- JSON
)

-- Bézier profiles
bezier_profiles (
    name TEXT PRIMARY KEY,
    description TEXT,
    tau_c_json TEXT,  -- 4 control points
    rho_json TEXT,
    delta_r_json TEXT,
    kappa_json TEXT  -- optionnel
)

-- Trajectory logging (for analysis)
trajectory_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    message_index INTEGER,
    t REAL,
    tau_c REAL,
    rho REAL,
    delta_r REAL,
    kappa REAL
)
```

#### graph_delta.py (Nouveau)

Gestion des mutations atomiques du graphe avec piste d'audit.

```python
class DeltaOperation(Enum):
    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    UPDATE_NODE = "update_node"
    UPDATE_EDGE = "update_edge"
    DELETE_NODE = "delete_node"
    DELETE_EDGE = "delete_edge"

@dataclass
class GraphDelta:
    operation: DeltaOperation
    source: str
    target: Optional[str] = None
    weight: Optional[float] = None
    confidence: float = 1.0
    model_source: str = "system"
    reason: Optional[str] = None

@dataclass
class DeltaBatch:
    deltas: List[GraphDelta]
    max_mutation_ratio: float = 0.05  # Limite 5% par batch

class KappaCalculator:
    """Calculateur courbure hybride (Ollivier + Jaccard)"""

    def ollivier_approx(self, degree_u, degree_v, weight) -> float:
        # kappa = 1/deg(u) + 1/deg(v) - 2/w

    def jaccard_kappa(self, neighbors_u, neighbors_v) -> float:
        # kappa = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    def compute_hybrid(self, ...) -> Dict[str, float]:
        # Retourne : kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha
```

#### pool.py (Nouveau)

Pool de connexions et utilitaires de performance.

```python
class SQLiteConnectionPool:
    """Pool de connexions async avec gestion overflow"""

    def __init__(self, db_path: str, pool_size: int = 10, max_overflow: int = 5):
        self._pragmas = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",
            "PRAGMA mmap_size=268435456",
        ]

    async def acquire(self):
        # Context manager pour acquisition connexion

class ConceptCache:
    """Cache LRU avec TTL pour concepts (1000 entrées, 1h TTL)"""

class ConcurrencyLimiter:
    """Contrôle concurrence via Semaphore"""

class SQLValidator:
    """Prévention injection SQL"""

    @classmethod
    def validate_identifier(cls, value: str) -> bool
    @classmethod
    def sanitize_string(cls, value: str) -> str
    @classmethod
    def validate_concept_id(cls, concept_id: str) -> str
```

### 4. Core Layer (core/)

#### security.py (Nouveau)

Gestion sécurisée des secrets et clés API.

```python
from core.security import get_api_key, validate_environment, init_security

# Au démarrage
validate_environment()

# Récupérer clés API
ollama_url = get_api_key("OLLAMA_URL", default="http://localhost:11434")
mistral_key = get_api_key("MISTRAL_API_KEY")

# Variables d'environnement attendues :
# OLLAMA_URL       - URL serveur Ollama
# OLLAMA_MODEL     - Modèle par défaut
# MISTRAL_API_KEY  - API Mistral (optionnel)
# OPENAI_API_KEY   - API OpenAI (optionnel)
# LYRA_SECRET_KEY  - Clé secrète application
# LYRA_ENV         - Environnement (development/production)
# LYRA_DEBUG       - Mode debug

# Utilitaires :
mask_secret("sk-abc123def456")  # Retourne "************f456"
generate_session_token()         # Retourne token hex 64 caractères
is_production()                  # Vérifie si en production
```

### 5. Moteur Physique (core/physics/)

#### bezier.py

Moteur de trajectoires Bézier (pure math, no side effects).

**Classes principales :**

```python
@dataclass(frozen=True)
class BezierPoint:
    t: float      # Parameter [0, 1]
    value: float  # Value [0, 1]

class CubicBezier:
    """Cubic Bézier curve interpolator."""

    def __init__(self, points: List[BezierPoint]):
        # Must have exactly 4 control points
        # Validate monotonicity and endpoints

    def evaluate(self, t: float) -> float:
        """Evaluate curve at parameter t ∈ [0, 1]."""
        # De Casteljau's algorithm

    def derivative(self, t: float) -> float:
        """Rate of change at t."""

    @classmethod
    def from_json(cls, json_str: str) -> 'CubicBezier':
        """Deserialize from JSON array."""

@dataclass(frozen=True)
class PhysicsState:
    t: float        # Normalized time [0, 1]
    tau_c: float    # Tension/temperature
    rho: float      # Focus/polarity
    delta_r: float  # Scheduling
    kappa: Optional[float] = None  # Curvature/style

class BezierEngine:
    """Main trajectory engine."""

    def __init__(
        self,
        tau_c_curve: CubicBezier,
        rho_curve: CubicBezier,
        delta_r_curve: CubicBezier,
        kappa_curve: Optional[CubicBezier] = None
    ):
        self.curves = {...}

    def compute_state(self, t: float) -> PhysicsState:
        """Compute physics state at normalized time t."""
        return PhysicsState(
            t=t,
            tau_c=self.curves['tau_c'].evaluate(t),
            rho=self.curves['rho'].evaluate(t),
            delta_r=self.curves['delta_r'].evaluate(t),
            kappa=self.curves.get('kappa')?.evaluate(t)
        )

    def sample_trajectory(self, num_points: int) -> List[PhysicsState]:
        """Sample trajectory for visualization."""

class TimeMapper:
    """Map message count to normalized time t ∈ [0, 1]."""

    @staticmethod
    def linear(n: int, max_n: int) -> float:
        return n / max_n

    @staticmethod
    def logarithmic(n: int, max_n: int) -> float:
        # Slower early progress
        return math.log(1 + n) / math.log(1 + max_n)

    @staticmethod
    def sigmoid(n: int, max_n: int) -> float:
        # Smooth S-curve
        x = (n / max_n - 0.5) * 10
        return 1.0 / (1.0 + math.exp(-x))
```

**Parameter mappers :**
```python
def map_tau_to_temperature(tau_c: float) -> float:
    """Map tau_c ∈ [0, 1] to temperature ∈ [0.1, 1.5]."""
    return 0.1 + tau_c * 1.4

def map_rho_to_penalties(rho: float) -> Dict[str, float]:
    """Map rho to presence/frequency penalties."""
    # rho=0.5 → neutral
    # rho<0.5 → more repetition allowed
    # rho>0.5 → penalize repetition
    return {
        "presence_penalty": (rho - 0.5) * 0.4,
        "frequency_penalty": (rho - 0.5) * 0.6
    }

def map_kappa_to_style_hints(kappa: float) -> str:
    """Generate style hints for prompt."""
    if kappa < 0.3:
        return "Be concise and direct."
    elif kappa > 0.7:
        return "Elaborate deeply with examples."
    else:
        return ""
```

---

## Flux de données

### Chat Request Flow

```
1. POST /chat/message
   ↓
2. Pydantic validation (ChatRequest)
   ↓
3. Session lookup or create
   ↓
4. Compute physics state (Bézier curves + time mapping)
   ↓
5. Retrieve conversation history (sliding window)
   ↓
6. Extract keywords from user message
   ↓
7. Query semantic neighbors (database)
   ↓
8. [If level ≥ 3] Recall similar past messages
   ↓
9. Assemble enriched prompt:
   - System prompt + semantic context + memory echoes
   - Conversation history
   - User message
   ↓
10. LLM API call (Ollama) with physics parameters
   ↓
11. Log event to database (user + assistant messages)
   ↓
12. [If level ≥ 1] Compute consciousness metrics
   ↓
13. [If level = 2] Generate adaptation suggestion
   ↓
14. Log trajectory point
   ↓
15. Return ChatResponse
```

### Database Query Flow

**Semantic neighbor query :**
```sql
-- O(k log N) where k = number of keywords
WITH neighbors AS (
    SELECT target AS concept, weight
    FROM semantic_relations
    WHERE source IN (keyword1, keyword2, ...)
      AND weight > min_weight
    ORDER BY weight DESC
    LIMIT max_neighbors
)
SELECT * FROM neighbors
```

**Conversation history retrieval :**
```sql
-- O(log N) via idx_events_session
SELECT role, content, timestamp, message_index, physics_state
FROM events
WHERE session_id = ?
ORDER BY message_index DESC
LIMIT ?
```

### Memory Recall Flow

```
1. User message arrives
   ↓
2. Generate embedding (mxbai-embed-large)
   ↓
3. Compute cosine similarity with all past messages in session
   ↓
4. Apply temporal decay: max(0.5, 1.0 - turns_ago * 0.01)
   ↓
5. Filter by threshold (0.7)
   ↓
6. Return top 3 matches
   ↓
7. Inject as [MEMORY ECHO] in system prompt
```

---

## Moteur physique Bézier

### Pourquoi des courbes de Bézier ?

**Avantages :**
1. **Contrôle intuitif** : 4 points définissent toute la trajectoire
2. **Interpolation lisse** : Dérivées continues (C1)
3. **Contraintes naturelles** : t ∈ [0, 1], valeurs bornées
4. **Visualisable** : Facile à prévisualiser et débugger
5. **Efficace** : Évaluation O(1) via de Casteljau

**Alternatives considérées :**
- Polynômes : Instables (Runge phenomenon)
- Splines : Trop complexe pour 4 points
- Linéaire par morceaux : Pas assez lisse

### Anatomie d'une courbe

**4 points de contrôle :**
```
P0 (t=0)    : Point de départ (valeur initiale)
P1 (t≈0.33) : Contrôle pente début
P2 (t≈0.67) : Contrôle pente fin
P3 (t=1)    : Point d'arrivée (valeur finale)
```

**Exemple : tau_c curve pour "balanced" profile**
```json
[
  {"t": 0.0,  "value": 0.50},  // P0: Start at 0.5 (neutral temp)
  {"t": 0.33, "value": 0.45},  // P1: Dip slightly
  {"t": 0.67, "value": 0.55},  // P2: Rise slightly
  {"t": 1.0,  "value": 0.50}   // P3: Return to 0.5
]
```

**Visualisation :**
```
value
1.0 │
    │
0.5 │  •─────•─────•  (gentle wave)
    │
0.0 │
    └──────────────────> t
      0    0.5    1.0
```

### Créer un profil personnalisé

**Étape 1 : Définir les objectifs**
- Début de conversation : Créatif ou précis ?
- Milieu : Maintenir ou ajuster ?
- Fin : Converger ou explorer ?

**Étape 2 : Choisir les valeurs**
```python
# Exemple : Profile "creative_to_precise"
tau_c = [
    {"t": 0.0,  "value": 0.9},   # Start high (creative)
    {"t": 0.33, "value": 0.85},  # Stay high early
    {"t": 0.67, "value": 0.5},   # Drop mid-conversation
    {"t": 1.0,  "value": 0.3}    # End low (precise)
]
```

**Étape 3 : Valider**
```python
from core.physics.bezier import CubicBezier, BezierPoint

points = [BezierPoint(**p) for p in tau_c]
curve = CubicBezier(points)

# Preview
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"t={t:.2f} → tau_c={curve.evaluate(t):.3f}")
```

**Étape 4 : Insérer en base**
```sql
INSERT INTO bezier_profiles (name, description, tau_c_json, rho_json, delta_r_json)
VALUES (
    'creative_to_precise',
    'Start creative, end precise',
    '[{"t":0.0,"value":0.9}, ...]',
    '...',  -- définir rho
    '...'   -- définir delta_r
);
```

### Time Mapping

Le mapping temps convertit le nombre de messages en `t ∈ [0, 1]`.

**Linear :**
```python
t = n / max_messages
# t=0.5 atteint à mi-conversation
```

**Logarithmic (défaut) :**
```python
t = log(1 + n) / log(1 + max_messages)
# Progrès plus lent en début (plus de temps pour ajustement)
```

**Sigmoid :**
```python
x = (n / max_messages - 0.5) * 10
t = 1 / (1 + exp(-x))
# Courbe en S, progression douce début/fin
```

**Comparaison (max_messages=100) :**

| Messages | Linear | Logarithmic | Sigmoid |
|----------|--------|-------------|---------|
| 10 | 0.10 | 0.23 | 0.02 |
| 25 | 0.25 | 0.42 | 0.12 |
| 50 | 0.50 | 0.62 | 0.50 |
| 75 | 0.75 | 0.78 | 0.88 |
| 100 | 1.00 | 1.00 | 1.00 |

**Recommandation :**
- **Linear** : Conversations courtes, progression régulière
- **Logarithmic** : Conversations longues, adaptation précoce
- **Sigmoid** : Transitions douces, éviter changements brusques

---

## Système de conscience

### Architecture à 3 phases

```
Phase 1: Passive Monitoring
    │ Calcule métriques, aucune action
    ↓
Phase 2: Active Adaptation
    │ Hérite Phase 1 + suggère ajustements
    ↓
Phase 3: Semantic Memory
    │ Hérite Phase 2 + rappel par similarité
```

### Implémentation

**Phase 1 : ConsciousnessMonitor**

```python
# services/consciousness/metrics.py

class ConsciousnessMonitor:
    async def compute_metrics(
        self,
        context: GraphContext,
        response: str,
        physics_state: PhysicsState,
        message_index: int
    ) -> ConsciousnessMetrics:

        # 1. Coherence (semantic density)
        coherence = min(1.0, context.total_weight / 10.0)

        # 2. Tension (system stress)
        tension = physics_state.tau_c * math.log(1 + len(response) / 500)
        tension = min(1.0, tension)

        # 3. Fit (length alignment)
        expected_length = 100 + 400 * physics_state.tau_c
        actual_length = len(response)
        fit = 1.0 - abs(expected - actual) / max(expected, actual)

        # 4. Pressure (exploration vs exploitation)
        pressure = (physics_state.tau_c + physics_state.delta_r) / 2.0

        # 5. Stability (composite)
        stability = (coherence + (1 - tension) + fit) / 3.0

        return ConsciousnessMetrics(
            coherence=coherence,
            tension=tension,
            fit=fit,
            pressure=pressure,
            stability_score=stability
        )
```

**Phase 2 : AdaptiveConsciousness**

```python
# services/consciousness/adaptation.py

class AdaptiveConsciousness(ConsciousnessMonitor):
    def suggest_adaptation(
        self,
        metrics: ConsciousnessMetrics,
        state: PhysicsState,
        message_index: int
    ) -> Optional[AdaptationSuggestion]:

        # Rule 1: High tension
        if metrics.tension > 0.75:
            return AdaptationSuggestion(
                reason="High tension detected",
                adjustments={"tau_c": -0.05}  # Reduce by 5%
            )

        # Rule 2: Low coherence
        if metrics.coherence < 0.3:
            adjustment = 0.05 if state.rho < 0.5 else -0.05
            return AdaptationSuggestion(
                reason="Low coherence",
                adjustments={"rho": adjustment}
            )

        # Rule 3: High fit + stability
        if metrics.fit > 0.8 and metrics.stability_score > 0.7:
            return AdaptationSuggestion(
                reason="Stable performance, encourage exploration",
                adjustments={"tau_c": 0.03}
            )

        # Rule 4: High pressure
        if metrics.pressure > 0.85:
            return AdaptationSuggestion(
                reason="High pressure",
                adjustments={"tau_c": -0.075, "delta_r": -0.05}
            )

        # Rule 5: Long session convergence
        if message_index > 30 and 0.4 <= metrics.tension <= 0.6:
            return None  # No change, converged

        return None
```

**Phase 3 : SemanticMemory**

```python
# services/consciousness/memory.py

class SemanticMemory(AdaptiveConsciousness):
    def __init__(self):
        super().__init__()
        self._memories: Dict[str, List[MemoryEntry]] = {}

    async def record(
        self,
        session_id: str,
        content: str,
        message_index: int
    ):
        # Generate embedding
        embedding = await get_embeddings(content)

        entry = MemoryEntry(
            content=content,
            embedding=embedding,
            timestamp=datetime.utcnow(),
            message_index=message_index
        )

        # Store (limit to 50 per session)
        if session_id not in self._memories:
            self._memories[session_id] = []

        self._memories[session_id].append(entry)
        if len(self._memories[session_id]) > 50:
            self._memories[session_id].pop(0)  # Remove oldest

    async def recall(
        self,
        session_id: str,
        query_text: str,
        current_index: int,
        threshold: float = 0.7,
        max_results: int = 3
    ) -> List[MemoryEcho]:

        if session_id not in self._memories:
            return []

        # Generate query embedding
        query_emb = await get_embeddings(query_text)

        # Compute similarities
        matches = []
        for entry in self._memories[session_id]:
            # Cosine similarity
            similarity = np.dot(query_emb, entry.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(entry.embedding)
            )

            # Temporal decay
            turns_ago = current_index - entry.message_index
            decay = max(0.5, 1.0 - turns_ago * 0.01)
            adjusted_similarity = similarity * decay

            if adjusted_similarity >= threshold:
                matches.append(MemoryEcho(
                    content=entry.content,
                    similarity=similarity,
                    turns_ago=turns_ago
                ))

        # Return top N
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:max_results]
```

### Activation

```python
# app/api/chat.py

# Phase 0: Aucune conscience
if consciousness_level == 0:
    # Standard response only
    pass

# Phase 1: Observer
elif consciousness_level == 1:
    monitor = ConsciousnessMonitor()
    metrics = await monitor.compute_metrics(context, response, state, index)
    # Return metrics in response

# Phase 2: Adaptive
elif consciousness_level == 2:
    adaptive = AdaptiveConsciousness()
    metrics = await adaptive.compute_metrics(...)
    suggestion = adaptive.suggest_adaptation(metrics, state, index)
    # Return metrics + suggestion

# Phase 3: Memory
elif consciousness_level == 3:
    memory = SemanticMemory()
    # Record user + assistant messages
    await memory.record(session_id, user_message, index)
    await memory.record(session_id, assistant_response, index + 1)
    # Recall similar messages
    echoes = await memory.recall(session_id, user_message, index)
    # Inject into context
    # Return metrics + suggestion + echoes
```

---

## Base de données

### Schéma relationnel

```
concepts ─┐
          │
          ├─< semantic_relations >─ (source, target, weight)
          │
          └─> embeddings (optionnel)

sessions ─┐
          │
          ├─< events >─ (role, content, timestamp, physics_state)
          │
          └─< trajectory_log >─ (t, tau_c, rho, delta_r, kappa)

bezier_profiles ─> (tau_c_json, rho_json, delta_r_json, kappa_json)
```

### Indexes et performance

**Indexes critiques :**
```sql
-- Relations : O(log N) pour requêtes de voisins
CREATE INDEX idx_relations_source ON semantic_relations(source);
CREATE INDEX idx_relations_target ON semantic_relations(target);

-- Events : O(log N) pour historique de session
CREATE INDEX idx_events_session ON events(session_id, timestamp);
CREATE INDEX idx_events_index ON events(session_id, message_index);

-- Sessions : Cleanup queries
CREATE INDEX idx_sessions_created ON sessions(created_at);
```

**Analyse de performance :**
```sql
-- Vérifier utilisation des index
EXPLAIN QUERY PLAN
SELECT target, weight
FROM semantic_relations
WHERE source = 'quantum_physics'
ORDER BY weight DESC
LIMIT 15;

-- Output attendu :
-- SEARCH TABLE semantic_relations USING INDEX idx_relations_source (source=?)
```

### Optimisations PRAGMA

```python
# database/engine.py

async def initialize(self):
    async with aiosqlite.connect(self._db_path) as db:
        # WAL mode : concurrent reads
        await db.execute("PRAGMA journal_mode=WAL")

        # Cache size : 64MB
        await db.execute("PRAGMA cache_size=-64000")

        # Memory-mapped I/O : 256MB
        await db.execute("PRAGMA mmap_size=268435456")

        # Synchronous : NORMAL (balance safety/speed)
        await db.execute("PRAGMA synchronous=NORMAL")

        # Temp store : memory
        await db.execute("PRAGMA temp_store=MEMORY")
```

### Maintenance

**Vacuum (défragmentation) :**
```python
async def vacuum(self):
    """Defragment and optimize database."""
    async with aiosqlite.connect(self._db_path) as db:
        await db.execute("VACUUM")
        await db.execute("ANALYZE")
```

**Exécution automatique (config.yaml) :**
```yaml
database:
  vacuum_interval_days: 7
```

**Backups automatiques :**
```python
async def backup(self, backup_path: str):
    """Backup database to file."""
    async with aiosqlite.connect(self._db_path) as db:
        async with aiosqlite.connect(backup_path) as backup_db:
            await db.backup(backup_db)
```

---

## Tests et benchmarks

### Structure

```
tests/
├── test_ab_metrics.py        # Unit tests
└── benchmarks/
    ├── benchmark_phase_1.py  # Consciousness metrics
    ├── benchmark_phase_2.py  # Adaptation
    └── benchmark_phase_3.py  # Memory
```

### Exécuter les tests

```bash
# Unit tests
pytest tests/test_ab_metrics.py -v

# Benchmarks Phase 1
python tests/benchmarks/benchmark_phase_1.py

# Benchmarks complets
python tests/benchmarks/benchmark_suite.py
```

### Métriques de benchmark

**Latence (Phase 0-3) :**
- Phase 0 : ~1.2s baseline
- Phase 1 : +100ms (metrics)
- Phase 2 : +150ms (adaptation)
- Phase 3 : +250ms (embeddings)

**Throughput :**
- Concurrent requests : ~50 req/s (level 0)
- Database queries : ~1000 queries/s (indexed)

### Ajouter des tests

**Unit test example :**
```python
# tests/test_my_feature.py

import pytest
from app.models import ChatRequest

def test_chat_request_validation():
    # Valid request
    req = ChatRequest(message="Hello", consciousness_level=2)
    assert req.consciousness_level == 2

    # Invalid level
    with pytest.raises(ValueError):
        ChatRequest(message="Hello", consciousness_level=5)

@pytest.mark.asyncio
async def test_semantic_memory():
    from services.consciousness.memory import SemanticMemory

    memory = SemanticMemory()
    await memory.record("session1", "I love Python", 1)
    echoes = await memory.recall("session1", "What do I love?", 2)

    assert len(echoes) > 0
    assert "Python" in echoes[0].content
```

---

## Contribution

### Workflow

1. **Fork le dépôt**
   ```bash
   # Via GitHub UI
   ```

2. **Cloner votre fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lyra_clean_bis.git
   cd lyra_clean_bis
   ```

3. **Créer une branche**
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

4. **Développer**
   - Suivez les conventions de code
   - Ajoutez des tests
   - Documentez les changements

5. **Commiter**
   ```bash
   git add .
   git commit -m "Add: my awesome feature"
   ```

6. **Pusher**
   ```bash
   git push origin feature/my-awesome-feature
   ```

7. **Ouvrir une Pull Request**
   - Via GitHub UI
   - Remplissez le template de PR

### Conventions de code

**Python style :**
- PEP 8 compliant
- Type hints partout
- Docstrings (Google style)
- Max line length : 100 caractères

**Commit messages :**
```
<type>: <description>

[optional body]

Types: Add, Fix, Update, Refactor, Docs, Test, Chore
```

**Exemples :**
```
Add: semantic memory recall with temporal decay
Fix: retry logic bug in llm_client.py
Update: increase default consciousness level to 1
Docs: add developer guide for Bézier engine
```

### Pull Request template

```markdown
## Description
Brief description of changes

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Benchmarks run

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

### Code review process

1. **Automated checks** (CI/CD)
   - Linting (flake8)
   - Type checking (mypy)
   - Tests (pytest)

2. **Manual review**
   - Mainteneurs review code
   - Commentaires et suggestions
   - Discussion si nécessaire

3. **Approval**
   - Au moins 1 approval requis
   - Tous les checks passed

4. **Merge**
   - Squash and merge (preferred)
   - Rebase and merge (pour features complexes)

---

## Roadmap

### Phase 4 : Persistent Memory (Q2 2025)

**Objectifs :**
- Persister la mémoire sémantique en SQLite
- Limites par utilisateur (pas seulement session)
- Oubli progressif (decay exponentiel)

**Schema :**
```sql
CREATE TABLE semantic_memory (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    content TEXT,
    embedding BLOB,
    timestamp TEXT,
    access_count INTEGER,
    last_accessed TEXT
);
```

### Phase 5 : Multi-modal support (Q3 2025)

**Objectifs :**
- Support images (vision models)
- Embeddings multi-modaux (CLIP)
- Trajectoires Bézier pour paramètres visuels

### Phase 6 : Distributed deployment (Q4 2025)

**Objectifs :**
- Horizontal scaling (Redis pour mémoire)
- Load balancing
- Multi-tenant support

### Contributions welcome

Consultez les [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues) pour :
- 🐛 Bugs à corriger
- ✨ Features à implémenter
- 📚 Documentation à améliorer
- 🎨 Améliorations UI

---

## Ressources

### Documentation externe

- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Bézier curves (Wikipedia)](https://en.wikipedia.org/wiki/B%C3%A9zier_curve)
- [SQLite optimization](https://www.sqlite.org/optoverview.html)

### Papers

- **Consciousness metrics** : Epistemological approaches to AI introspection
- **Ballistic trajectories** : Deterministic parameter control vs reactive feedback

### Community

- 💬 [GitHub Discussions](https://github.com/yourusername/lyra_clean_bis/discussions)
- 📧 Mailing list : lyra-dev@example.com
- 🐦 Twitter : @lyra_clean

---

**Prochaine étape :** Consultez la [Configuration](CONFIGURATION.md) pour personnaliser Lyra.
