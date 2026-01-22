# Lyra Clean Developer Guide

Complete guide to understanding, modifying, and contributing to the Lyra Clean framework.

## Table of Contents

1. [Architecture](#architecture)
2. [Code Structure](#code-structure)
3. [Main Components](#main-components)
4. [Data Flow](#data-flow)
5. [Bézier Physics Engine](#bézier-physics-engine)
6. [Consciousness System](#consciousness-system)
7. [Database Layer](#database-layer)
8. [Testing and Benchmarks](#testing-and-benchmarks)
9. [Contributing](#contributing)
10. [Roadmap](#roadmap)

---

## Architecture

### Overview

Lyra Clean follows a layered architecture with clear separation of concerns:

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

### Design Principles

1. **Deterministic Physics**
   - LLM parameters controlled by mathematical trajectories
   - Predictable and reproducible behavior
   - No reactive feedback loops

2. **Separation of Concerns**
   - API layer: Validation, routing
   - Services: Business logic, context injection
   - Database: Persistence, queries
   - Core: Pure math, no side effects

3. **Async-First**
   - Everything is async/await (aiosqlite, httpx)
   - Non-blocking I/O
   - Horizontal scalability

4. **Dependency Injection**
   - FastAPI `Depends()` for DI
   - Singleton pattern for DB and LLM client
   - Testable and modular

5. **Type Safety**
   - Pydantic models for validation
   - Type hints everywhere
   - Mypy-friendly (mostly)

---

## Code Structure

```
lyra_clean_bis/
│
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                   # Entry point (369 LOC)
│   ├── models.py                 # Pydantic models (307 LOC)
│   ├── llm_client.py             # Ollama async client (308 LOC)
│   ├── embeddings.py             # Embedding wrapper (91 LOC)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py               # Chat endpoint (350 LOC)
│   │   ├── sessions.py           # Session management (332 LOC)
│   │   ├── graph.py              # [NEW] Graph mutations API (Lyra-ACE)
│   │   └── multimodel.py         # [NEW] Multi-model generation API
│   └── static/
│       └── index.html            # Web UI
│
├── services/                     # Business logic
│   ├── __init__.py
│   ├── injector.py               # Context injection (442 LOC)
│   ├── entity_resolver.py        # [NEW] Semantic entity deduplication
│   ├── relation_normalizer.py    # [NEW] Relation canonicalization
│   ├── kappa_worker.py           # [NEW] Async curvature calculation
│   ├── session_storage.py        # [NEW] Session export/import
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
│   ├── graph_delta.py            # [NEW] Graph mutation tracking
│   ├── models.py                 # [NEW] Pydantic data models
│   └── pool.py                   # [NEW] Connection pooling & caching
│
├── core/                         # Pure logic
│   ├── __init__.py
│   ├── security.py               # [NEW] Secrets management
│   └── physics/
│       ├── __init__.py
│       └── bezier.py             # Bézier engine (471 LOC)
│
├── data/                         # Runtime data
│   ├── ispace.db                 # SQLite database
│   ├── embeddings_cache.json    # Embeddings cache
│   └── weaver.log                # Application logs
│
├── saves/                        # [NEW] Session exports
│   └── {model_name}/             # Organized by LLM model
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

### Codebase Statistics

| Component | Files | LOC | Complexity |
|-----------|----------|-----|------------|
| **App** | 8 | ~1,800 | Medium |
| **Services** | 8 | ~1,200 | High |
| **Database** | 5 | ~1,100 | Medium |
| **Core** | 2 | ~600 | High |
| **Total** | 23 | ~4,700 | - |

---

## Main Components

### 1. Application Layer (app/)

#### main.py

FastAPI entry point with lifecycle management.

**Responsibilities:**
- Database and LLM client initialization
- CORS configuration
- Mounting static files
- Health checks

**Lifecycle hooks:**
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

**Mounted endpoints:**
- `/` : Web UI
- `/api` : Root API endpoint
- `/health` : Health checks
- `/stats` : System stats
- `/chat/*` : Chat router
- `/sessions/*` : Sessions router
- `/profiles/*` : Profiles router

#### models.py

Pydantic models for request/response validation.

**Main Models:**
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

**Validators:**
```python
@field_validator('consciousness_level')
def validate_consciousness_level(cls, v):
    if not 0 <= v <= 3:
        raise ValueError("Must be 0-3")
    return v
```

#### llm_client.py

Async client for Ollama API.

**Features:**
- Connection pooling (httpx)
- Retry logic with exponential backoff
- Timeout handling
- Physics parameter mapping

**Usage:**
```python
client = await get_ollama_client()
response = await client.chat(
    messages=[{"role": "user", "content": "..."}],
    physics_state=state
)
```

**Physics → Ollama mapping:**
```python
temperature = map_tau_to_temperature(tau_c)  # [0.1, 1.5]
repeat_penalty = 1.0 + map_rho_to_penalties(rho)["frequency_penalty"]
```

#### embeddings.py

Wrapper for embedding generation (mxbai-embed-large, 1024D).

```python
# Single text
emb = await get_embeddings("Hello world")  # shape: (1024,)

# Batch (sequential for now)
embs = await get_embeddings_batch(["text1", "text2", ...])
```

### 2. Services Layer (services/)

#### injector.py

Semantic context injection from the knowledge graph.

**Workflow:**
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

**Main Classes:**
```python
class ContextInjector:
    async def inject_context(self, message, physics_state, db) -> GraphContext
        # Returns GraphContext(neighbors, keywords, total_weight)

class ConversationMemory:
    async def format_history(self, session_id, max_messages, max_tokens) -> List[Dict]
        # Returns conversation history with token budget
```

**Keyword extraction:**
```python
def extract_keywords(text: str, max_keywords: int) -> List[str]:
    # TF-IDF-like scoring
    # Stop words filtering (English + French)
    # Returns top N keywords
```

#### consciousness/metrics.py

Phase 1: Computation of epistemological metrics (passive, no side effects).

**Metrics:**
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

**Formulas:**
- **Coherence**: `min(1.0, total_weight / 10.0)`
- **Tension**: `tau_c * log(1 + len(response) / 500)`
- **Fit**: `1.0 - abs(expected_len - actual_len) / max(expected_len, actual_len)`
- **Pressure**: `(tau_c + delta_r) / 2.0`
- **Stability**: Composite score based on coherence, tension, fit

#### consciousness/adaptation.py

Phase 2: Active adaptation (suggests adjustments).

**Adaptation Rules:**
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

**Characteristics:**
- Gradual adjustments (5-7.5% per turn)
- Non-conflicting rules
- Guaranteed convergence (long sessions)

#### consciousness/memory.py

Phase 3: Semantic memory with similarity-based recall.

**Architecture:**
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

**Injection into context:**
```python
# Added to system prompt
[MEMORY ECHO] (similarity=0.89, 12 turns ago):
{recalled_content}
```

#### Lyra-ACE Services (New)

**entity_resolver.py** - Semantic Entity Deduplication

Resolves concepts to their canonical form using embedding similarity.

```python
class EntityResolver:
    async def resolve(self, concept: str, auto_create: bool = True) -> ResolutionResult

# Resolution strategy:
# 1. Check existing aliases (exact match)
# 2. Check concept directly (exact match)
# 3. Search by embedding similarity
# 4. Create if new (auto_create=True)

# Thresholds:
SIMILARITY_THRESHOLD = 0.92  # Auto-fusion
REVIEW_THRESHOLD = 0.85      # Candidate for review
```

**relation_normalizer.py** - Relation Canonicalization

Maps raw relations to 20 canonical forms with inverse and symmetry tracking.

```python
class RelationNormalizer:
    async def normalize(self, relation: str) -> str
    async def get_inverse(self, relation: str) -> Optional[str]
    async def is_symmetric(self, relation: str) -> bool
    async def get_category(self, relation: str) -> str

# Categories: causal, hierarchical, associative, property,
#            temporal, epistemic, transformational, comparative

# Example mappings:
# "provoque" -> "cause"
# "est un" -> "is_a"
# "cause" <-> "caused_by" (inverse pair)
```

**kappa_worker.py** - Async Curvature Calculation

Background worker for batch Ollivier curvature calculation.

```python
class KappaWorker:
    def __init__(self, db: ISpaceDB, alpha: float = 0.5):
        self.calculator = KappaCalculator(alpha=alpha)

    async def process_batch(self, limit: int = 100) -> int:
        # Process pending edges, return count processed

    async def run_continuous(self, interval: float = 5.0):
        # Run as background worker

# Strategy:
# - Insert edges with Jaccard kappa (fast, O(1))
# - Calculate Ollivier kappa in background (deferred)
# - Update with hybrid kappa when ready
```

**session_storage.py** - Session Export/Import

Persists sessions to JSON files organized by model.

```python
class SessionStorage:
    def __init__(self, base_dir: str = "saves"):
        # Organization: saves/{model_name}/{timestamp}_{session_id}.json

    async def export_session(self, db, session_id: str, model: str) -> Dict:
        # Exports: messages, trajectories, consciousness_adjustments

    async def import_session(self, db, filepath: str, new_session_id: Optional[str]) -> Dict:
        # Restores session with new or specified ID

    def list_saves(self, model: Optional[str] = None) -> List[Dict]:
        # List available saves, optionally filtered by model
```

### 3. Database Layer (database/)

#### engine.py

Unified async SQLite engine.

**Main Class:**
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

**Optimizations:**
```sql
-- WAL mode for concurrent reads
PRAGMA journal_mode=WAL;

-- Large cache
PRAGMA cache_size=-64000;  -- 64MB

-- Memory-mapped I/O
PRAGMA mmap_size=268435456;  -- 256MB
```

**Indexes:**
```sql
-- O(log N) lookups
CREATE INDEX idx_relations_source ON semantic_relations(source);
CREATE INDEX idx_events_session ON events(session_id, timestamp);
CREATE INDEX idx_sessions_created ON sessions(created_at);
```

#### schema.sql

Database schema (13KB).

**Main Tables:**

```sql
-- Knowledge graph
concepts (
    concept TEXT PRIMARY KEY,
    embedding BLOB  -- 1024D float32 array (optional)
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
    kappa_json TEXT  -- optional
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

#### graph_delta.py (New)

Manages atomic graph mutations with audit trail.

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
    max_mutation_ratio: float = 0.05  # 5% limit per batch

class KappaCalculator:
    """Hybrid curvature calculator (Ollivier + Jaccard)"""

    def ollivier_approx(self, degree_u, degree_v, weight) -> float:
        # kappa = 1/deg(u) + 1/deg(v) - 2/w

    def jaccard_kappa(self, neighbors_u, neighbors_v) -> float:
        # kappa = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    def compute_hybrid(self, ...) -> Dict[str, float]:
        # Returns: kappa_ollivier, kappa_jaccard, kappa_hybrid, alpha
```

#### pool.py (New)

Connection pooling and performance utilities.

```python
class SQLiteConnectionPool:
    """Async connection pool with overflow management"""

    def __init__(self, db_path: str, pool_size: int = 10, max_overflow: int = 5):
        self._pragmas = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=-64000",
            "PRAGMA mmap_size=268435456",
        ]

    async def acquire(self):
        # Context manager for connection acquisition

class ConceptCache:
    """TTL LRU cache for concepts (1000 entries, 1h TTL)"""

class ConcurrencyLimiter:
    """Semaphore-based concurrency control"""

class SQLValidator:
    """SQL injection prevention"""

    @classmethod
    def validate_identifier(cls, value: str) -> bool
    @classmethod
    def sanitize_string(cls, value: str) -> str
    @classmethod
    def validate_concept_id(cls, concept_id: str) -> str
```

### 4. Core Layer (core/)

#### security.py (New)

Secure secrets and API key management.

```python
from core.security import get_api_key, validate_environment, init_security

# At startup
validate_environment()

# Get API keys
ollama_url = get_api_key("OLLAMA_URL", default="http://localhost:11434")
mistral_key = get_api_key("MISTRAL_API_KEY")

# Expected environment variables:
# OLLAMA_URL       - Ollama server URL
# OLLAMA_MODEL     - Default model
# MISTRAL_API_KEY  - Mistral API (optional)
# OPENAI_API_KEY   - OpenAI API (optional)
# LYRA_SECRET_KEY  - Application secret key
# LYRA_ENV         - Environment (development/production)
# LYRA_DEBUG       - Debug mode flag

# Utilities:
mask_secret("sk-abc123def456")  # Returns "************f456"
generate_session_token()         # Returns 64-char hex token
is_production()                  # Check if in production mode
```

### 5. Physics Engine (core/physics/)

#### bezier.py

Bézier trajectory engine (pure math, no side effects).

**Main Classes:**

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

**Parameter mappers:**
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

## Data Flow

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

**Semantic neighbor query:**
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

**Conversation history retrieval:**
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

## Bézier Physics Engine

### Why Bézier Curves?

**Advantages:**
1. **Intuitive Control**: 4 points define the entire trajectory
2. **Smooth Interpolation**: Continuous derivatives (C1)
3. **Natural Constraints**: t ∈ [0, 1], bounded values
4. **Visualizable**: Easy to preview and debug
5. **Efficient**: O(1) evaluation via de Casteljau

**Alternatives Considered:**
- Polynomials: Unstable (Runge phenomenon)
- Splines: Too complex for 4 points
- Piecewise linear: Not smooth enough

### Anatomy of a Curve

**4 control points:**
```
P0 (t=0)    : Starting point (initial value)
P1 (t≈0.33) : Controls early slope
P2 (t≈0.67) : Controls late slope
P3 (t=1)    : Ending point (final value)
```

**Example: tau_c curve for "balanced" profile**
```json
[
  {"t": 0.0,  "value": 0.50},  // P0: Start at 0.5 (neutral temp)
  {"t": 0.33, "value": 0.45},  // P1: Dip slightly
  {"t": 0.67, "value": 0.55},  // P2: Rise slightly
  {"t": 1.0,  "value": 0.50}   // P3: Return to 0.5
]
```

**Visualization:**
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

### Create a Custom Profile

**Step 1: Define objectives**
- Conversation start: Creative or precise?
- Middle: Maintain or adjust?
- End: Converge or explore?

**Step 2: Choose values**
```python
# Example: Profile "creative_to_precise"
tau_c = [
    {"t": 0.0,  "value": 0.9},   # Start high (creative)
    {"t": 0.33, "value": 0.85},  # Stay high early
    {"t": 0.67, "value": 0.5},   # Drop mid-conversation
    {"t": 1.0,  "value": 0.3}    # End low (precise)
]
```

**Step 3: Validate**
```python
from core.physics.bezier import CubicBezier, BezierPoint

points = [BezierPoint(**p) for p in tau_c]
curve = CubicBezier(points)

# Preview
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"t={t:.2f} → tau_c={curve.evaluate(t):.3f}")
```

**Step 4: Insert into database**
```sql
INSERT INTO bezier_profiles (name, description, tau_c_json, rho_json, delta_r_json)
VALUES (
    'creative_to_precise',
    'Start creative, end precise',
    '[{"t":0.0,"value":0.9}, ...]',
    '...',  -- define rho
    '...'   -- define delta_r
);
```

### Time Mapping

Time mapping converts message count to `t ∈ [0, 1]`.

**Linear:**
```python
t = n / max_messages
# t=0.5 reached at mid-conversation
```

**Logarithmic (default):**
```python
t = log(1 + n) / log(1 + max_messages)
# Slower progress early (more time for adjustment)
```

**Sigmoid:**
```python
x = (n / max_messages - 0.5) * 10
t = 1 / (1 + exp(-x))
# S-curve, smooth progression at start/end
```

**Comparison (max_messages=100):**

| Messages | Linear | Logarithmic | Sigmoid |
|----------|--------|-------------|---------|
| 10 | 0.10 | 0.23 | 0.02 |
| 25 | 0.25 | 0.42 | 0.12 |
| 50 | 0.50 | 0.62 | 0.50 |
| 75 | 0.75 | 0.78 | 0.88 |
| 100 | 1.00 | 1.00 | 1.00 |

**Recommendation:**
- **Linear**: Short conversations, regular progression
- **Logarithmic**: Long conversations, early adaptation
- **Sigmoid**: Smooth transitions, avoid abrupt changes

---

## Consciousness System

### 3-Phase Architecture

```
Phase 1: Passive Monitoring
    │ Compute metrics, no action
    ↓
Phase 2: Active Adaptation
    │ Inherits Phase 1 + suggests adjustments
    ↓
Phase 3: Semantic Memory
    │ Inherits Phase 2 + similarity-based recall
```

### Implementation

**Phase 1: ConsciousnessMonitor**

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

**Phase 2: AdaptiveConsciousness**

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

**Phase 3: SemanticMemory**

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

# Phase 0: No consciousness
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

## Database Layer

### Relational Schema

```
concepts ─┐
          │
          ├─< semantic_relations >─ (source, target, weight)
          │
          └─> embeddings (optional)

sessions ─┐
          │
          ├─< events >─ (role, content, timestamp, physics_state)
          │
          └─< trajectory_log >─ (t, tau_c, rho, delta_r, kappa)

bezier_profiles ─> (tau_c_json, rho_json, delta_r_json, kappa_json)
```

### Indexes and Performance

**Critical Indexes:**
```sql
-- Relations: O(log N) for neighbor queries
CREATE INDEX idx_relations_source ON semantic_relations(source);
CREATE INDEX idx_relations_target ON semantic_relations(target);

-- Events: O(log N) for session history
CREATE INDEX idx_events_session ON events(session_id, timestamp);
CREATE INDEX idx_events_index ON events(session_id, message_index);

-- Sessions: Cleanup queries
CREATE INDEX idx_sessions_created ON sessions(created_at);
```

**Performance Analysis:**
```sql
-- Check index usage
EXPLAIN QUERY PLAN
SELECT target, weight
FROM semantic_relations
WHERE source = 'quantum_physics'
ORDER BY weight DESC
LIMIT 15;

-- Expected output:
-- SEARCH TABLE semantic_relations USING INDEX idx_relations_source (source=?)
```

### PRAGMA Optimizations

```python
# database/engine.py

async def initialize(self):
    async with aiosqlite.connect(self._db_path) as db:
        # WAL mode: concurrent reads
        await db.execute("PRAGMA journal_mode=WAL")

        # Cache size: 64MB
        await db.execute("PRAGMA cache_size=-64000")

        # Memory-mapped I/O: 256MB
        await db.execute("PRAGMA mmap_size=268435456")

        # Synchronous: NORMAL (balance safety/speed)
        await db.execute("PRAGMA synchronous=NORMAL")

        # Temp store: memory
        await db.execute("PRAGMA temp_store=MEMORY")
```

### Maintenance

**Vacuum (defragmentation):**
```python
async def vacuum(self):
    """Defragment and optimize database."""
    async with aiosqlite.connect(self._db_path) as db:
        await db.execute("VACUUM")
        await db.execute("ANALYZE")
```

**Automatic execution (config.yaml):**
```yaml
database:
  vacuum_interval_days: 7
```

**Automatic backups:**
```python
async def backup(self, backup_path: str):
    """Backup database to file."""
    async with aiosqlite.connect(self._db_path) as db:
        async with aiosqlite.connect(backup_path) as backup_db:
            await db.backup(backup_db)
```

---

## Testing and Benchmarks

### Structure

```
tests/
├── test_ab_metrics.py        # Unit tests
└── benchmarks/
    ├── benchmark_phase_1.py  # Consciousness metrics
    ├── benchmark_phase_2.py  # Adaptation
    └── benchmark_phase_3.py  # Memory
```

### Run Tests

```bash
# Unit tests
pytest tests/test_ab_metrics.py -v

# Phase 1 benchmarks
python tests/benchmarks/benchmark_phase_1.py

# Complete benchmarks
python tests/benchmarks/benchmark_suite.py
```

### Benchmark Metrics

**Latency (Phase 0-3):**
- Phase 0: ~1.2s baseline
- Phase 1: +100ms (metrics)
- Phase 2: +150ms (adaptation)
- Phase 3: +250ms (embeddings)

**Throughput:**
- Concurrent requests: ~50 req/s (level 0)
- Database queries: ~1000 queries/s (indexed)

### Add Tests

**Unit test example:**
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

## Contributing

### Workflow

1. **Fork the repository**
   ```bash
   # Via GitHub UI
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lyra_clean_bis.git
   cd lyra_clean_bis
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

4. **Develop**
   - Follow code conventions
   - Add tests
   - Document changes

5. **Commit**
   ```bash
   git add .
   git commit -m "Add: my awesome feature"
   ```

6. **Push**
   ```bash
   git push origin feature/my-awesome-feature
   ```

7. **Open a Pull Request**
   - Via GitHub UI
   - Fill in the PR template

### Code Conventions

**Python style:**
- PEP 8 compliant
- Type hints everywhere
- Docstrings (Google style)
- Max line length: 100 characters

**Commit messages:**
```
<type>: <description>

[optional body]

Types: Add, Fix, Update, Refactor, Docs, Test, Chore
```

**Examples:**
```
Add: semantic memory recall with temporal decay
Fix: retry logic bug in llm_client.py
Update: increase default consciousness level to 1
Docs: add developer guide for Bézier engine
```

### Pull Request Template

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

### Code Review Process

1. **Automated checks** (CI/CD)
   - Linting (flake8)
   - Type checking (mypy)
   - Tests (pytest)

2. **Manual review**
   - Maintainers review code
   - Comments and suggestions
   - Discussion if needed

3. **Approval**
   - At least 1 approval required
   - All checks passed

4. **Merge**
   - Squash and merge (preferred)
   - Rebase and merge (for complex features)

---

## Roadmap

### Phase 4: Persistent Memory (Q2 2025)

**Objectives:**
- Persist semantic memory in SQLite
- Per-user limits (not just session)
- Progressive forgetting (exponential decay)

**Schema:**
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

### Phase 5: Multi-modal Support (Q3 2025)

**Objectives:**
- Image support (vision models)
- Multi-modal embeddings (CLIP)
- Bézier trajectories for visual parameters

### Phase 6: Distributed Deployment (Q4 2025)

**Objectives:**
- Horizontal scaling (Redis for memory)
- Load balancing
- Multi-tenant support

### Contributions Welcome

See [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues) for:
- 🐛 Bugs to fix
- ✨ Features to implement
- 📚 Documentation to improve
- 🎨 UI improvements

---

## Resources

### External Documentation

- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Bézier curves (Wikipedia)](https://en.wikipedia.org/wiki/B%C3%A9zier_curve)
- [SQLite optimization](https://www.sqlite.org/optoverview.html)
- [ACE framework](https://arxiv.org/pdf/2510.04618)
### Papers

- **Consciousness metrics**: Epistemological approaches to AI introspection
- **Ballistic trajectories**: Deterministic parameter control vs reactive feedback

### Community

- 💬 [GitHub Discussions](https://github.com/SimonBouhier/Lyra_ACE/discussions)
- 📧 Mailing list: simon.bouhier@proton.me
- 🐦 Twitter: @SimonOrdos

---

**Next step:** See [Configuration](CONFIGURATION.md) to customize Lyra.

