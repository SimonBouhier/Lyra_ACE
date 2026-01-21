-- ============================================================================
-- LYRA-ACE ESMM - SCHEMA V2 OPTIMISÉ
-- ============================================================================
-- 
-- Améliorations par rapport à schema.sql:
-- ✓ Table concept_aliases (canonicalisation sémantique)
-- ✓ Table pending_kappa_recalc (calcul différé Ollivier)
-- ✓ Tables ESMM complètes (cochain, cycles, triplets, gaps)
-- ✓ Indexes optimisés pour les patterns de requêtes fréquents
-- ✓ Contraintes et triggers pour intégrité
--
-- Performance:
-- - WAL mode pour concurrence lecture/écriture
-- - Cache 64MB, memory-mapped I/O 256MB
-- - Indexes composites pour requêtes multi-colonnes
--
-- Auteur: Lyra-ACE ESMM Protocol
-- Version: 2.0
-- Date: 2026-01-21
-- ============================================================================

-- ============================================================================
-- PRAGMAS DE PERFORMANCE (à exécuter à chaque connexion)
-- ============================================================================
-- Note: Ces pragmas sont aussi appliqués dans engine.py mais documentés ici
-- pour référence.

-- PRAGMA journal_mode=WAL;           -- Write-Ahead Logging (concurrence)
-- PRAGMA synchronous=NORMAL;         -- Balance sécurité/performance
-- PRAGMA cache_size=-65536;          -- 64MB cache (négatif = KB)
-- PRAGMA temp_store=MEMORY;          -- Temp tables en RAM
-- PRAGMA mmap_size=268435456;        -- 256MB memory-mapped I/O
-- PRAGMA busy_timeout=30000;         -- 30s timeout pour locks

-- ============================================================================
-- TABLE 1: CONCEPTS (Nœuds du graphe sémantique)
-- ============================================================================
CREATE TABLE IF NOT EXISTS concepts (
    -- Clé primaire
    id TEXT PRIMARY KEY NOT NULL,           -- Identifiant canonique (ex: "entropie")
    
    -- Métriques sémantiques
    rho_static REAL NOT NULL DEFAULT 0.0,   -- Densité pré-calculée ∈ [0, 1]
    degree INTEGER NOT NULL DEFAULT 0,       -- Degré du nœud (nombre de connexions)
    
    -- Embeddings (mxbai-embed-large: 1024D)
    embedding BLOB,                          -- Vecteur 1024D (float32 sérialisé)
    embedding_model TEXT DEFAULT 'mxbai-embed-large',
    embedding_updated_at REAL,               -- Timestamp dernière mise à jour
    
    -- Métadonnées ESMM
    source TEXT DEFAULT 'manual',            -- 'manual' | 'seed' | 'extracted' | 'merged'
    first_seen_model TEXT,                   -- Premier modèle ayant introduit ce concept
    
    -- Tracking
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    last_accessed REAL,
    access_count INTEGER DEFAULT 0
);

-- Index pour recherches fréquentes
CREATE INDEX IF NOT EXISTS idx_concepts_rho ON concepts(rho_static DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_degree ON concepts(degree DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_source ON concepts(source);

-- ============================================================================
-- TABLE 2: CONCEPT_ALIASES (Canonicalisation sémantique)
-- ============================================================================
-- Résout le problème: "IA" vs "Intelligence Artificielle" vs "AI"
-- Tous pointent vers le même concept canonique.

CREATE TABLE IF NOT EXISTS concept_aliases (
    -- Clé primaire: l'alias lui-même
    alias TEXT PRIMARY KEY NOT NULL,         -- Ex: "intelligence artificielle"
    
    -- Référence vers le concept canonique
    canonical_id TEXT NOT NULL,              -- Ex: "ia" (concept principal)
    
    -- Métadonnées de fusion
    similarity REAL NOT NULL,                -- Score cosinus au moment de la fusion
    fusion_method TEXT DEFAULT 'embedding',  -- 'embedding' | 'manual' | 'lemmatization'
    
    -- Tracking
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    created_by TEXT DEFAULT 'system',        -- 'system' | 'user' | nom du modèle
    
    FOREIGN KEY (canonical_id) REFERENCES concepts(id) ON DELETE CASCADE
);

-- Index pour résolution rapide
CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON concept_aliases(canonical_id);

-- ============================================================================
-- TABLE 3: RELATIONS (Arêtes du graphe sémantique)
-- ============================================================================
CREATE TABLE IF NOT EXISTS relations (
    -- Clé primaire composite
    source TEXT NOT NULL,                    -- Concept source (canonique)
    target TEXT NOT NULL,                    -- Concept cible (canonique)
    
    -- Poids et courbure
    weight REAL NOT NULL DEFAULT 0.0,        -- Poids PPMI ou confiance
    kappa REAL NOT NULL DEFAULT 0.5,         -- Courbure locale κ ∈ [0, 1]
    kappa_method TEXT DEFAULT 'jaccard',     -- 'jaccard' | 'ollivier' | 'hybrid'
    
    -- Type de relation (canonique)
    relation_type TEXT DEFAULT 'related_to', -- Relation canonique (voir liste ci-dessous)
    
    -- Provenance ESMM
    confidence REAL DEFAULT 1.0,             -- Confiance dans cette relation
    model_source TEXT DEFAULT 'system',      -- Modèle ayant extrait cette relation
    extraction_count INTEGER DEFAULT 1,      -- Nombre de fois extraite (renforcement)
    
    -- Tracking
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    updated_at REAL,
    
    PRIMARY KEY (source, target),
    FOREIGN KEY (source) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES concepts(id) ON DELETE CASCADE
);

-- Index critiques pour les requêtes de voisinage (opération la plus fréquente)
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source, weight DESC);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target, weight DESC);
CREATE INDEX IF NOT EXISTS idx_relations_weight ON relations(weight DESC);
CREATE INDEX IF NOT EXISTS idx_relations_kappa ON relations(kappa);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- ============================================================================
-- TABLE 4: PENDING_KAPPA_RECALC (Calcul différé de courbure Ollivier)
-- ============================================================================
-- Les insertions utilisent Jaccard (rapide), Ollivier est calculé en batch.

CREATE TABLE IF NOT EXISTS pending_kappa_recalc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    
    -- Priorité de recalcul
    priority INTEGER DEFAULT 0,              -- 0=normal, 1=haute (arête importante)
    
    -- Tracking
    queued_at REAL NOT NULL DEFAULT (unixepoch('now')),
    attempts INTEGER DEFAULT 0,              -- Nombre de tentatives
    last_error TEXT,                         -- Dernière erreur si échec
    
    UNIQUE(source, target),
    FOREIGN KEY (source) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES concepts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pending_kappa_priority ON pending_kappa_recalc(priority DESC, queued_at ASC);

-- ============================================================================
-- TABLE 5: SESSIONS (Sessions de conversation)
-- ============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY NOT NULL,
    
    -- Configuration
    profile TEXT DEFAULT 'balanced',
    params_snapshot TEXT,                    -- JSON des paramètres initiaux
    
    -- Statistiques
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    
    -- Tracking
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    last_activity REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity DESC);

-- ============================================================================
-- TABLE 6: EVENTS (Journal des événements - append-only)
-- ============================================================================
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    
    -- Type et contenu
    event_type TEXT NOT NULL,                -- 'user_message' | 'assistant_message' | 'system_event'
    role TEXT,                               -- 'user' | 'assistant' | 'system'
    content TEXT,
    
    -- Contexte ESMM
    injected_concepts TEXT,                  -- JSON array des concepts injectés
    graph_weight REAL DEFAULT 0.0,
    triplets_extracted INTEGER DEFAULT 0,    -- Nombre de triplets extraits de ce message
    
    -- Performance
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    latency_ms REAL,
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

-- ============================================================================
-- TABLE 7: TRAJECTORIES (Points de trajectoire Bézier)
-- ============================================================================
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_id INTEGER,
    
    -- Paramètres physiques
    t_param REAL NOT NULL,                   -- t ∈ [0, 1]
    tau_c REAL NOT NULL,
    rho REAL NOT NULL,
    delta_r REAL NOT NULL,
    kappa REAL,
    
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_trajectories_session ON trajectories(session_id, t_param);

-- ============================================================================
-- TABLE 8: PROFILES (Profils Bézier)
-- ============================================================================
CREATE TABLE IF NOT EXISTS profiles (
    profile_name TEXT PRIMARY KEY NOT NULL,
    description TEXT,
    
    -- Courbes Bézier (JSON: [[t, value], ...])
    tau_c_curve TEXT NOT NULL,
    rho_curve TEXT NOT NULL,
    delta_r_curve TEXT NOT NULL,
    kappa_curve TEXT,
    
    created_at REAL NOT NULL DEFAULT (unixepoch('now')),
    is_default INTEGER DEFAULT 0
);

-- Profils par défaut
INSERT OR IGNORE INTO profiles (profile_name, description, tau_c_curve, rho_curve, delta_r_curve, is_default) VALUES
('balanced', 'Exploration-exploitation équilibré',
 '[[0, 1.0], [0.3, 1.1], [0.7, 0.95], [1, 1.0]]',
 '[[0, 0.0], [0.3, 0.4], [0.7, 0.2], [1, 0.0]]',
 '[[0, 0.0], [0.5, 0.1], [0.8, -0.05], [1, 0.0]]',
 1),
('creative', 'Haute exploration, contraintes lâches',
 '[[0, 1.3], [0.2, 1.5], [0.6, 1.2], [1, 1.0]]',
 '[[0, 0.5], [0.3, 0.7], [0.7, 0.4], [1, 0.2]]',
 '[[0, 0.2], [0.4, 0.3], [0.8, 0.1], [1, 0.0]]',
 0),
('safe', 'Conservateur, réponses structurées',
 '[[0, 0.7], [0.3, 0.8], [0.7, 0.75], [1, 0.8]]',
 '[[0, -0.3], [0.3, -0.2], [0.7, -0.1], [1, 0.0]]',
 '[[0, -0.2], [0.5, -0.1], [0.8, 0.0], [1, 0.0]]',
 0),
('analytical', 'Haute précision, basse température',
 '[[0, 0.6], [0.2, 0.65], [0.8, 0.7], [1, 0.75]]',
 '[[0, -0.5], [0.3, -0.3], [0.7, -0.2], [1, 0.0]]',
 '[[0, -0.3], [0.5, -0.2], [0.8, -0.05], [1, 0.0]]',
 0);

-- ============================================================================
-- TABLE 9: SESSION_ADJUSTMENTS (Conscience adaptative Phase 2)
-- ============================================================================
CREATE TABLE IF NOT EXISTS session_adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    
    metrics TEXT NOT NULL,                   -- JSON: coherence, tension, fit, pressure
    adjustments TEXT NOT NULL,               -- JSON: multiplicateurs appliqués
    
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_adjustments ON session_adjustments(session_id, turn_number DESC);

-- ============================================================================
-- TABLE 10: SEMANTIC_MEMORY (Mémoire sémantique Phase 3)
-- ============================================================================
CREATE TABLE IF NOT EXISTS semantic_memory (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    
    content TEXT NOT NULL,
    embeddings BLOB NOT NULL,                -- JSON array 1024 floats
    
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_semantic_memory_session ON semantic_memory(session_id, turn_number DESC);

-- ============================================================================
-- TABLE 11: GRAPH_DELTAS (Historique des mutations - audit & rollback)
-- ============================================================================
CREATE TABLE IF NOT EXISTS graph_deltas (
    delta_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    
    -- Opération
    operation TEXT NOT NULL,                 -- 'add_node' | 'add_edge' | 'update_edge' | 'delete_*'
    source TEXT NOT NULL,
    target TEXT,
    
    -- Valeurs avant/après
    old_weight REAL,
    new_weight REAL,
    old_kappa REAL,
    new_kappa REAL,
    
    -- Métadonnées
    confidence REAL DEFAULT 1.0,
    model_source TEXT DEFAULT 'system',
    reason TEXT,
    
    -- Timestamps
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    applied_at REAL,
    rolled_back_at REAL,
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_deltas_session ON graph_deltas(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_deltas_source ON graph_deltas(source);
CREATE INDEX IF NOT EXISTS idx_deltas_rollback ON graph_deltas(rolled_back_at) WHERE rolled_back_at IS NOT NULL;

-- ============================================================================
-- TABLE 12: KAPPA_HISTORY (Historique des courbures pour analyse)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kappa_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    
    kappa_ollivier REAL NOT NULL,
    kappa_jaccard REAL NOT NULL,
    kappa_hybrid REAL NOT NULL,
    alpha REAL NOT NULL DEFAULT 0.5,
    
    timestamp REAL NOT NULL DEFAULT (unixepoch('now')),
    
    FOREIGN KEY (source) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES concepts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kappa_history_edge ON kappa_history(source, target, timestamp DESC);

-- ============================================================================
-- ============================================================================
--                         TABLES ESMM SPÉCIFIQUES
-- ============================================================================
-- ============================================================================

-- ============================================================================
-- TABLE 13: ESMM_RUNS (Exécutions du protocole complet)
-- ============================================================================
CREATE TABLE IF NOT EXISTS esmm_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Configuration
    config TEXT NOT NULL,                    -- JSON: models, cycles_per_type, etc.
    models_used TEXT NOT NULL,               -- JSON array: ["deepseek-r1", "llama3.3", ...]
    seed_type TEXT DEFAULT 'standard',       -- 'standard' | 'minimal' | 'custom'
    
    -- Progression
    status TEXT DEFAULT 'initializing',      -- 'initializing' | 'running' | 'completed' | 'failed' | 'paused'
    current_cycle TEXT,                      -- 'divergent' | 'debate' | 'meta' | NULL
    current_iteration INTEGER DEFAULT 0,
    
    -- Statistiques finales
    cycles_completed INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    total_triplets INTEGER DEFAULT 0,
    triplets_injected INTEGER DEFAULT 0,
    concepts_created INTEGER DEFAULT 0,
    relations_created INTEGER DEFAULT 0,
    final_cochain_size INTEGER,
    
    -- Métriques d'évaluation
    coverage_score REAL,                     -- Couverture sémantique
    consensus_density REAL,                  -- Densité de consensus moyenne
    epistemic_diversity REAL,                -- Entropie des types épistémiques
    structural_stability REAL,               -- κ moyen
    
    -- Timing
    started_at REAL NOT NULL DEFAULT (unixepoch('now')),
    completed_at REAL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_esmm_runs_status ON esmm_runs(status, started_at DESC);

-- ============================================================================
-- TABLE 14: EXPLORATION_CYCLES (Historique des cycles d'exploration)
-- ============================================================================
CREATE TABLE IF NOT EXISTS exploration_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    
    -- Identification
    cycle_type TEXT NOT NULL,                -- 'divergent' | 'debate' | 'meta'
    iteration INTEGER NOT NULL,
    
    -- Question posée
    question_template TEXT NOT NULL,
    question_rendered TEXT NOT NULL,         -- Question avec variables substituées
    target_concepts TEXT,                    -- JSON: concepts ciblés par cette question
    
    -- Réponses des modèles
    responses TEXT NOT NULL,                 -- JSON: {model_name: response_text}
    response_latencies TEXT,                 -- JSON: {model_name: latency_ms}
    
    -- Extraction
    triplets_extracted INTEGER DEFAULT 0,
    triplets_data TEXT,                      -- JSON: liste des triplets extraits
    
    -- Métriques du cycle
    consensus_map TEXT,                      -- JSON: {triplet_hash: consensus_score}
    exploration_metrics TEXT,                -- JSON: coverage, diversity, etc.
    
    -- Timing
    started_at REAL NOT NULL DEFAULT (unixepoch('now')),
    completed_at REAL,
    
    FOREIGN KEY (run_id) REFERENCES esmm_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_exploration_cycles_run ON exploration_cycles(run_id, cycle_type, iteration);

-- ============================================================================
-- TABLE 15: TRIPLET_EXTRACTIONS (Historique détaillé des extractions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS triplet_extractions (
    extraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Référence au cycle (optionnel, peut venir du chat aussi)
    cycle_id INTEGER,
    event_id INTEGER,                        -- Si extrait d'un message chat
    
    -- Triplet
    subject TEXT NOT NULL,                   -- Concept source (avant canonicalisation)
    subject_canonical TEXT,                  -- Concept source (après canonicalisation)
    relation TEXT NOT NULL,                  -- Type de relation brut
    relation_canonical TEXT,                 -- Type de relation normalisé
    object TEXT NOT NULL,                    -- Concept cible (avant canonicalisation)
    object_canonical TEXT,                   -- Concept cible (après canonicalisation)
    
    -- Métadonnées d'extraction
    confidence REAL NOT NULL,
    extraction_method TEXT NOT NULL,         -- 'llm_structured' | 'pattern' | 'hybrid'
    model_source TEXT NOT NULL,              -- Modèle ayant généré le texte source
    source_text TEXT,                        -- Extrait du texte source (100 chars max)
    
    -- Statut d'injection
    injected_to_graph INTEGER DEFAULT 0,     -- 0=non, 1=oui
    delta_id INTEGER,                        -- FK vers graph_deltas si injecté
    injection_skipped_reason TEXT,           -- Raison si non injecté (doublon, confiance, etc.)
    
    -- Timing
    extracted_at REAL NOT NULL DEFAULT (unixepoch('now')),
    
    FOREIGN KEY (cycle_id) REFERENCES exploration_cycles(cycle_id) ON DELETE SET NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE SET NULL,
    FOREIGN KEY (delta_id) REFERENCES graph_deltas(delta_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplet_extractions(subject_canonical);
CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplet_extractions(object_canonical);
CREATE INDEX IF NOT EXISTS idx_triplets_relation ON triplet_extractions(relation_canonical);
CREATE INDEX IF NOT EXISTS idx_triplets_injected ON triplet_extractions(injected_to_graph);
CREATE INDEX IF NOT EXISTS idx_triplets_model ON triplet_extractions(model_source);

-- ============================================================================
-- TABLE 16: COCHAIN_ENTRIES (0-Cochaîne de consensus)
-- ============================================================================
CREATE TABLE IF NOT EXISTS cochain_entries (
    concept_id TEXT PRIMARY KEY,
    
    -- Score de consensus composite [0, 1]
    consensus_score REAL NOT NULL,
    
    -- Composantes du score
    model_agreement REAL NOT NULL,           -- Accord inter-modèles
    semantic_consistency REAL NOT NULL,      -- Cohérence sémantique interne
    structural_centrality REAL NOT NULL,     -- Centralité dans le graphe
    stability_score REAL NOT NULL,           -- Stabilité temporelle
    
    -- Signature vectorielle (5D, pour visualisation)
    signature_vector TEXT NOT NULL,          -- JSON: [f1, f2, f3, f4, f5]
    
    -- Classification épistémique
    epistemic_type TEXT NOT NULL,            -- 'generalist' | 'specialized' | 'hybrid'
    
    -- Provenance
    contributing_models TEXT NOT NULL,       -- JSON: {"deepseek-r1": 0.3, "llama3.3": 0.25, ...}
    triplet_count INTEGER DEFAULT 0,         -- Nombre de triplets impliquant ce concept
    
    -- Versioning
    computed_at REAL NOT NULL DEFAULT (unixepoch('now')),
    run_id INTEGER,                          -- Dernier run ESMM ayant mis à jour
    protocol_version TEXT DEFAULT 'v2',
    
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES esmm_runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_cochain_consensus ON cochain_entries(consensus_score DESC);
CREATE INDEX IF NOT EXISTS idx_cochain_type ON cochain_entries(epistemic_type);
CREATE INDEX IF NOT EXISTS idx_cochain_run ON cochain_entries(run_id);

-- ============================================================================
-- TABLE 17: KNOWLEDGE_GAPS (Lacunes de connaissances identifiées)
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    
    -- Type de lacune
    gap_type TEXT NOT NULL,                  -- 'isolated' | 'unstable' | 'bridge'
    
    -- Détails selon le type
    -- isolated: concept_id, degree
    -- unstable: source, target, kappa
    -- bridge: cluster_a, cluster_b, similarity
    details TEXT NOT NULL,                   -- JSON avec les détails
    
    -- Priorité et statut
    priority REAL NOT NULL,                  -- Score de priorité pour exploration
    addressed INTEGER DEFAULT 0,             -- 0=non traité, 1=traité
    addressed_by_cycle_id INTEGER,           -- Cycle qui a adressé cette lacune
    
    -- Timing
    detected_at REAL NOT NULL DEFAULT (unixepoch('now')),
    addressed_at REAL,
    
    FOREIGN KEY (run_id) REFERENCES esmm_runs(run_id) ON DELETE SET NULL,
    FOREIGN KEY (addressed_by_cycle_id) REFERENCES exploration_cycles(cycle_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_gaps_type ON knowledge_gaps(gap_type, addressed);
CREATE INDEX IF NOT EXISTS idx_gaps_priority ON knowledge_gaps(priority DESC) WHERE addressed = 0;

-- ============================================================================
-- TABLE 18: CANONICAL_RELATIONS (Types de relations normalisées)
-- ============================================================================
-- Table de référence pour la canonicalisation des relations

CREATE TABLE IF NOT EXISTS canonical_relations (
    canonical TEXT PRIMARY KEY NOT NULL,     -- Ex: "cause"
    category TEXT NOT NULL,                  -- 'causal' | 'hierarchical' | 'associative' | etc.
    description TEXT,
    aliases TEXT NOT NULL,                   -- JSON: ["provoque", "engendre", "entraîne"]
    inverse TEXT,                            -- Relation inverse (ex: "caused_by" pour "cause")
    symmetric INTEGER DEFAULT 0              -- 1 si A-R-B implique B-R-A
);

-- Relations canoniques par défaut
INSERT OR IGNORE INTO canonical_relations (canonical, category, description, aliases, inverse, symmetric) VALUES
-- Causalité
('cause', 'causal', 'A provoque B', '["provoque", "engendre", "entraîne", "résulte_en", "mène_à"]', 'caused_by', 0),
('caused_by', 'causal', 'A est causé par B', '["causé_par", "dû_à", "résulte_de", "provient_de"]', 'cause', 0),

-- Hiérarchie
('is_a', 'hierarchical', 'A est un type de B', '["est_un", "est_une", "type_de", "sorte_de", "instance_de"]', 'has_instance', 0),
('part_of', 'hierarchical', 'A fait partie de B', '["fait_partie_de", "composant_de", "élément_de", "appartient_à"]', 'has_part', 0),
('has_part', 'hierarchical', 'A contient B', '["contient", "comprend", "inclut", "composé_de"]', 'part_of', 0),

-- Association
('related_to', 'associative', 'A est lié à B', '["lié_à", "associé_à", "connecté_à", "en_relation_avec"]', NULL, 1),
('similar_to', 'associative', 'A est similaire à B', '["similaire_à", "comparable_à", "analogue_à", "ressemble_à"]', NULL, 1),
('opposite_of', 'associative', 'A est opposé à B', '["opposé_à", "contraire_de", "antithèse_de", "inverse_de"]', NULL, 1),
('different_from', 'associative', 'A est différent de B', '["différent_de", "distinct_de", "ne_pas_confondre_avec"]', NULL, 1),

-- Propriété
('has_property', 'property', 'A possède la propriété B', '["possède", "caractérisé_par", "présente", "a_pour_propriété"]', 'property_of', 0),
('used_for', 'property', 'A est utilisé pour B', '["utilisé_pour", "sert_à", "permet_de", "employé_pour"]', 'uses', 0),

-- Temporalité
('precedes', 'temporal', 'A précède B', '["précède", "avant", "antérieur_à", "préalable_à"]', 'follows', 0),
('follows', 'temporal', 'A suit B', '["suit", "après", "postérieur_à", "succède_à"]', 'precedes', 0),
('cooccurs_with', 'temporal', 'A se produit avec B', '["cooccurs", "simultané_à", "en_même_temps_que"]', NULL, 1),

-- Épistémique
('implies', 'epistemic', 'A implique B', '["implique", "suggère", "indique", "signifie"]', 'implied_by', 0),
('contradicts', 'epistemic', 'A contredit B', '["contredit", "incompatible_avec", "nie", "réfute"]', NULL, 1),
('supports', 'epistemic', 'A supporte B', '["supporte", "confirme", "renforce", "corrobore"]', 'supported_by', 0),
('requires', 'epistemic', 'A nécessite B', '["nécessite", "requiert", "demande", "exige"]', 'required_by', 0),

-- Transformationnel
('transforms_into', 'transformational', 'A se transforme en B', '["devient", "se_transforme_en", "évolue_en", "mute_en"]', 'transforms_from', 0),
('produces', 'transformational', 'A produit B', '["produit", "génère", "crée", "fabrique"]', 'produced_by', 0),

-- Comparatif
('greater_than', 'comparative', 'A est supérieur à B', '["supérieur_à", "plus_grand_que", "dépasse"]', 'less_than', 0),
('less_than', 'comparative', 'A est inférieur à B', '["inférieur_à", "plus_petit_que", "en_dessous_de"]', 'greater_than', 0),
('equivalent_to', 'comparative', 'A est équivalent à B', '["équivalent_à", "égal_à", "identique_à"]', NULL, 1);

-- ============================================================================
-- VUES UTILITAIRES
-- ============================================================================

-- Top concepts par connectivité
CREATE VIEW IF NOT EXISTS v_top_concepts AS
SELECT 
    c.id, 
    c.rho_static, 
    c.degree, 
    c.access_count,
    c.source,
    COALESCE(ce.consensus_score, 0) as consensus_score,
    COALESCE(ce.epistemic_type, 'unknown') as epistemic_type
FROM concepts c
LEFT JOIN cochain_entries ce ON c.id = ce.concept_id
ORDER BY c.degree DESC, c.rho_static DESC
LIMIT 1000;

-- Aliases par concept canonique
CREATE VIEW IF NOT EXISTS v_concept_with_aliases AS
SELECT 
    c.id,
    c.degree,
    c.rho_static,
    GROUP_CONCAT(ca.alias, ', ') as aliases,
    COUNT(ca.alias) as alias_count
FROM concepts c
LEFT JOIN concept_aliases ca ON c.id = ca.canonical_id
GROUP BY c.id;

-- Sessions actives (dernières 24h)
CREATE VIEW IF NOT EXISTS v_active_sessions AS
SELECT 
    s.session_id, 
    s.last_activity, 
    s.message_count, 
    s.profile,
    COUNT(e.event_id) as event_count
FROM sessions s
LEFT JOIN events e ON s.session_id = e.session_id
WHERE s.last_activity > unixepoch('now') - 86400
GROUP BY s.session_id
ORDER BY s.last_activity DESC;

-- Statistiques ESMM par run
CREATE VIEW IF NOT EXISTS v_esmm_run_stats AS
SELECT 
    r.run_id,
    r.status,
    r.started_at,
    r.completed_at,
    (r.completed_at - r.started_at) / 60.0 as duration_minutes,
    r.total_triplets,
    r.triplets_injected,
    r.concepts_created,
    r.coverage_score,
    r.consensus_density,
    COUNT(DISTINCT ec.cycle_id) as cycles_count
FROM esmm_runs r
LEFT JOIN exploration_cycles ec ON r.run_id = ec.run_id
GROUP BY r.run_id
ORDER BY r.started_at DESC;

-- Triplets en attente d'injection
CREATE VIEW IF NOT EXISTS v_pending_triplets AS
SELECT 
    te.extraction_id,
    te.subject_canonical,
    te.relation_canonical,
    te.object_canonical,
    te.confidence,
    te.model_source,
    te.extracted_at
FROM triplet_extractions te
WHERE te.injected_to_graph = 0 
  AND te.injection_skipped_reason IS NULL
  AND te.confidence >= 0.5
ORDER BY te.confidence DESC;

-- Lacunes non adressées par priorité
CREATE VIEW IF NOT EXISTS v_active_gaps AS
SELECT 
    kg.gap_id,
    kg.gap_type,
    kg.details,
    kg.priority,
    kg.detected_at,
    er.run_id
FROM knowledge_gaps kg
LEFT JOIN esmm_runs er ON kg.run_id = er.run_id
WHERE kg.addressed = 0
ORDER BY kg.priority DESC;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Mise à jour automatique du degree quand une relation est ajoutée
CREATE TRIGGER IF NOT EXISTS tr_relation_insert_update_degree
AFTER INSERT ON relations
BEGIN
    UPDATE concepts SET degree = degree + 1 WHERE id = NEW.source;
    UPDATE concepts SET degree = degree + 1 WHERE id = NEW.target;
END;

-- Mise à jour automatique du degree quand une relation est supprimée
CREATE TRIGGER IF NOT EXISTS tr_relation_delete_update_degree
AFTER DELETE ON relations
BEGIN
    UPDATE concepts SET degree = MAX(0, degree - 1) WHERE id = OLD.source;
    UPDATE concepts SET degree = MAX(0, degree - 1) WHERE id = OLD.target;
END;

-- Queue automatique pour recalcul kappa quand relation modifiée
CREATE TRIGGER IF NOT EXISTS tr_relation_update_queue_kappa
AFTER UPDATE OF weight ON relations
WHEN OLD.weight != NEW.weight
BEGIN
    INSERT OR REPLACE INTO pending_kappa_recalc (source, target, priority, queued_at)
    VALUES (NEW.source, NEW.target, 1, unixepoch('now'));
END;

-- Mise à jour last_activity de la session quand un event est ajouté
CREATE TRIGGER IF NOT EXISTS tr_event_insert_update_session
AFTER INSERT ON events
BEGIN
    UPDATE sessions 
    SET last_activity = unixepoch('now'),
        message_count = message_count + 1
    WHERE session_id = NEW.session_id;
END;

-- ============================================================================
-- NOTES DE PERFORMANCE
-- ============================================================================
-- 
-- Patterns de requêtes fréquents et leurs index:
--
-- 1. Recherche de voisins (le plus fréquent):
--    SELECT target, weight, kappa FROM relations
--    WHERE source = ? ORDER BY weight DESC LIMIT 20;
--    → idx_relations_source (source, weight DESC)
--
-- 2. Résolution d'alias:
--    SELECT canonical_id FROM concept_aliases WHERE alias = ?;
--    → PRIMARY KEY sur alias
--
-- 3. Historique de session:
--    SELECT * FROM events WHERE session_id = ? ORDER BY timestamp;
--    → idx_events_session (session_id, timestamp DESC)
--
-- 4. Concepts par consensus:
--    SELECT * FROM cochain_entries WHERE consensus_score > 0.8;
--    → idx_cochain_consensus (consensus_score DESC)
--
-- 5. Triplets par relation:
--    SELECT * FROM triplet_extractions WHERE relation_canonical = ?;
--    → idx_triplets_relation
--
-- 6. Lacunes actives:
--    SELECT * FROM knowledge_gaps WHERE addressed = 0 ORDER BY priority DESC;
--    → idx_gaps_priority (priority DESC) WHERE addressed = 0
--
-- ============================================================================
