# Base de Donnees Lyra-ACE v2

## Vue d'ensemble

Le schema v2 comprend **18 tables** optimisees pour le protocole ESMM (Exploration Semantique Multi-Modeles).

### Ameliorations par rapport a v1
- Canonicalisation semantique (aliases)
- Calcul kappa differe (Ollivier en batch)
- Tables ESMM completes (cochaine, cycles, triplets, gaps)
- Indexes optimises pour les patterns frequents
- Triggers pour integrite automatique

---

## Tables

### 1. Graphe Semantique (Tables 1-4)

| Table | Description |
|-------|-------------|
| `concepts` | Noeuds du graphe avec embeddings 1024D |
| `concept_aliases` | Canonicalisation ("IA" -> "ia") |
| `relations` | Aretes avec poids PPMI et kappa |
| `pending_kappa_recalc` | Queue de recalcul Ollivier |

### 2. Sessions & Evenements (Tables 5-7)

| Table | Description |
|-------|-------------|
| `sessions` | Sessions de conversation |
| `events` | Journal des messages (append-only) |
| `trajectories` | Points de trajectoire Bezier |

### 3. Configuration (Tables 8-10)

| Table | Description |
|-------|-------------|
| `profiles` | Profils Bezier (balanced, creative, safe, analytical) |
| `session_adjustments` | Conscience adaptative Phase 2 |
| `semantic_memory` | Memoire semantique Phase 3 |

### 4. Audit & Historique (Tables 11-12)

| Table | Description |
|-------|-------------|
| `graph_deltas` | Historique des mutations (rollback possible) |
| `kappa_history` | Historique des calculs de courbure |

### 5. ESMM Protocol (Tables 13-18)

| Table | Description |
|-------|-------------|
| `esmm_runs` | Executions du protocole |
| `exploration_cycles` | Cycles d'exploration (Divergent/Debat/Meta) |
| `triplet_extractions` | Triplets extraits des reponses |
| `cochain_entries` | 0-Cochaine de consensus |
| `knowledge_gaps` | Lacunes identifiees |
| `canonical_relations` | 20 types de relations normalisees |

---

## Canonicalisation

### Concepts

Les concepts sont normalises vers une forme canonique:

```
"Intelligence Artificielle" -> "ia"
"IA" -> "ia"
"AI" -> "ia"
```

La table `concept_aliases` stocke ces mappings avec le score de similarite.

### Relations

Les 20 relations canoniques sont groupees par categorie:

| Categorie | Relations |
|-----------|-----------|
| Causal | `cause`, `caused_by` |
| Hierarchical | `is_a`, `part_of`, `has_part` |
| Associative | `related_to`, `similar_to`, `opposite_of`, `different_from` |
| Property | `has_property`, `used_for` |
| Temporal | `precedes`, `follows`, `cooccurs_with` |
| Epistemic | `implies`, `contradicts`, `supports`, `requires` |
| Transformational | `transforms_into`, `produces` |
| Comparative | `greater_than`, `less_than`, `equivalent_to` |

---

## Calcul Kappa Differe

### Strategie

1. **Insertion rapide**: Utilise `kappa_jaccard` (O(1))
2. **Recalcul batch**: `kappa_ollivier` calcule en arriere-plan
3. **Kappa hybride**: `alpha * ollivier + (1-alpha) * jaccard`

### Queue de recalcul

```sql
-- Aretes en attente
SELECT * FROM pending_kappa_recalc
ORDER BY priority DESC, queued_at ASC;
```

### Trigger automatique

Quand le poids d'une relation change, elle est automatiquement ajoutee a la queue:

```sql
CREATE TRIGGER tr_relation_update_queue_kappa
AFTER UPDATE OF weight ON relations
WHEN OLD.weight != NEW.weight
BEGIN
    INSERT OR REPLACE INTO pending_kappa_recalc (source, target, priority)
    VALUES (NEW.source, NEW.target, 1);
END;
```

---

## Vues Utilitaires

| Vue | Description |
|-----|-------------|
| `v_top_concepts` | Top 1000 concepts par connectivite |
| `v_concept_with_aliases` | Concepts avec leurs aliases |
| `v_active_sessions` | Sessions des 24 dernieres heures |
| `v_esmm_run_stats` | Statistiques par run ESMM |
| `v_pending_triplets` | Triplets en attente d'injection |
| `v_active_gaps` | Lacunes non adressees |

---

## Performance

### Pragmas SQLite

```sql
PRAGMA journal_mode=WAL;      -- Concurrence lecture/ecriture
PRAGMA synchronous=NORMAL;    -- Balance securite/perf
PRAGMA cache_size=-65536;     -- 64MB cache
PRAGMA mmap_size=268435456;   -- 256MB memory-mapped I/O
```

### Index critiques

```sql
-- Recherche de voisins (operation la plus frequente)
idx_relations_source (source, weight DESC)

-- Resolution d'alias
PRIMARY KEY sur concept_aliases(alias)

-- Concepts par consensus
idx_cochain_consensus (consensus_score DESC)
```

---

## Migration

Le schema v2 est concu pour une **creation a neuf**. Pas de migration depuis v1.

```bash
# Supprimer l'ancienne base
del data\ispace.db

# Redemarrer le serveur (recree automatiquement)
python -m uvicorn app.main:app
```
