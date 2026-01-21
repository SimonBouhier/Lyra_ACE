# Changelog

Toutes les modifications notables de Lyra-ACE sont documentees ici.

Format base sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [2.0.0] - 2026-01-21

### Resume
Evolution majeure vers le protocole ESMM (Exploration Semantique Multi-Modeles).

### Ajoute

#### Base de donnees (schema v2)
- **8 nouvelles tables** pour ESMM:
  - `concept_aliases` - Canonicalisation semantique
  - `pending_kappa_recalc` - Queue de calcul kappa differe
  - `canonical_relations` - 20 types de relations normalisees
  - `esmm_runs` - Executions du protocole
  - `exploration_cycles` - Historique des cycles
  - `triplet_extractions` - Triplets extraits
  - `cochain_entries` - 0-Cochaine de consensus
  - `knowledge_gaps` - Lacunes identifiees

- **6 vues utilitaires**:
  - `v_top_concepts`, `v_concept_with_aliases`, `v_active_sessions`
  - `v_esmm_run_stats`, `v_pending_triplets`, `v_active_gaps`

- **4 triggers automatiques**:
  - Mise a jour du degre sur insertion/suppression de relation
  - Queue kappa sur modification de poids
  - Mise a jour session sur nouvel evenement

#### Engine (database/engine.py)
- **~30 nouvelles methodes** pour ESMM:
  - Canonicalisation: `resolve_concept`, `add_alias`, `get_concept_with_aliases`
  - Kappa differe: `queue_kappa_recalc`, `get_pending_kappa_batch`, `mark_kappa_recalc_done/failed`
  - Relations: `get_canonical_relation`, `get_all_canonical_relations`
  - ESMM Runs: `create_esmm_run`, `update_esmm_run_status`, `finalize_esmm_run`
  - ESMM Cycles: `log_exploration_cycle`, `update_cycle_extraction`
  - ESMM Triplets: `store_triplet_extraction`, `mark_triplet_injected`, `skip_triplet_injection`
  - ESMM Cochain: `upsert_cochain_entry`, `get_cochain_entry`, `get_cochain_by_type`, `export_cochain_for_viz`
  - ESMM Gaps: `add_knowledge_gap`, `get_active_gaps`, `mark_gap_addressed`

- **Methodes helper**:
  - `add_concept` - Creation de concepts
  - `get_concepts_with_embeddings` - Pour recherche de similarite
  - `get_relation` - Recuperation d'une relation
  - `update_edge_kappa` - Mise a jour kappa
  - `log_kappa_history` - Historique des calculs

#### Services
- **EntityResolver** (`services/entity_resolver.py`):
  - Resolution semantique via embeddings
  - Seuils: SIMILARITY_THRESHOLD=0.92, REVIEW_THRESHOLD=0.85
  - Fusion automatique des doublons

- **RelationNormalizer** (`services/relation_normalizer.py`):
  - Normalisation vers 20 relations canoniques
  - Gestion des inverses et symetrie
  - Cache en memoire

- **KappaWorker** (`services/kappa_worker.py`):
  - Calcul differe de kappa Ollivier
  - Mode batch et mode continu
  - Integration avec KappaCalculator

#### Documentation
- `docs/fr/DATABASE.md` - Schema v2 complet
- `docs/fr/ESMM_PROTOCOL.md` - Protocole ESMM
- `docs/fr/CHANGELOG.md` - Ce fichier

### Modifie
- `database/schema.sql` - Remplace par schema v2 (18 tables)
- `services/__init__.py` - Export des nouveaux services

### Notes de migration
- **Base de donnees**: Creation a neuf requise (pas de migration)
- **Compatibilite**: API existante preservee

---

## [1.0.0] - 2025-11-01

### Resume
Version initiale de Lyra Clean.

### Ajoute
- Moteur SQLite async avec WAL mode
- Gestion des trajectoires Bezier
- 4 profils par defaut (balanced, creative, safe, analytical)
- API FastAPI complete
- Interface web (Lyra Lite UI)
- Conscience adaptative Phase 2
- Memoire semantique Phase 3
- Graph Delta Management

---

## Liens

- [Documentation](../README.md)
- [Guide Developpeur](DEVELOPER_GUIDE.md)
- [Reference API](API_REFERENCE.md)
