# Lyra ACE

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

[🇫🇷 Français](#-français) | [🇬🇧 English](#-english)

---

## 🇫🇷 Français

### Vue d'ensemble

**Lyra ACE** est un système conversationnel LLM innovant qui utilise des **trajectoires de Bézier** pour contrôler de manière déterministe les paramètres de génération (température, pénalités) et qui a pour objectif de "chercher le 0-cochain" des systèmes cognitifs augmentés à travers la construction d'un RAG partagé et pondéré par plusieurs moteurs.

**Philosophie de conception :**
- 🎯 **Trajectoires balistiques** : Comportement prévisible et ajustable
- 🧠 **Trois niveaux de conscience** : Passif → Adaptatif → Mémoire sémantique
- 📊 **Graphe de connaissances sémantiques** : Injection de contexte intelligent via SQLite
- ⚡ **Dépendances minimalistes** : ~150MB (réduction de 77% vs versions précédentes)
- 🔬 **Physique déterministe** : Contrôle mathématique précis des paramètres LLM

### Caractéristiques principales

- ✅ **API FastAPI asynchrone** avec gestion de sessions persistantes
- ✅ **Moteur physique Bézier** : 4 paramètres (τ_c, ρ, δ_r, κ) contrôlés par courbes cubiques
- ✅ **Injection de contexte sémantique** : Extraction TF-IDF + requêtes de voisinage PPMI
- ✅ **Conscience adaptative** : Ajustements automatiques basés sur métriques épistemologiques
- ✅ **Mémoire sémantique** : Rappel par similarité cosinus avec décroissance temporelle
- ✅ **Base de données SQLite optimisée** : Mode WAL, index O(log N), pooling de connexions
- ✅ **Client Ollama async** : Pooling HTTP, retry avec backoff exponentiel
- ✅ **Interface web** : UI minimaliste pour tests rapides

### Lyra-ACE (Advanced Consciousness Engine) - Phase 1

**Nouvelles fonctionnalités :**

- ✅ **Graphe dynamique avec deltas** : Mutations incrémentielles auditables du graphe sémantique
- ✅ **Calcul κ hybride** : Courbure Ollivier + Jaccard pour analyse structurelle
- ✅ **Rollback transactionnel** : Annulation des mutations avec historique complet
- ✅ **Wrapper multi-modèles** : Génération séquentielle avec plusieurs LLM Ollama
- ✅ **Métriques de consensus** : Comparaison et sélection automatique des meilleures réponses
- ✅ **Limite de mutation 5%** : Protection contre les modifications massives du graphe

**Nouveaux endpoints API :**

- `POST /graph/delta` - Appliquer une mutation au graphe
- `GET /graph/kappa/{source}/{target}` - Calculer la courbure κ hybride
- `POST /graph/rollback` - Annuler des mutations
- `GET /graph/stats` - Statistiques des mutations
- `GET /multimodel/models` - Lister les modèles Ollama disponibles
- `POST /multimodel/generate` - Génération multi-modèles avec consensus

### Démarrage rapide

#### Prérequis
- Python 3.10+
- [Ollama](https://ollama.ai/) installé et en cours d'exécution
- Un modèle LLM (gemma3, llama3.1, mistral, etc.)

#### Installation

```bash


#### Lancement

```bash
# Scripts par modèle (Windows) - choisir un:
start_gemma3.bat      # Gemma 3 (3.3GB) - Léger, rapide
start_mistral.bat     # Mistral (4.4GB) - Multilingue
start_llama3.bat      # Llama 3.1 8B (4.9GB) - Polyvalent
start_granite.bat     # Granite 3.3 (4.9GB) - Code, enterprise
start_deepseek.bat    # DeepSeek R1 (5.2GB) - Raisonnement
start_gptoss.bat      # GPT-OSS 20B (13GB) - Plus puissant

# Grand contexte (32k tokens)
start_large_context.bat

# Manuel avec variables d'environnement
set LYRA_MODEL=mistral:latest
set LYRA_NUM_CTX=16384
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Docker
docker-compose up
```

#### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `LYRA_MODEL` | gpt-oss:20b | Modèle Ollama à utiliser |
| `LYRA_OLLAMA_URL` | http://localhost:11434 | URL du serveur Ollama |
| `LYRA_NUM_CTX` | 8192 | Taille du contexte (tokens) |

#### Premier test

```bash
# Health check
curl http://localhost:8000/health

# Envoyer un message
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bonjour Lyra, explique-moi comment tu fonctionnes",
    "consciousness_level": 2
  }'
```

Ou ouvrez votre navigateur : http://localhost:8000

### Architecture

```
lyra_clean_bis/
├── app/                    # Application FastAPI
│   ├── main.py             # Point d'entrée
│   ├── models.py           # Modèles Pydantic
│   ├── llm_client.py       # Client Ollama + MultiModelClient
│   ├── embeddings.py       # Wrapper embeddings
│   └── api/
│       ├── chat.py         # Endpoint conversation
│       ├── sessions.py     # Gestion sessions
│       ├── graph.py        # [ACE] Mutations graphe
│       └── multimodel.py   # [ACE] Multi-modèles
│
├── services/               # Couche métier
│   ├── injector.py         # Injection contexte
│   └── consciousness/
│       ├── metrics.py      # Phase 1: Métriques passives
│       ├── adaptation.py   # Phase 2: Adaptation active
│       └── memory.py       # Phase 3: Mémoire sémantique
│
├── database/               # Moteur SQLite
│   ├── engine.py           # ISpaceDB + méthodes delta
│   ├── graph_delta.py      # [ACE] GraphDelta, KappaCalculator
│   └── schema.sql          # Schéma SQL (11 tables)
│
├── core/physics/           # Moteur Bézier
│   └── bezier.py           # Trajectoires physiques
│
├── config.yaml             # Configuration centralisée
└── docs/                   # Documentation complète
    ├── fr/                 # Documentation française
    └── en/                 # English documentation
```

### Niveaux de conscience

| Niveau | Nom | Description | Capacités |
|--------|-----|-------------|-----------|
| 0 | **Passif** | Aucune introspection | Réponses standard uniquement |
| 1 | **Observateur** | Monitoring sans action | Calcul métriques (cohérence, tension, fit) |
| 2 | **Adaptatif** | Auto-ajustement actif | Modifie τ_c, ρ, δ_r selon les métriques N-1 |
| 3 | **Mémoire** | Rappel sémantique | Injection [MEMORY ECHO] + adaptation niveau 2 |

### Documentation complète

- 📘 **[Guide utilisateur (FR)](docs/fr/USER_GUIDE.md)** : Utilisation de l'API
- 🔧 **[Guide développeur (FR)](docs/fr/DEVELOPER_GUIDE.md)** : Architecture et contribution
- 🎨 **[Référence API (FR)](docs/fr/API_REFERENCE.md)** : Endpoints détaillés
- ⚙️ **[Configuration (FR)](docs/fr/CONFIGURATION.md)** : Paramètres système

### Contribuer

Les contributions sont bienvenues ! Consultez le [guide de contribution](docs/fr/DEVELOPER_GUIDE.md#contribution).

### Licence

MIT License - voir [LICENSE](LICENSE)

---

## 🇬🇧 English

### Overview

**Lyra ACE** is an innovative LLM conversational system that uses **Bézier trajectories** to deterministically control generation parameters (temperature, penalties) rather than reactive feedback loops.

**Design philosophy:**
- 🎯 **Ballistic trajectories**: Predictable and tunable behavior
- 🧠 **Three consciousness levels**: Passive → Adaptive → Semantic memory
- 📊 **Semantic knowledge graph**: Intelligent context injection via SQLite
- ⚡ **Minimalist dependencies**: ~150MB (77% reduction vs previous versions)
- 🔬 **Deterministic physics**: Precise mathematical control of LLM parameters

### Key features

- ✅ **Async FastAPI** with persistent session management
- ✅ **Bézier physics engine**: 4 parameters (τ_c, ρ, δ_r, κ) controlled by cubic curves
- ✅ **Semantic context injection**: TF-IDF extraction + PPMI neighborhood queries
- ✅ **Adaptive consciousness**: Automatic adjustments based on epistemological metrics
- ✅ **Semantic memory**: Cosine similarity recall with temporal decay
- ✅ **Optimized SQLite database**: WAL mode, O(log N) indexes, connection pooling
- ✅ **Async Ollama client**: HTTP pooling, exponential backoff retry
- ✅ **Web interface**: Minimalist UI for quick testing

### Lyra-ACE (Advanced Consciousness Engine) - Phase 1

**New features:**

- ✅ **Dynamic graph with deltas**: Auditable incremental mutations of the semantic graph
- ✅ **Hybrid κ calculation**: Ollivier + Jaccard curvature for structural analysis
- ✅ **Transactional rollback**: Undo mutations with complete history
- ✅ **Multi-model wrapper**: Sequential generation with multiple Ollama LLMs
- ✅ **Consensus metrics**: Automatic comparison and selection of best responses
- ✅ **5% mutation limit**: Protection against massive graph modifications

**New API endpoints:**

- `POST /graph/delta` - Apply a mutation to the graph
- `GET /graph/kappa/{source}/{target}` - Calculate hybrid κ curvature
- `POST /graph/rollback` - Undo mutations
- `GET /graph/stats` - Mutation statistics
- `GET /multimodel/models` - List available Ollama models
- `POST /multimodel/generate` - Multi-model generation with consensus

### Quick start

#### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- An LLM model (gemma3, llama3.1, mistral, etc.)

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lyra_clean_bis.git
cd lyra_clean_bis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Launch

```bash
# Model-specific scripts (Windows) - choose one:
start_gemma3.bat      # Gemma 3 (3.3GB) - Light, fast
start_mistral.bat     # Mistral (4.4GB) - Multilingual
start_llama3.bat      # Llama 3.1 8B (4.9GB) - Versatile
start_granite.bat     # Granite 3.3 (4.9GB) - Code, enterprise
start_deepseek.bat    # DeepSeek R1 (5.2GB) - Reasoning
start_gptoss.bat      # GPT-OSS 20B (13GB) - Most powerful

# Large context (32k tokens)
start_large_context.bat

# Manual with environment variables
set LYRA_MODEL=mistral:latest
set LYRA_NUM_CTX=16384
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Docker
docker-compose up
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LYRA_MODEL` | gpt-oss:20b | Ollama model to use |
| `LYRA_OLLAMA_URL` | http://localhost:11434 | Ollama server URL |
| `LYRA_NUM_CTX` | 8192 | Context window size (tokens) |

#### First test

```bash
# Health check
curl http://localhost:8000/health

# Send message
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello Lyra, explain how you work",
    "consciousness_level": 2
  }'
```

Or open your browser: http://localhost:8000

### Architecture

See French section above for directory structure.

### Consciousness levels

| Level | Name | Description | Capabilities |
|-------|------|-------------|--------------|
| 0 | **Passive** | No introspection | Standard responses only |
| 1 | **Observer** | Monitoring without action | Compute metrics (coherence, tension, fit) |
| 2 | **Adaptive** | Active self-adjustment | Modifies τ_c, ρ, δ_r based on N-1 metrics |
| 3 | **Memory** | Semantic recall | Inject [MEMORY ECHO] + level 2 adaptation |

### Complete documentation

- 📘 **[User Guide (EN)](docs/en/USER_GUIDE.md)**: API usage
- 🔧 **[Developer Guide (EN)](docs/en/DEVELOPER_GUIDE.md)**: Architecture and contribution
- 🎨 **[API Reference (EN)](docs/en/API_REFERENCE.md)**: Detailed endpoints
- ⚙️ **[Configuration (EN)](docs/en/CONFIGURATION.md)**: System parameters

### Contributing

Contributions welcome! See [contribution guide](docs/en/DEVELOPER_GUIDE.md#contributing).

### License

CC-BY NC 4.0 License - see [LICENSE](LICENSE)

---

## Credits

**Author**: Simon ([GitHub Profile](https://github.com/SimonBouhier))

**Acknowledgments**:
- Ollama team for the inference engine
- FastAPI contributors
- Open-source community

## Support

- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/SimonBouhier/Lyra_ACE/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/SimonBouhier/Lyra_ACE/discussions)
