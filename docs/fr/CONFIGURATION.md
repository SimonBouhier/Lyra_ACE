# Configuration Lyra Clean

Guide complet de configuration du système Lyra Clean via le fichier `config.yaml`.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Structure du fichier](#structure-du-fichier)
3. [Sections de configuration](#sections-de-configuration)
4. [Variables d'environnement](#variables-denvironnement)
5. [Configuration production](#configuration-production)
6. [Exemples de configurations](#exemples-de-configurations)

---

## Vue d'ensemble

Lyra Clean utilise un fichier `config.yaml` centralisé pour toute la configuration système.

**Emplacement :** `config.yaml` (racine du projet)

**Format :** YAML (indentation stricte, 2 espaces)

**Chargement :** Au démarrage de l'application (voir `app/main.py`)

---

## Structure du fichier

```yaml
server:          # Configuration serveur FastAPI
database:        # Configuration SQLite
llm:             # Configuration Ollama
physics:         # Moteur physique Bézier
context:         # Injection de contexte sémantique
sessions:        # Gestion des sessions
performance:     # Paramètres de performance
logging:         # Configuration logs
cors:            # Cross-Origin Resource Sharing
monitoring:      # Monitoring (futur)
security:        # Sécurité et rate limiting
```

---

## Sections de configuration

### Server

Configuration du serveur FastAPI et Uvicorn.

```yaml
server:
  host: "0.0.0.0"              # Interface d'écoute (0.0.0.0 = toutes interfaces)
  port: 8000                   # Port HTTP
  workers: 4                   # Nombre de workers Uvicorn (multi-processing)
  reload: false                # Auto-reload sur changement code (dev uniquement)
  log_level: "info"            # Niveau de log : debug, info, warning, error, critical
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `host` | string | "0.0.0.0" | Adresse IP d'écoute (0.0.0.0 = toutes) |
| `port` | integer | 8000 | Port HTTP du serveur |
| `workers` | integer | 4 | Nombre de workers (1 par CPU recommandé) |
| `reload` | boolean | false | Auto-reload (dev uniquement, désactiver en prod) |
| `log_level` | string | "info" | Verbosité des logs (debug < info < warning < error < critical) |

**Recommandations :**

- **Développement :**
  ```yaml
  host: "127.0.0.1"  # Localhost uniquement
  port: 8000
  workers: 1         # Single worker pour debugging
  reload: true       # Auto-reload pratique
  log_level: "debug" # Logs détaillés
  ```

- **Production :**
  ```yaml
  host: "0.0.0.0"    # Accessible de l'extérieur
  port: 8000
  workers: 4         # 1 par CPU (ajuster selon machine)
  reload: false      # Désactiver
  log_level: "info"  # Éviter trop de logs
  ```

---

### Database

Configuration SQLite et paramètres de maintenance.

```yaml
database:
  path: "data/ispace.db"       # Chemin vers fichier SQLite
  backup_interval_hours: 24    # Fréquence backup automatique
  vacuum_interval_days: 7      # Fréquence VACUUM + ANALYZE
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `path` | string | "data/ispace.db" | Chemin relatif ou absolu vers fichier SQLite |
| `backup_interval_hours` | integer | 24 | Backup automatique tous les N heures (0 = désactivé) |
| `vacuum_interval_days` | integer | 7 | Défragmentation tous les N jours (0 = désactivé) |

**Notes :**

- **Chemin relatif** : Relatif au répertoire de travail (CWD)
- **Backup** : Crée un fichier `.backup` dans le même répertoire
- **VACUUM** : Défragmente et optimise la base (peut prendre du temps sur grandes DB)

**Exemple production :**
```yaml
database:
  path: "/var/lib/lyra/ispace.db"  # Chemin absolu
  backup_interval_hours: 6         # Backup toutes les 6h
  vacuum_interval_days: 1          # Vacuum quotidien
```

---

### LLM (Ollama)

Configuration du client Ollama pour inférence LLM.

```yaml
llm:
  base_url: "http://localhost:11434"  # URL serveur Ollama
  model: "gpt-oss:20b"                # Modèle par défaut
  timeout: 120.0                      # Timeout requête (secondes)
  max_retries: 3                      # Tentatives sur échec
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `base_url` | string | "http://localhost:11434" | URL complète du serveur Ollama |
| `model` | string | "gpt-oss:20b" | Nom du modèle (doit être installé via `ollama pull`) |
| `timeout` | float | 120.0 | Timeout requête en secondes |
| `max_retries` | integer | 3 | Nombre de tentatives en cas d'échec réseau |

**Modèles populaires :**

| Modèle | Taille | RAM requise | Usage |
|--------|--------|-------------|-------|
| `llama3:8b` | 8B params | ~6GB | Rapide, général |
| `llama3:70b` | 70B params | ~40GB | Qualité élevée |
| `gpt-oss:20b` | 20B params | ~12GB | Équilibré |
| `mistral:latest` | 7B params | ~5GB | Rapide, efficace |

**Configuration multi-modèles :**

Pour utiliser différents modèles par endpoint (futur) :
```yaml
llm:
  base_url: "http://localhost:11434"
  models:
    default: "gpt-oss:20b"
    creative: "llama3:70b"    # Haute qualité
    fast: "mistral:latest"    # Réponses rapides
```

**Ollama distant :**
```yaml
llm:
  base_url: "http://192.168.1.100:11434"  # Serveur distant
  timeout: 180.0                           # Timeout augmenté
```

---

### Physics

Configuration du moteur physique Bézier.

```yaml
physics:
  default_profile: "balanced"     # Profil par défaut
  time_mapping: "logarithmic"     # Mapping temps : linear, logarithmic, sigmoid
  max_messages_window: 100        # Fenêtre max pour t ∈ [0, 1]
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `default_profile` | string | "balanced" | Nom du profil Bézier à utiliser par défaut |
| `time_mapping` | string | "logarithmic" | Type de mapping temps (linear, logarithmic, sigmoid) |
| `max_messages_window` | integer | 100 | Nombre max de messages pour atteindre t=1.0 |

**Time mapping :**

- **linear** : `t = n / max_messages`
  - Progression régulière
  - Bon pour conversations courtes

- **logarithmic** (défaut) : `t = log(1 + n) / log(1 + max_messages)`
  - Progrès plus lent au début
  - Bon pour conversations longues (plus de temps pour adaptation)

- **sigmoid** : `t = 1 / (1 + exp(-(n/max - 0.5)*10))`
  - Courbe en S, transitions douces
  - Bon pour éviter changements brusques

**Exemples :**

```yaml
# Conversations courtes et dynamiques
physics:
  default_profile: "aggressive"
  time_mapping: "linear"
  max_messages_window: 50

# Conversations longues avec stabilité
physics:
  default_profile: "balanced"
  time_mapping: "logarithmic"
  max_messages_window: 200

# Transitions très douces
physics:
  default_profile: "conservative"
  time_mapping: "sigmoid"
  max_messages_window: 100
```

---

### Context

Configuration de l'injection de contexte sémantique.

```yaml
context:
  enabled: true                # Activer/désactiver contexte sémantique
  max_keywords: 5              # Max mots-clés extraits par message
  max_neighbors: 15            # Max voisins sémantiques par requête
  min_weight: 0.1              # Seuil PPMI minimum
  max_context_length: 200      # Taille max contexte (caractères)
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | true | Activer injection de contexte (false = LLM standard) |
| `max_keywords` | integer | 5 | Nombre max de mots-clés extraits par message utilisateur |
| `max_neighbors` | integer | 15 | Nombre max de voisins sémantiques à injecter |
| `min_weight` | float | 0.1 | Seuil minimum pour poids PPMI (filtre relations faibles) |
| `max_context_length` | integer | 200 | Longueur max du contexte injecté (caractères) |

**Impact sur performance :**

| max_keywords | max_neighbors | Latence | Qualité contexte |
|--------------|---------------|---------|------------------|
| 3 | 10 | +50ms | Basique |
| 5 | 15 | +80ms | Équilibré (défaut) |
| 10 | 30 | +150ms | Riche |

**Désactiver le contexte :**
```yaml
context:
  enabled: false  # Mode LLM standard, sans graphe
```

**Configuration minimale (performance) :**
```yaml
context:
  enabled: true
  max_keywords: 3
  max_neighbors: 8
  min_weight: 0.2
  max_context_length: 150
```

**Configuration maximale (qualité) :**
```yaml
context:
  enabled: true
  max_keywords: 10
  max_neighbors: 30
  min_weight: 0.05
  max_context_length: 400
```

---

### Sessions

Gestion des sessions et historique de conversation.

```yaml
sessions:
  max_history_messages: 20     # Max messages d'historique par requête
  max_token_budget: 4000       # Budget approximatif tokens pour historique
  auto_cleanup_days: 30        # Suppression sessions inactives > N jours
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `max_history_messages` | integer | 20 | Nombre max de messages précédents inclus dans contexte |
| `max_token_budget` | integer | 4000 | Budget tokens approximatif pour historique (soft limit) |
| `auto_cleanup_days` | integer | 30 | Supprimer sessions inactives depuis N jours (0 = désactivé) |

**Sliding window :**

Le système utilise une fenêtre glissante pour l'historique :
```
Messages : 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 50
Window :                              [46 47 48 49 50]
```

**Token budget :**

Si l'historique dépasse le budget tokens, les messages les plus anciens sont tronqués :
```python
# Approximation : 1 token ≈ 4 caractères
total_chars = sum(len(msg) for msg in history)
if total_chars > max_token_budget * 4:
    # Truncate oldest messages
```

**Exemples :**

```yaml
# Conversations courtes (chatbot rapide)
sessions:
  max_history_messages: 5
  max_token_budget: 1000
  auto_cleanup_days: 7

# Conversations longues (support client)
sessions:
  max_history_messages: 50
  max_token_budget: 8000
  auto_cleanup_days: 90

# Mode stateless (pas de mémoire)
sessions:
  max_history_messages: 0
  max_token_budget: 0
  auto_cleanup_days: 1
```

---

### Performance

Paramètres de performance HTTP et rate limiting.

```yaml
performance:
  connection_pool_size: 10      # Taille pool connexions HTTP
  request_timeout: 180.0        # Timeout global requête
  max_concurrent_requests: 100  # Limite requêtes concurrentes
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `connection_pool_size` | integer | 10 | Taille du pool de connexions HTTP réutilisables |
| `request_timeout` | float | 180.0 | Timeout global pour toute requête (secondes) |
| `max_concurrent_requests` | integer | 100 | Nombre max de requêtes concurrentes (soft limit) |

**Tuning pour charge élevée :**
```yaml
performance:
  connection_pool_size: 50      # Plus de connexions réutilisables
  request_timeout: 300.0        # Timeout augmenté
  max_concurrent_requests: 500  # Plus de parallélisme
```

**Tuning pour ressources limitées :**
```yaml
performance:
  connection_pool_size: 5
  request_timeout: 120.0
  max_concurrent_requests: 20
```

---

### Logging

Configuration des logs applicatifs.

```yaml
logging:
  level: "info"                # Niveau : debug, info, warning, error, critical
  format: "json"               # Format : json, text
  file: "logs/lyra.log"        # Chemin fichier log (optionnel)
  rotate_size_mb: 100          # Rotation à N MB
  rotate_backups: 5            # Nombre de backups conservés
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `level` | string | "info" | Niveau minimum de log |
| `format` | string | "json" | Format de sortie (json pour parsing, text pour humain) |
| `file` | string | "logs/lyra.log" | Chemin fichier log (null = stdout uniquement) |
| `rotate_size_mb` | integer | 100 | Taille max avant rotation (MB) |
| `rotate_backups` | integer | 5 | Nombre de fichiers backup conservés |

**Niveaux de log :**

| Niveau | Usage | Verbosité |
|--------|-------|-----------|
| `debug` | Développement, debugging | Très élevée |
| `info` | Production standard | Modérée |
| `warning` | Production, uniquement warnings | Faible |
| `error` | Production, uniquement erreurs | Minimale |
| `critical` | Uniquement erreurs critiques | Très faible |

**Format JSON (production) :**
```json
{
  "timestamp": "2025-01-14T10:30:00Z",
  "level": "INFO",
  "logger": "app.api.chat",
  "message": "Chat request processed",
  "session_id": "abc123...",
  "latency_ms": 1234.5
}
```

**Format text (développement) :**
```
2025-01-14 10:30:00 INFO     app.api.chat - Chat request processed
```

**Exemples :**

```yaml
# Développement
logging:
  level: "debug"
  format: "text"
  file: null  # stdout uniquement

# Production
logging:
  level: "info"
  format: "json"
  file: "/var/log/lyra/app.log"
  rotate_size_mb: 500
  rotate_backups: 10

# Production silencieuse
logging:
  level: "error"
  format: "json"
  file: "/var/log/lyra/errors.log"
  rotate_size_mb: 100
  rotate_backups: 3
```

---

### CORS

Configuration Cross-Origin Resource Sharing.

```yaml
cors:
  enabled: true                # Activer/désactiver CORS
  origins:                     # Liste origines autorisées
    - "http://localhost:3000"
    - "http://localhost:8080"
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | true | Activer CORS |
| `origins` | array | ["*"] | Liste des origines autorisées (["*"] = toutes) |

**⚠️ Sécurité :**

- **Jamais utiliser `["*"]` en production !**
- Listez explicitement vos domaines frontend

**Exemples :**

```yaml
# Développement (permissif)
cors:
  enabled: true
  origins:
    - "*"  # Toutes origines (dev uniquement !)

# Production (restrictif)
cors:
  enabled: true
  origins:
    - "https://lyra.example.com"
    - "https://app.example.com"

# Désactivé (API interne uniquement)
cors:
  enabled: false
```

---

### Monitoring

Monitoring et métriques (fonctionnalité future).

```yaml
monitoring:
  enabled: false               # Activer monitoring
  prometheus_port: 9090        # Port Prometheus
  health_check_interval: 60    # Intervalle health checks (secondes)
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | false | Activer endpoint Prometheus |
| `prometheus_port` | integer | 9090 | Port pour exposition métriques |
| `health_check_interval` | integer | 60 | Intervalle entre health checks automatiques |

**Métriques exposées (futur) :**
- `lyra_requests_total` : Compteur requêtes
- `lyra_request_duration_seconds` : Histogram latence
- `lyra_consciousness_level` : Gauge niveau conscience moyen
- `lyra_db_size_bytes` : Gauge taille base de données
- `lyra_active_sessions` : Gauge sessions actives

**Configuration production (futur) :**
```yaml
monitoring:
  enabled: true
  prometheus_port: 9090
  health_check_interval: 30
```

---

### Security

Sécurité et rate limiting.

```yaml
security:
  api_key_enabled: false       # Activer authentification API key
  rate_limit_per_minute: 60    # Requêtes max par minute par IP
  max_request_size_mb: 10      # Taille max corps requête (MB)
```

**Paramètres :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `api_key_enabled` | boolean | false | Activer authentification par API key |
| `rate_limit_per_minute` | integer | 60 | Nombre max de requêtes par minute par IP |
| `max_request_size_mb` | integer | 10 | Taille max du corps de requête (MB) |

**API Key :**

Si activé, les requêtes doivent inclure le header :
```
X-API-Key: your-secret-key
```

Clés stockées dans variables d'environnement :
```bash
export LYRA_API_KEY="your-secret-key"
```

**Rate limiting :**

Réponse si limite atteinte :
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642156800

{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

**Exemples :**

```yaml
# Développement (ouvert)
security:
  api_key_enabled: false
  rate_limit_per_minute: 1000  # Pas de limite pratique
  max_request_size_mb: 50

# Production (sécurisé)
security:
  api_key_enabled: true
  rate_limit_per_minute: 30    # Limite stricte
  max_request_size_mb: 5

# Production (charge élevée)
security:
  api_key_enabled: true
  rate_limit_per_minute: 200   # Plus permissif
  max_request_size_mb: 10
```

---

## Variables d'environnement

Variables d'environnement qui override `config.yaml`.

| Variable | Override | Type | Exemple |
|----------|----------|------|---------|
| `LYRA_MODEL` | `llm.model` | string | `mistral:latest` |
| `LYRA_OLLAMA_URL` | `llm.base_url` | string | `http://ollama:11434` |
| `LYRA_NUM_CTX` | `llm.num_ctx` | integer | `16384` |
| `LYRA_DB_PATH` | `database.path` | string | `/var/lib/lyra/db` |
| `LYRA_API_KEY` | - | string | `secret123` |
| `LYRA_LOG_LEVEL` | `logging.level` | string | `debug` |
| `LYRA_PORT` | `server.port` | integer | `9000` |

**Utilisation :**

```bash
# Linux/Mac
export LYRA_MODEL="mistral:latest"
export LYRA_NUM_CTX="16384"
export LYRA_OLLAMA_URL="http://192.168.1.100:11434"
uvicorn app.main:app

# Windows
set LYRA_MODEL=mistral:latest
set LYRA_NUM_CTX=16384
set LYRA_OLLAMA_URL=http://192.168.1.100:11434
uvicorn app.main:app
```

**Docker Compose :**
```yaml
services:
  lyra:
    environment:
      - LYRA_DB_PATH=/data/ispace.db
      - LYRA_OLLAMA_URL=http://ollama:11434
      - LYRA_LOG_LEVEL=info
```

---

## Configuration production

Exemple complet de configuration production sécurisée.

```yaml
# config.production.yaml

server:
  host: "0.0.0.0"
  port: 8000
  workers: 8                    # 8 CPU cores
  reload: false
  log_level: "info"

database:
  path: "/var/lib/lyra/ispace.db"
  backup_interval_hours: 6
  vacuum_interval_days: 1

llm:
  base_url: "http://ollama-server:11434"
  model: "gpt-oss:20b"
  timeout: 180.0
  max_retries: 5

physics:
  default_profile: "balanced"
  time_mapping: "logarithmic"
  max_messages_window: 200

context:
  enabled: true
  max_keywords: 5
  max_neighbors: 15
  min_weight: 0.1
  max_context_length: 200

sessions:
  max_history_messages: 20
  max_token_budget: 4000
  auto_cleanup_days: 90

performance:
  connection_pool_size: 50
  request_timeout: 300.0
  max_concurrent_requests: 500

logging:
  level: "info"
  format: "json"
  file: "/var/log/lyra/app.log"
  rotate_size_mb: 500
  rotate_backups: 10

cors:
  enabled: true
  origins:
    - "https://lyra.example.com"
    - "https://app.example.com"

monitoring:
  enabled: true
  prometheus_port: 9090
  health_check_interval: 30

security:
  api_key_enabled: true
  rate_limit_per_minute: 100
  max_request_size_mb: 5
```

**Utilisation :**
```bash
cp config.production.yaml config.yaml
# Ou :
export LYRA_CONFIG_FILE=config.production.yaml
uvicorn app.main:app
```

---

## Exemples de configurations

### Configuration développement

```yaml
# config.dev.yaml

server:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  reload: true
  log_level: "debug"

database:
  path: "data/ispace.db"
  backup_interval_hours: 0     # Désactivé
  vacuum_interval_days: 0      # Désactivé

llm:
  base_url: "http://localhost:11434"
  model: "llama3:8b"           # Modèle léger pour dev
  timeout: 60.0
  max_retries: 1

logging:
  level: "debug"
  format: "text"
  file: null

cors:
  enabled: true
  origins:
    - "*"

security:
  api_key_enabled: false
  rate_limit_per_minute: 10000  # Pas de limite
  max_request_size_mb: 100
```

### Configuration Docker

```yaml
# config.docker.yaml

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  log_level: "info"

database:
  path: "/data/ispace.db"      # Volume monté
  backup_interval_hours: 12
  vacuum_interval_days: 3

llm:
  base_url: "http://ollama:11434"  # Service Docker
  model: "gpt-oss:20b"
  timeout: 120.0
  max_retries: 3

logging:
  level: "info"
  format: "json"
  file: "/logs/lyra.log"       # Volume monté
  rotate_size_mb: 200
  rotate_backups: 5
```

**docker-compose.yml :**
```yaml
services:
  lyra:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./logs:/logs
      - ./config.docker.yaml:/app/config.yaml
    environment:
      - LYRA_CONFIG_FILE=/app/config.yaml
```

### Configuration tests

```yaml
# config.test.yaml

server:
  host: "127.0.0.1"
  port: 8001               # Port différent pour éviter conflits
  workers: 1
  reload: false
  log_level: "warning"     # Moins verbeux

database:
  path: ":memory:"         # SQLite en mémoire (ephemeral)
  backup_interval_hours: 0
  vacuum_interval_days: 0

llm:
  base_url: "http://localhost:11434"
  model: "mistral:latest"  # Modèle rapide
  timeout: 30.0
  max_retries: 1

sessions:
  max_history_messages: 5  # Minimal pour tests
  max_token_budget: 500
  auto_cleanup_days: 0

logging:
  level: "warning"
  format: "text"
  file: null

security:
  rate_limit_per_minute: 100000  # Pas de limite
```

---

## Support

- 📖 [Guide utilisateur](USER_GUIDE.md)
- 🔧 [Guide développeur](DEVELOPER_GUIDE.md)
- 🐛 [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues)
