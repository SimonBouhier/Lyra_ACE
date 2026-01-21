# Lyra Clean Configuration

Complete configuration guide for the Lyra Clean system via the `config.yaml` file.

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Configuration Sections](#configuration-sections)
4. [Environment Variables](#environment-variables)
5. [Production Configuration](#production-configuration)
6. [Configuration Examples](#configuration-examples)

---

## Overview

Lyra Clean uses a centralized `config.yaml` file for all system configuration.

**Location:** `config.yaml` (project root)

**Format:** YAML (strict indentation, 2 spaces)

**Loading:** At application startup (see `app/main.py`)

---

## File Structure

```yaml
server:          # FastAPI server configuration
database:        # SQLite configuration
llm:             # Ollama configuration
physics:         # Bézier physics engine
context:         # Semantic context injection
sessions:        # Session management
performance:     # Performance parameters
logging:         # Logging configuration
cors:            # Cross-Origin Resource Sharing
monitoring:      # Monitoring (future)
security:        # Security and rate limiting
```

---

## Configuration Sections

### Server

FastAPI server and Uvicorn configuration.

```yaml
server:
  host: "0.0.0.0"              # Listen interface (0.0.0.0 = all interfaces)
  port: 8000                   # HTTP port
  workers: 4                   # Number of Uvicorn workers (multi-processing)
  reload: false                # Auto-reload on code change (dev only)
  log_level: "info"            # Log level: debug, info, warning, error, critical
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `host` | string | "0.0.0.0" | IP address to listen on (0.0.0.0 = all) |
| `port` | integer | 8000 | HTTP server port |
| `workers` | integer | 4 | Number of workers (1 per CPU recommended) |
| `reload` | boolean | false | Auto-reload (dev only, disable in prod) |
| `log_level` | string | "info" | Log verbosity (debug < info < warning < error < critical) |

**Recommendations:**

- **Development:**
  ```yaml
  host: "127.0.0.1"  # Localhost only
  port: 8000
  workers: 1         # Single worker for debugging
  reload: true       # Auto-reload convenient
  log_level: "debug" # Detailed logs
  ```

- **Production:**
  ```yaml
  host: "0.0.0.0"    # Accessible from outside
  port: 8000
  workers: 4         # 1 per CPU (adjust for machine)
  reload: false      # Disable
  log_level: "info"  # Avoid too many logs
  ```

---

### Database

SQLite configuration and maintenance parameters.

```yaml
database:
  path: "data/ispace.db"       # Path to SQLite file
  backup_interval_hours: 24    # Automatic backup frequency
  vacuum_interval_days: 7      # VACUUM + ANALYZE frequency
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `path` | string | "data/ispace.db" | Relative or absolute path to SQLite file |
| `backup_interval_hours` | integer | 24 | Automatic backup every N hours (0 = disabled) |
| `vacuum_interval_days` | integer | 7 | Defragmentation every N days (0 = disabled) |

**Notes:**

- **Relative path**: Relative to working directory (CWD)
- **Backup**: Creates a `.backup` file in the same directory
- **VACUUM**: Defragments and optimizes the database (can take time on large DBs)

**Production example:**
```yaml
database:
  path: "/var/lib/lyra/ispace.db"  # Absolute path
  backup_interval_hours: 6         # Backup every 6h
  vacuum_interval_days: 1          # Daily vacuum
```

---

### LLM (Ollama)

Ollama client configuration for LLM inference.

```yaml
llm:
  base_url: "http://localhost:11434"  # Ollama server URL
  model: "gpt-oss:20b"                # Default model
  timeout: 120.0                      # Request timeout (seconds)
  max_retries: 3                      # Retry attempts on failure
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `base_url` | string | "http://localhost:11434" | Complete Ollama server URL |
| `model` | string | "gpt-oss:20b" | Model name (must be installed via `ollama pull`) |
| `timeout` | float | 120.0 | Request timeout in seconds |
| `max_retries` | integer | 3 | Number of retry attempts on network failure |

**Popular models:**

| Model | Size | RAM required | Usage |
|--------|--------|-------------|-------|
| `llama3:8b` | 8B params | ~6GB | Fast, general |
| `llama3:70b` | 70B params | ~40GB | High quality |
| `gpt-oss:20b` | 20B params | ~12GB | Balanced |
| `mistral:latest` | 7B params | ~5GB | Fast, efficient |

**Multi-model configuration:**

To use different models per endpoint (future):
```yaml
llm:
  base_url: "http://localhost:11434"
  models:
    default: "gpt-oss:20b"
    creative: "llama3:70b"    # High quality
    fast: "mistral:latest"    # Fast responses
```

**Remote Ollama:**
```yaml
llm:
  base_url: "http://192.168.1.100:11434"  # Remote server
  timeout: 180.0                           # Increased timeout
```

---

### Physics

Bézier physics engine configuration.

```yaml
physics:
  default_profile: "balanced"     # Default profile
  time_mapping: "logarithmic"     # Time mapping: linear, logarithmic, sigmoid
  max_messages_window: 100        # Max window for t ∈ [0, 1]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `default_profile` | string | "balanced" | Default Bézier profile name |
| `time_mapping` | string | "logarithmic" | Time mapping type (linear, logarithmic, sigmoid) |
| `max_messages_window` | integer | 100 | Max number of messages to reach t=1.0 |

**Time mapping:**

- **linear**: `t = n / max_messages`
  - Regular progression
  - Good for short conversations

- **logarithmic** (default): `t = log(1 + n) / log(1 + max_messages)`
  - Slower progress at start
  - Good for long conversations (more time for adaptation)

- **sigmoid**: `t = 1 / (1 + exp(-(n/max - 0.5)*10))`
  - S-curve, smooth transitions
  - Good to avoid abrupt changes

**Examples:**

```yaml
# Short and dynamic conversations
physics:
  default_profile: "aggressive"
  time_mapping: "linear"
  max_messages_window: 50

# Long conversations with stability
physics:
  default_profile: "balanced"
  time_mapping: "logarithmic"
  max_messages_window: 200

# Very smooth transitions
physics:
  default_profile: "conservative"
  time_mapping: "sigmoid"
  max_messages_window: 100
```

---

### Context

Semantic context injection configuration.

```yaml
context:
  enabled: true                # Enable/disable semantic context
  max_keywords: 5              # Max keywords extracted per message
  max_neighbors: 15            # Max semantic neighbors per query
  min_weight: 0.1              # Minimum PPMI threshold
  max_context_length: 200      # Max context size (characters)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | true | Enable context injection (false = standard LLM) |
| `max_keywords` | integer | 5 | Max number of keywords extracted per user message |
| `max_neighbors` | integer | 15 | Max number of semantic neighbors to inject |
| `min_weight` | float | 0.1 | Minimum threshold for PPMI weight (filters weak relations) |
| `max_context_length` | integer | 200 | Max length of injected context (characters) |

**Performance impact:**

| max_keywords | max_neighbors | Latency | Context quality |
|--------------|---------------|---------|------------------|
| 3 | 10 | +50ms | Basic |
| 5 | 15 | +80ms | Balanced (default) |
| 10 | 30 | +150ms | Rich |

**Disable context:**
```yaml
context:
  enabled: false  # Standard LLM mode, no graph
```

**Minimal configuration (performance):**
```yaml
context:
  enabled: true
  max_keywords: 3
  max_neighbors: 8
  min_weight: 0.2
  max_context_length: 150
```

**Maximum configuration (quality):**
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

Session management and conversation history.

```yaml
sessions:
  max_history_messages: 20     # Max history messages per request
  max_token_budget: 4000       # Approximate token budget for history
  auto_cleanup_days: 30        # Delete inactive sessions > N days
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `max_history_messages` | integer | 20 | Max number of previous messages included in context |
| `max_token_budget` | integer | 4000 | Approximate token budget for history (soft limit) |
| `auto_cleanup_days` | integer | 30 | Delete sessions inactive for N days (0 = disabled) |

**Sliding window:**

The system uses a sliding window for history:
```
Messages: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 50
Window:                              [46 47 48 49 50]
```

**Token budget:**

If history exceeds token budget, oldest messages are truncated:
```python
# Approximation: 1 token ≈ 4 characters
total_chars = sum(len(msg) for msg in history)
if total_chars > max_token_budget * 4:
    # Truncate oldest messages
```

**Examples:**

```yaml
# Short conversations (fast chatbot)
sessions:
  max_history_messages: 5
  max_token_budget: 1000
  auto_cleanup_days: 7

# Long conversations (customer support)
sessions:
  max_history_messages: 50
  max_token_budget: 8000
  auto_cleanup_days: 90

# Stateless mode (no memory)
sessions:
  max_history_messages: 0
  max_token_budget: 0
  auto_cleanup_days: 1
```

---

### Performance

HTTP performance parameters and rate limiting.

```yaml
performance:
  connection_pool_size: 10      # HTTP connection pool size
  request_timeout: 180.0        # Global request timeout
  max_concurrent_requests: 100  # Concurrent request limit
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `connection_pool_size` | integer | 10 | Size of reusable HTTP connection pool |
| `request_timeout` | float | 180.0 | Global timeout for any request (seconds) |
| `max_concurrent_requests` | integer | 100 | Max number of concurrent requests (soft limit) |

**Tuning for high load:**
```yaml
performance:
  connection_pool_size: 50      # More reusable connections
  request_timeout: 300.0        # Increased timeout
  max_concurrent_requests: 500  # More parallelism
```

**Tuning for limited resources:**
```yaml
performance:
  connection_pool_size: 5
  request_timeout: 120.0
  max_concurrent_requests: 20
```

---

### Logging

Application logging configuration.

```yaml
logging:
  level: "info"                # Level: debug, info, warning, error, critical
  format: "json"               # Format: json, text
  file: "logs/lyra.log"        # Log file path (optional)
  rotate_size_mb: 100          # Rotation at N MB
  rotate_backups: 5            # Number of backups kept
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `level` | string | "info" | Minimum log level |
| `format` | string | "json" | Output format (json for parsing, text for human) |
| `file` | string | "logs/lyra.log" | Log file path (null = stdout only) |
| `rotate_size_mb` | integer | 100 | Max size before rotation (MB) |
| `rotate_backups` | integer | 5 | Number of backup files kept |

**Log levels:**

| Level | Usage | Verbosity |
|--------|-------|-----------|
| `debug` | Development, debugging | Very high |
| `info` | Standard production | Moderate |
| `warning` | Production, warnings only | Low |
| `error` | Production, errors only | Minimal |
| `critical` | Critical errors only | Very low |

**JSON format (production):**
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

**Text format (development):**
```
2025-01-14 10:30:00 INFO     app.api.chat - Chat request processed
```

**Examples:**

```yaml
# Development
logging:
  level: "debug"
  format: "text"
  file: null  # stdout only

# Production
logging:
  level: "info"
  format: "json"
  file: "/var/log/lyra/app.log"
  rotate_size_mb: 500
  rotate_backups: 10

# Silent production
logging:
  level: "error"
  format: "json"
  file: "/var/log/lyra/errors.log"
  rotate_size_mb: 100
  rotate_backups: 3
```

---

### CORS

Cross-Origin Resource Sharing configuration.

```yaml
cors:
  enabled: true                # Enable/disable CORS
  origins:                     # List of allowed origins
    - "http://localhost:3000"
    - "http://localhost:8080"
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | true | Enable CORS |
| `origins` | array | ["*"] | List of allowed origins (["*"] = all) |

**⚠️ Security:**

- **Never use `["*"]` in production!**
- Explicitly list your frontend domains

**Examples:**

```yaml
# Development (permissive)
cors:
  enabled: true
  origins:
    - "*"  # All origins (dev only!)

# Production (restrictive)
cors:
  enabled: true
  origins:
    - "https://lyra.example.com"
    - "https://app.example.com"

# Disabled (internal API only)
cors:
  enabled: false
```

---

### Monitoring

Monitoring and metrics (future feature).

```yaml
monitoring:
  enabled: false               # Enable monitoring
  prometheus_port: 9090        # Prometheus port
  health_check_interval: 60    # Health check interval (seconds)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `enabled` | boolean | false | Enable Prometheus endpoint |
| `prometheus_port` | integer | 9090 | Port for metrics exposition |
| `health_check_interval` | integer | 60 | Interval between automatic health checks |

**Exposed metrics (future):**
- `lyra_requests_total`: Request counter
- `lyra_request_duration_seconds`: Latency histogram
- `lyra_consciousness_level`: Average consciousness level gauge
- `lyra_db_size_bytes`: Database size gauge
- `lyra_active_sessions`: Active sessions gauge

**Production configuration (future):**
```yaml
monitoring:
  enabled: true
  prometheus_port: 9090
  health_check_interval: 30
```

---

### Security

Security and rate limiting.

```yaml
security:
  api_key_enabled: false       # Enable API key authentication
  rate_limit_per_minute: 60    # Max requests per minute per IP
  max_request_size_mb: 10      # Max request body size (MB)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `api_key_enabled` | boolean | false | Enable API key authentication |
| `rate_limit_per_minute` | integer | 60 | Max number of requests per minute per IP |
| `max_request_size_mb` | integer | 10 | Max request body size (MB) |

**API Key:**

If enabled, requests must include the header:
```
X-API-Key: your-secret-key
```

Keys stored in environment variables:
```bash
export LYRA_API_KEY="your-secret-key"
```

**Rate limiting:**

Response when limit reached:
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642156800

{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

**Examples:**

```yaml
# Development (open)
security:
  api_key_enabled: false
  rate_limit_per_minute: 1000  # No practical limit
  max_request_size_mb: 50

# Production (secure)
security:
  api_key_enabled: true
  rate_limit_per_minute: 30    # Strict limit
  max_request_size_mb: 5

# Production (high load)
security:
  api_key_enabled: true
  rate_limit_per_minute: 200   # More permissive
  max_request_size_mb: 10
```

---

## Environment Variables

Environment variables that override `config.yaml`.

| Variable | Override | Type | Example |
|----------|----------|------|---------|
| `LYRA_MODEL` | `llm.model` | string | `mistral:latest` |
| `LYRA_OLLAMA_URL` | `llm.base_url` | string | `http://ollama:11434` |
| `LYRA_NUM_CTX` | `llm.num_ctx` | integer | `16384` |
| `LYRA_DB_PATH` | `database.path` | string | `/var/lib/lyra/db` |
| `LYRA_API_KEY` | - | string | `secret123` |
| `LYRA_LOG_LEVEL` | `logging.level` | string | `debug` |
| `LYRA_PORT` | `server.port` | integer | `9000` |

**Usage:**

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

**Docker Compose:**
```yaml
services:
  lyra:
    environment:
      - LYRA_DB_PATH=/data/ispace.db
      - LYRA_OLLAMA_URL=http://ollama:11434
      - LYRA_LOG_LEVEL=info
```

---

## Production Configuration

Complete example of secure production configuration.

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

**Usage:**
```bash
cp config.production.yaml config.yaml
# Or:
export LYRA_CONFIG_FILE=config.production.yaml
uvicorn app.main:app
```

---

## Configuration Examples

### Development Configuration

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
  backup_interval_hours: 0     # Disabled
  vacuum_interval_days: 0      # Disabled

llm:
  base_url: "http://localhost:11434"
  model: "llama3:8b"           # Lightweight model for dev
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
  rate_limit_per_minute: 10000  # No limit
  max_request_size_mb: 100
```

### Docker Configuration

```yaml
# config.docker.yaml

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  log_level: "info"

database:
  path: "/data/ispace.db"      # Mounted volume
  backup_interval_hours: 12
  vacuum_interval_days: 3

llm:
  base_url: "http://ollama:11434"  # Docker service
  model: "gpt-oss:20b"
  timeout: 120.0
  max_retries: 3

logging:
  level: "info"
  format: "json"
  file: "/logs/lyra.log"       # Mounted volume
  rotate_size_mb: 200
  rotate_backups: 5
```

**docker-compose.yml:**
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

### Test Configuration

```yaml
# config.test.yaml

server:
  host: "127.0.0.1"
  port: 8001               # Different port to avoid conflicts
  workers: 1
  reload: false
  log_level: "warning"     # Less verbose

database:
  path: ":memory:"         # In-memory SQLite (ephemeral)
  backup_interval_hours: 0
  vacuum_interval_days: 0

llm:
  base_url: "http://localhost:11434"
  model: "mistral:latest"  # Fast model
  timeout: 30.0
  max_retries: 1

sessions:
  max_history_messages: 5  # Minimal for tests
  max_token_budget: 500
  auto_cleanup_days: 0

logging:
  level: "warning"
  format: "text"
  file: null

security:
  rate_limit_per_minute: 100000  # No limit
```

---

## Support

- 📖 [User guide](USER_GUIDE.md)
- 🔧 [Developer guide](DEVELOPER_GUIDE.md)
- 🐛 [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues)
