# Lyra Clean User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Usage](#api-usage)
5. [Consciousness Levels](#consciousness-levels)
6. [Bézier Profiles](#bézier-profiles)
7. [Session Management](#session-management)
8. [Web Interface](#web-interface)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

Lyra Clean is an LLM conversational system that controls text generation through mathematical trajectories (Bézier curves) rather than static settings. This enables dynamic and predictable behavior throughout a conversation.

### Why Lyra Clean?

**Traditional Problem:**
- Static LLM parameters (fixed temperature = 0.7)
- Unpredictable behavior in long conversations
- Difficult to calibrate reactive adjustments

**Lyra Solution:**
- Bézier trajectories defining parameter evolution
- Predictable ballistic behavior (like a physical trajectory)
- Three consciousness levels for contextual adaptation

### Key Concepts

- **Deterministic Physics**: Parameters evolve according to mathematical curves
- **Epistemological Consciousness**: The system observes and adapts itself
- **Semantic Context**: Intelligent injection of knowledge from a graph
- **Semantic Memory**: Recall of past conversations through similarity

---

## Installation

### Prerequisites

1. **Python 3.10 or higher**
   ```bash
   python --version  # Must show Python 3.10.x or higher
   ```

2. **Ollama installed and running**
   - Download from [ollama.ai](https://ollama.ai/)
   - Install and start the service
   - Verify: `ollama list`

3. **LLM model available**
   ```bash
   ollama pull gpt-oss:20b
   # Or use another model and modify config.yaml
   ```

### Standard Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/lyra_clean_bis.git
cd lyra_clean_bis

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize the database
python -c "from database.engine import ISpaceDB; import asyncio; asyncio.run(ISpaceDB('data/ispace.db').initialize())"
```

### Docker Installation (Alternative)

```bash
# Build and launch
docker-compose up --build

# In background
docker-compose up -d
```

---

## Quick Start

### Start the Server

**Option 1: Automatic Script (Windows)**
```bash
start_server.bat
```

**Option 2: Manual**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Docker**
```bash
docker-compose up
```

The server starts on: **http://localhost:8000**

### Check Status

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "database": {"connected": true, "concepts": 1234},
  "ollama": {"connected": true, "model": "gpt-oss:20b"}
}
```

### First Conversation

```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello Lyra, who are you?",
    "consciousness_level": 0
  }'
```

**Response:**
```json
{
  "response": "I am Lyra Clean, a conversational system...",
  "session_id": "abc123...",
  "message_index": 1,
  "physics_state": {
    "t": 0.01,
    "tau_c": 0.45,
    "rho": 0.5,
    "delta_r": 0.3
  },
  "metadata": {
    "latency_ms": 1234,
    "tokens": {"prompt": 45, "completion": 120}
  }
}
```

---

## API Usage

### Main Endpoint: Chat

**POST /chat/message**

Send a message and receive a response.

#### Request Parameters

```json
{
  "message": "Your message here",
  "session_id": "abc123...",           // Optional: reuse a session
  "consciousness_level": 2,            // 0-3 (default: 0)
  "profile_name": "balanced",          // Bézier profile (default: "balanced")
  "max_history": 10,                   // Context messages (default: 20)
  "max_context_length": 200            // Max semantic context size
}
```

#### Required Fields
- `message`: User message (string, max 10000 characters)

#### Optional Fields
- `session_id`: Existing session ID (auto-generated if omitted)
- `consciousness_level`: Introspection level (0-3, default 0)
- `profile_name`: Bézier profile name to use
- `max_history`: Number of previous messages to include
- `max_context_length`: Maximum size of injected semantic context

#### Response

```json
{
  "response": "Generated response",
  "session_id": "abc123...",
  "message_index": 5,
  "physics_state": {
    "t": 0.05,                // Normalized time [0, 1]
    "tau_c": 0.52,            // Tension/temperature
    "rho": 0.48,              // Focus/polarity
    "delta_r": 0.35,          // Planning
    "kappa": 0.6              // Curvature/style (optional)
  },
  "consciousness": {          // If consciousness_level >= 1
    "coherence": 0.82,
    "tension": 0.45,
    "fit": 0.91,
    "pressure": 0.38,
    "stability_score": 0.87,
    "suggestion": null        // If level 2, contains suggested adjustments
  },
  "memory_echoes": [          // If consciousness_level >= 3
    {
      "content": "Recalled message from the past",
      "similarity": 0.89,
      "turns_ago": 12
    }
  ],
  "semantic_context": [       // Context injected from the graph
    "concept_a (weight=0.82)",
    "concept_b (weight=0.75)"
  ],
  "metadata": {
    "latency_ms": 1234,
    "tokens": {
      "prompt": 456,
      "completion": 123,
      "total": 579
    }
  }
}
```

### Practical Examples

#### Simple Conversation (level 0)

```python
import requests

response = requests.post("http://localhost:8000/chat/message", json={
    "message": "Explain special relativity to me",
    "consciousness_level": 0
})

print(response.json()["response"])
```

#### Conversation with Memory (level 3)

```python
# First message
r1 = requests.post("http://localhost:8000/chat/message", json={
    "message": "My name is Alice and I love quantum physics",
    "consciousness_level": 3
})
session_id = r1.json()["session_id"]

# Later in the conversation...
r2 = requests.post("http://localhost:8000/chat/message", json={
    "message": "What is my name and what am I interested in?",
    "session_id": session_id,
    "consciousness_level": 3
})

# Lyra should remember thanks to semantic memory
print(r2.json()["response"])
# Should mention "Alice" and "quantum physics"

# Check memory echoes
print(r2.json()["memory_echoes"])
```

#### Aggressive Profile for Brainstorming

```python
response = requests.post("http://localhost:8000/chat/message", json={
    "message": "Suggest 10 original ideas for a sci-fi novel",
    "profile_name": "aggressive",  # High temperature, exploratory
    "consciousness_level": 2       # Adaptive
})
```

---

## Consciousness Levels

Lyra has 4 consciousness levels (0-3) that determine its degree of introspection and adaptation.

### Level 0: Passive

**Behavior:**
- No introspection
- Standard generation only
- Maximum performance (no additional computation)

**Use Cases:**
- Simple conversations
- Factual queries
- Performance-critical situations

**Example:**
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?", "consciousness_level": 0}'
```

### Level 1: Observer

**Behavior:**
- Compute epistemological metrics
- No action, monitoring only
- Returns metrics in response

**Computed Metrics:**
- **Coherence** (0-1): Semantic density of injected context
- **Tension** (0-1): System stress (temperature × response length)
- **Fit** (0-1): Alignment between expected/actual length
- **Pressure** (0-1): Exploration vs exploitation
- **Stability Score** (0-1): Composite stability score

**Use Cases:**
- Behavior debugging
- Performance analysis
- Research on artificial consciousness

**Example:**
```json
{
  "message": "Tell me about quantum mechanics",
  "consciousness_level": 1
}

// Response includes:
{
  "consciousness": {
    "coherence": 0.75,
    "tension": 0.42,
    "fit": 0.88,
    "pressure": 0.31,
    "stability_score": 0.79
  }
}
```

### Level 2: Adaptive

**Behavior:**
- Inherits level 1 (metrics)
- **Automatically applies** adjustments to Bézier parameters (τ_c, ρ, δ_r)
- Feedback loop: metrics from interaction N-1 adapt interaction N
- Gradual modifications (5-7.5% per turn)

**Adaptation Rules:**

1. **High tension (> 0.75)**
   - Reduce τ_c by 5% (decrease temperature)
   - Reason: Stabilize the system

2. **Low coherence (< 0.3)**
   - Adjust ρ towards focus
   - Reason: Improve contextual relevance

3. **High fit (> 0.8) + Stability (> 0.7)**
   - Encourage exploration (increase τ_c)
   - Reason: Avoid over-optimization

4. **High pressure (> 0.85)**
   - Reduce τ_c by 7.5% and δ_r by 5%
   - Reason: Lighten system load

5. **Long session (> 30 messages) + stable tension**
   - No change
   - Reason: Convergence achieved

**Use Cases:**
- Long and complex conversations
- Real-time self-adjustment
- Automatic optimization

**Example:**
```json
{
  "message": "Continue the discussion",
  "consciousness_level": 2
}

// Response may include:
{
  "consciousness": {
    "suggestion": {
      "reason": "High tension detected",
      "adjustments": {
        "tau_c": -0.05  // Reduce by 5%
      }
    }
  }
}
```

### Level 3: Semantic Memory

**Behavior:**
- Inherits level 2 (metrics + adaptation)
- Records each message with embeddings (1024D)
- Recalls similar messages through cosine similarity
- Applies temporal decay: `max(0.5, 1.0 - turns_ago * 0.01)`

**How It Works:**

1. **Recording:**
   - Each message → mxbai-embed-large embeddings
   - In-memory storage (dict: session_id → entries)
   - Limit: 50 entries per session

2. **Recall:**
   - Compute cosine similarity with current message
   - Threshold: 0.7 minimum
   - Limit: 3 best matches
   - Decay: -1% per elapsed turn

3. **Injection:**
   - Added to system context as `[MEMORY ECHO]`
   - Format: content + metadata (similarity, age)

**Use Cases:**
- Multi-turn conversations with continuity
- Follow-up questions on past topics
- Maintained personal context

**Example:**
```python
# Initial message
r1 = requests.post("http://localhost:8000/chat/message", json={
    "message": "My dog's name is Rex and he loves playing frisbee",
    "consciousness_level": 3
})
session_id = r1.json()["session_id"]

# 20 messages later...
r2 = requests.post("http://localhost:8000/chat/message", json={
    "message": "What's my dog's name again?",
    "session_id": session_id,
    "consciousness_level": 3
})

# Check memory echoes
echoes = r2.json()["memory_echoes"]
# [{"content": "My dog's name is Rex...", "similarity": 0.91, "turns_ago": 20}]
```

**Limitations:**
- ⚠️ RAM-only memory (lost on server restart)
- ⚠️ Limit of 50 messages per session
- ⚠️ Cost: +100ms per request (embedding generation)

### Level Comparison

| Level | Metrics | Adaptation | Memory | Latency | Usage |
|-------|---------|------------|--------|---------|-------|
| 0 | ❌ | ❌ | ❌ | ~1.2s | Fast production |
| 1 | ✅ | ❌ | ❌ | ~1.3s | Monitoring/debug |
| 2 | ✅ | ✅ | ❌ | ~1.4s | Complex conversations |
| 3 | ✅ | ✅ | ✅ | ~1.5s | Long-term memory |

---

## Bézier Profiles

Profiles define how parameters evolve throughout a conversation via cubic Bézier curves.

### Available Profiles

#### 1. Balanced (default)

**Characteristics:**
- Stable temperature around 0.7
- No extreme drift
- Good for general use

**Curves:**
```yaml
tau_c:  [0, 0.5] → [0.5, 0.45] → [0.55, 0.55] → [1, 0.5]
rho:    [0, 0.5] → [0.33, 0.5] → [0.67, 0.5] → [1, 0.5]
delta_r:[0, 0.3] → [0.33, 0.35] → [0.67, 0.35] → [1, 0.3]
```

**Usage:**
```json
{"message": "...", "profile_name": "balanced"}
```

#### 2. Aggressive

**Characteristics:**
- High initial temperature (0.8-1.0)
- Exploratory, creative
- Good for brainstorming

**Use Cases:**
- Idea generation
- Creative writing
- Concept exploration

**Example:**
```json
{
  "message": "Invent 5 original fantasy creatures",
  "profile_name": "aggressive"
}
```

#### 3. Conservative

**Characteristics:**
- Low temperature (0.3-0.5)
- Precise, factual
- Good for analytical tasks

**Use Cases:**
- Code generation
- Factual summaries
- Mathematical calculations

**Example:**
```json
{
  "message": "Write a Python function to sort a list",
  "profile_name": "conservative"
}
```

### Create a Custom Profile

Profiles are defined in the database. You can create them via SQL:

```sql
INSERT INTO bezier_profiles (name, description, tau_c_json, rho_json, delta_r_json)
VALUES (
  'custom_profile',
  'My custom profile',
  '[{"t": 0.0, "value": 0.6}, {"t": 0.33, "value": 0.7}, {"t": 0.67, "value": 0.5}, {"t": 1.0, "value": 0.4}]',
  '[{"t": 0.0, "value": 0.5}, {"t": 0.33, "value": 0.5}, {"t": 0.67, "value": 0.5}, {"t": 1.0, "value": 0.5}]',
  '[{"t": 0.0, "value": 0.3}, {"t": 0.33, "value": 0.4}, {"t": 0.67, "value": 0.3}, {"t": 1.0, "value": 0.2}]'
);
```

**Constraints:**
- Exactly 4 control points (t=0, t≈0.33, t≈0.67, t=1)
- `t` must be strictly increasing
- `value` ∈ [0, 1]

### List Profiles

```bash
curl http://localhost:8000/profiles

# Response:
{
  "profiles": [
    {
      "name": "balanced",
      "description": "Balanced temperature and focus",
      "parameters": ["tau_c", "rho", "delta_r"]
    },
    ...
  ]
}
```

### Preview a Profile

```bash
curl "http://localhost:8000/profiles/balanced?preview=20"

# Returns 20 sampled points from the trajectory
{
  "name": "balanced",
  "trajectory": [
    {"t": 0.0, "tau_c": 0.50, "rho": 0.50, "delta_r": 0.30},
    {"t": 0.05, "tau_c": 0.48, "rho": 0.50, "delta_r": 0.31},
    ...
  ]
}
```

---

## Session Management

### Create a Session

**POST /sessions**

```json
{
  "profile_name": "balanced",
  "max_messages": 100,
  "time_mapping": "logarithmic"
}
```

**Response:**
```json
{
  "session_id": "abc123...",
  "created_at": "2025-01-14T10:30:00Z",
  "profile_name": "balanced"
}
```

### Retrieve a Session

**GET /sessions/{session_id}**

```bash
curl http://localhost:8000/sessions/abc123...
```

**Response:**
```json
{
  "session_id": "abc123...",
  "created_at": "2025-01-14T10:30:00Z",
  "message_count": 15,
  "profile_name": "balanced",
  "time_mapping": "logarithmic"
}
```

### Conversation History

**GET /sessions/{session_id}/history**

```bash
curl http://localhost:8000/sessions/abc123.../history
```

**Response:**
```json
{
  "session_id": "abc123...",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-01-14T10:30:15Z",
      "message_index": 1
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you?",
      "timestamp": "2025-01-14T10:30:17Z",
      "message_index": 2,
      "physics_state": {"t": 0.01, "tau_c": 0.50, ...}
    },
    ...
  ],
  "total_messages": 15
}
```

### Delete a Session

**DELETE /sessions/{session_id}**

```bash
curl -X DELETE http://localhost:8000/sessions/abc123...
```

**Response:**
```json
{
  "success": true,
  "message": "Session deleted"
}
```

---

## Web Interface

Lyra includes a minimalist web interface for quick testing.

### Access

Open your browser: **http://localhost:8000**

### Features

- ✅ Send messages in real-time
- ✅ Select consciousness level (0-3)
- ✅ Select Bézier profile
- ✅ Display conversation history
- ✅ Consciousness metrics (if level ≥ 1)
- ✅ Memory echoes (if level 3)
- ✅ Visualization of physical state

### Source File

The interface is a static HTML file: `app/static/index.html`

You can customize it according to your needs.

---

## Troubleshooting

### Error: "Ollama server not reachable"

**Symptoms:**
```json
{
  "error": "Ollama request failed after 3 attempts"
}
```

**Solutions:**
1. Check that Ollama is running:
   ```bash
   ollama list
   ```

2. Check the URL in `config.yaml`:
   ```yaml
   llm:
     base_url: "http://localhost:11434"  # Default port
   ```

3. Test manually:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Error: "Model not found"

**Symptoms:**
```json
{
  "error": "HTTP 404: model 'gpt-oss:20b' not found"
}
```

**Solutions:**
1. Download the model:
   ```bash
   ollama pull gpt-oss:20b
   ```

2. Or modify `config.yaml` to use an available model:
   ```yaml
   llm:
     model: "llama3:latest"  # Or another installed model
   ```

### Error: "Database locked"

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
1. Check that no other process is using the DB:
   ```bash
   lsof data/ispace.db  # Linux/Mac
   ```

2. Enable WAL mode (already done by default):
   ```sql
   PRAGMA journal_mode=WAL;
   ```

3. Restart the server.

### Slow Performance

**Symptoms:**
- Latency > 5 seconds per request

**Solutions:**

1. **Disable consciousness if unnecessary:**
   ```json
   {"consciousness_level": 0}  // Faster
   ```

2. **Reduce max_history:**
   ```json
   {"max_history": 5}  // Instead of 20
   ```

3. **Check Ollama resources:**
   - CPU: Ollama uses 100% of one core by default
   - RAM: 20B model requires ~12GB
   - GPU: Use CUDA if available

4. **Optimize the database:**
   ```bash
   curl -X POST http://localhost:8000/admin/vacuum
   ```

### Semantic Memory Not Working

**Symptoms:**
- `memory_echoes: []` even with level 3

**Possible Causes:**

1. **Similarity too low (< 0.7)**
   - Messages too different
   - Solution: Rephrase to be more explicit

2. **Too recent (temporal decay)**
   - Wait a few turns
   - Solution: Test with >5 messages apart

3. **Empty session**
   - First use of the session_id
   - Solution: First accumulate messages

4. **Server restarted**
   - RAM memory lost
   - Solution: Coming soon (SQLite persistence)

---

## FAQ

### Q: What is the difference between τ_c, ρ, δ_r, and κ?

**Answer:**

- **τ_c (tau_c)**: Tension/temperature
  - Controls creativity (high) vs determinism (low)
  - Mapped to Ollama temperature: [0.1, 1.5]

- **ρ (rho)**: Focus/polarity
  - Controls repetition vs diversity
  - Mapped to presence_penalty and frequency_penalty

- **δ_r (delta_r)**: Planning/scheduling
  - Controls injected context density
  - Influences the number of semantic neighbors

- **κ (kappa)**: Curvature/style (optional)
  - Generates style hints in the prompt
  - Ex: "Be concise" or "Elaborate deeply"

### Q: Can I use another LLM besides Ollama?

**Answer:**
Currently, only Ollama is supported. To add another backend:

1. Create a new client in `app/` (e.g., `openai_client.py`)
2. Implement the same interface as `OllamaClient`
3. Modify `app/main.py` to inject the new client

Contributions welcome!

### Q: How do I export a conversation?

**Answer:**
Use the history endpoint:

```bash
curl http://localhost:8000/sessions/{session_id}/history > conversation.json
```

Or directly from SQLite:

```bash
sqlite3 data/ispace.db "SELECT * FROM events WHERE session_id='...' ORDER BY timestamp"
```

### Q: Is semantic memory persistent?

**Answer:**
⚠️ No, currently it's RAM-only. Restarting the server clears the memory.

**Workaround:**
- Use `consciousness_level: 0-2` for conversations not requiring memory
- Keep-alive the server in production

**Roadmap:**
- Phase 4 will add SQLite persistence for memory

### Q: How many tokens maximum per request?

**Answer:**
Configured to **4096 tokens** by default in `app/llm_client.py:180`.

To modify:
```python
# app/llm_client.py
"options": {
    "num_predict": 8192  # Increase if your model supports it
}
```

⚠️ Check your Ollama model's limits.

### Q: Can I use Lyra without a knowledge graph?

**Answer:**
Yes! Disable context injection in `config.yaml`:

```yaml
context:
  enabled: false
```

Lyra will work as a standard LLM with session management.

### Q: How do I add concepts to the semantic graph?

**Answer:**

**Option 1: Direct SQL**
```sql
INSERT INTO concepts (concept, embedding)
VALUES ('new_concept', NULL);  -- Embedding optional

INSERT INTO semantic_relations (source, target, weight)
VALUES ('concept_a', 'new_concept', 0.8);
```

**Option 2: Python Script**
```python
from database.engine import ISpaceDB
import asyncio

async def add_concept():
    db = ISpaceDB('data/ispace.db')
    # Use db engine methods
    # (to implement according to your needs)
```

**Option 3: Import from File**
See `scripts/build_global_map.py` for a batch import example.

### Q: Are Bézier curves modifiable in real-time?

**Answer:**
No, a Bézier profile is fixed for the entire duration of a session.

**Workaround:**
- Create a new session with another profile
- Or implement modification via SQL:
  ```sql
  UPDATE sessions SET profile_name='aggressive' WHERE session_id='...';
  ```
  ⚠️ This may create discontinuities in trajectories

### Q: What is typical latency?

**Answer:**

| Configuration | Average Latency |
|---------------|-----------------|
| Level 0, no context | 1.0-1.5s |
| Level 1, with context | 1.3-1.8s |
| Level 2, adaptive | 1.4-2.0s |
| Level 3, with memory | 1.5-2.2s |

**Factors:**
- LLM model size
- CPU vs GPU (Ollama)
- History length
- Semantic graph complexity

### Q: Can I host Lyra in production?

**Answer:**
Yes, but consider these points:

**To do before production:**
1. ✅ Restrict CORS in `config.yaml`
   ```yaml
   cors:
     origins:
       - "https://yourdomain.com"
   ```

2. ✅ Enable API authentication
   ```yaml
   security:
     api_key_enabled: true
   ```

3. ✅ Configure rate limiting
   ```yaml
   security:
     rate_limit_per_minute: 60
   ```

4. ✅ Use a reverse proxy (nginx, Caddy)
5. ✅ Configure HTTPS
6. ✅ Monitoring (logs, metrics)
7. ✅ Automatic DB backups

**Recommended Docker deployment:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Q: How do I contribute to the project?

**Answer:**
See the [Developer Guide](DEVELOPER_GUIDE.md#contribution)!

Summary:
1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add: my new feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## Support

- 📖 Complete documentation: [docs/en/](.)
- 🐛 Report a bug: [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/lyra_clean_bis/discussions)
- 📧 Email: support@example.com

---

**Next step:** See the [Developer Guide](DEVELOPER_GUIDE.md) to contribute or customize Lyra.
