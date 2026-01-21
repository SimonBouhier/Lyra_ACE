# Lyra Clean Documentation

Complete documentation for the Lyra Clean framework.

## Languages / Langues

- 🇬🇧 **[English Documentation](en/)** - Full English documentation
- 🇫🇷 **[Documentation française](fr/)** - Documentation complète en français

---

## 🇬🇧 English Documentation

Complete guides for users and developers:

### For Users
- **[User Guide](en/USER_GUIDE.md)** - Installation, quick start, API usage, FAQ
- **[API Reference](en/API_REFERENCE.md)** - Complete REST API documentation
- **[Configuration](en/CONFIGURATION.md)** - System configuration guide

### For Developers
- **[Developer Guide](en/DEVELOPER_GUIDE.md)** - Architecture, components, contribution

---

## 🇫🇷 Documentation française

Guides complets pour utilisateurs et développeurs :

### Pour les utilisateurs
- **[Guide utilisateur](fr/USER_GUIDE.md)** - Installation, démarrage, utilisation API, FAQ
- **[Référence API](fr/API_REFERENCE.md)** - Documentation complète REST API
- **[Configuration](fr/CONFIGURATION.md)** - Guide de configuration système

### Pour les développeurs
- **[Guide développeur](fr/DEVELOPER_GUIDE.md)** - Architecture, composants, contribution

---

## Quick Links

| Topic | English | Français |
|-------|---------|----------|
| **Getting Started** | [Quick Start →](en/USER_GUIDE.md#quick-start) | [Démarrage rapide →](fr/USER_GUIDE.md#démarrage-rapide) |
| **API Endpoints** | [Endpoints →](en/API_REFERENCE.md#endpoints) | [Endpoints →](fr/API_REFERENCE.md#chat) |
| **Lyra-ACE** | Graph mutations & Multi-model | Mutations graphe & Multi-modèles |
| **Architecture** | [Overview →](en/DEVELOPER_GUIDE.md#architecture) | [Vue d'ensemble →](fr/DEVELOPER_GUIDE.md#architecture) |
| **Configuration** | [Config Reference →](en/CONFIGURATION.md) | [Référence config →](fr/CONFIGURATION.md) |
| **Contributing** | [How to Contribute →](en/DEVELOPER_GUIDE.md#contributing) | [Contribuer →](fr/DEVELOPER_GUIDE.md#contribution) |

---

## Documentation Structure

```
docs/
├── README.md                   # This file (navigation)
│
├── en/                         # English documentation
│   ├── USER_GUIDE.md          # User guide
│   ├── DEVELOPER_GUIDE.md     # Developer guide
│   ├── API_REFERENCE.md       # API reference
│   └── CONFIGURATION.md       # Configuration guide
│
├── fr/                         # French documentation
│   ├── USER_GUIDE.md          # Guide utilisateur
│   ├── DEVELOPER_GUIDE.md     # Guide développeur
│   ├── API_REFERENCE.md       # Référence API
│   └── CONFIGURATION.md       # Configuration
│
└── Instructions_pour_Lyra_ACE.py  # Lyra-ACE implementation spec
```

**Total documentation**: 8 comprehensive markdown files + ACE spec

---

## Key Features Documented

✅ **Physics-driven LLM system** - Bézier trajectory control
✅ **3 consciousness levels** - Passive → Adaptive → Memory
✅ **Semantic context injection** - Knowledge graph integration
✅ **Session management** - Persistent conversations with export/import
✅ **REST API** - Complete endpoint documentation
✅ **Configuration** - Full system customization
✅ **Contributing** - Developer workflow and architecture

### Lyra-ACE Features (New)

#### Graph Intelligence

✅ **Dynamic graph mutations** - Auditable deltas with rollback capability
✅ **Hybrid κ curvature** - Ollivier + Jaccard structural analysis
✅ **Entity deduplication** - Semantic resolution via embeddings (threshold 0.92)
✅ **Relation canonicalization** - 20 canonical forms with inverse tracking

#### Multi-Model Support

✅ **Multi-model generation** - Sequential LLM comparison with consensus
✅ **Best response selection** - Automatic selection based on model weights
✅ **Consensus metrics** - Length variance, latency, success rate

#### New API Endpoints

✅ **`/graph/delta`** - Apply atomic graph mutations
✅ **`/graph/kappa/{source}/{target}`** - Compute hybrid curvature
✅ **`/graph/deltas`** - Query mutation history
✅ **`/graph/rollback`** - Restore previous graph state
✅ **`/graph/stats`** - Mutation statistics
✅ **`/multimodel/models`** - List available LLMs
✅ **`/multimodel/generate`** - Multi-model generation with consensus

#### Performance & Security

✅ **Connection pooling** - SQLite pool with overflow management
✅ **Concept caching** - TTL LRU cache (1000 entries, 1h TTL)
✅ **SQL validation** - Injection prevention
✅ **Secrets management** - Environment-based API key handling
✅ **Session storage** - Export/import sessions to JSON files

---

## Support

- 📖 Documentation: You're here!
- 🐛 Report bugs: [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/lyra_clean_bis/discussions)
- 📧 Email: support@example.com

---

**License**: MIT - see [LICENSE](../LICENSE)
