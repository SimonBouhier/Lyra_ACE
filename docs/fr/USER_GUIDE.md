# Guide Utilisateur Lyra Clean

## Table des matières

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Démarrage rapide](#démarrage-rapide)
4. [Utilisation de l'API](#utilisation-de-lapi)
5. [Niveaux de conscience](#niveaux-de-conscience)
6. [Profils Bézier](#profils-bézier)
7. [Gestion des sessions](#gestion-des-sessions)
8. [Interface Web](#interface-web)
9. [Dépannage](#dépannage)
10. [FAQ](#faq)

---

## Introduction

Lyra Clean est un système conversationnel LLM qui contrôle la génération de texte via des trajectoires mathématiques (courbes de Bézier) plutôt que des réglages statiques. Cela permet un comportement dynamique et prévisible tout au long d'une conversation.

### Pourquoi Lyra Clean ?

**Problème traditionnel :**
- Paramètres LLM statiques (température fixe = 0.7)
- Comportement imprévisible dans les longues conversations
- Ajustements réactifs difficiles à calibrer

**Solution Lyra :**
- Trajectoires Bézier définissant l'évolution des paramètres
- Comportement balistique prévisible (comme une trajectoire physique)
- Trois niveaux de conscience pour adaptation contextuelle

### Concepts clés

- **Physique déterministe** : Les paramètres évoluent selon des courbes mathématiques
- **Conscience épistemologique** : Le système s'observe et s'adapte
- **Contexte sémantique** : Injection intelligente de connaissances depuis un graphe
- **Mémoire sémantique** : Rappel d'anciennes conversations par similarité

---

## Installation

### Prérequis

1. **Python 3.10 ou supérieur**
   ```bash
   python --version  # Doit afficher Python 3.10.x ou plus
   ```

2. **Ollama installé et en cours d'exécution**
   - Téléchargez depuis [ollama.ai](https://ollama.ai/)
   - Installez et lancez le service
   - Vérifiez : `ollama list`

3. **Modèle LLM disponible**
   ```bash
   ollama pull gpt-oss:20b
   # Ou utilisez un autre modèle et modifiez config.yaml
   ```

### Installation standard

```bash
# 1. Cloner le dépôt
git clone https:/SimonBouhier/github.com//lyra_clean_bis.git
cd lyra_clean_bis

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Initialiser la base de données
python -c "from database.engine import ISpaceDB; import asyncio; asyncio.run(ISpaceDB('data/ispace.db').initialize())"
```

### Installation Docker (alternative)

```bash
# Construire et lancer
docker-compose up --build

# En arrière-plan
docker-compose up -d
```

---

## Démarrage rapide

### Lancer le serveur

**Option 1 : Script automatique (Windows)**
```bash
start_server.bat
```

**Option 2 : Manuel**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3 : Docker**
```bash
docker-compose up
```

Le serveur démarre sur : **http://localhost:8000**

### Vérifier le statut

```bash
# Health check
curl http://localhost:8000/health

# Réponse attendue :
{
  "status": "healthy",
  "database": {"connected": true, "concepts": 1234},
  "ollama": {"connected": true, "model": "gpt-oss:20b"}
}
```

### Première conversation

```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Bonjour Lyra, qui es-tu ?",
    "consciousness_level": 0
  }'
```

**Réponse :**
```json
{
  "response": "Je suis Lyra Clean, un système conversationnel...",
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

## Utilisation de l'API

### Endpoint principal : Chat

**POST /chat/message**

Envoie un message et reçoit une réponse.

#### Paramètres de requête

```json
{
  "message": "Votre message ici",
  "session_id": "abc123...",           // Optionnel : réutiliser une session
  "consciousness_level": 2,            // 0-3 (défaut: 0)
  "profile_name": "balanced",          // Profil Bézier (défaut: "balanced")
  "max_history": 10,                   // Messages de contexte (défaut: 20)
  "max_context_length": 200            // Taille max contexte sémantique
}
```

#### Champs obligatoires
- `message` : Le message utilisateur (string, max 10000 caractères)

#### Champs optionnels
- `session_id` : ID de session existante (auto-généré si omis)
- `consciousness_level` : Niveau d'introspection (0-3, défaut 0)
- `profile_name` : Nom du profil Bézier à utiliser
- `max_history` : Nombre de messages précédents à inclure
- `max_context_length` : Taille maximale du contexte sémantique injecté

#### Réponse

```json
{
  "response": "Réponse générée",
  "session_id": "abc123...",
  "message_index": 5,
  "physics_state": {
    "t": 0.05,                // Temps normalisé [0, 1]
    "tau_c": 0.52,            // Tension/température
    "rho": 0.48,              // Focus/polarité
    "delta_r": 0.35,          // Planification
    "kappa": 0.6              // Courbure/style (optionnel)
  },
  "consciousness": {          // Si consciousness_level >= 1
    "coherence": 0.82,
    "tension": 0.45,
    "fit": 0.91,
    "pressure": 0.38,
    "stability_score": 0.87,
    "suggestion": null        // Si niveau 2, contient ajustements suggérés
  },
  "memory_echoes": [          // Si consciousness_level >= 3
    {
      "content": "Message rappelé du passé",
      "similarity": 0.89,
      "turns_ago": 12
    }
  ],
  "semantic_context": [       // Contexte injecté depuis le graphe
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

### Exemples pratiques

#### Conversation simple (niveau 0)

```python
import requests

response = requests.post("http://localhost:8000/chat/message", json={
    "message": "Explique-moi la relativité restreinte",
    "consciousness_level": 0
})

print(response.json()["response"])
```

#### Conversation avec mémoire (niveau 3)

```python
# Premier message
r1 = requests.post("http://localhost:8000/chat/message", json={
    "message": "Je m'appelle Alice et j'aime la physique quantique",
    "consciousness_level": 3
})
session_id = r1.json()["session_id"]

# Plus tard dans la conversation...
r2 = requests.post("http://localhost:8000/chat/message", json={
    "message": "Quel est mon nom et mes intérêts ?",
    "session_id": session_id,
    "consciousness_level": 3
})

# Lyra devrait se souvenir grâce à la mémoire sémantique
print(r2.json()["response"])
# Devrait mentionner "Alice" et "physique quantique"

# Vérifier les memory echoes
print(r2.json()["memory_echoes"])
```

#### Profil agressif pour brainstorming

```python
response = requests.post("http://localhost:8000/chat/message", json={
    "message": "Propose 10 idées originales pour un roman de SF",
    "profile_name": "aggressive",  # Haute température, exploratoire
    "consciousness_level": 2       # Adaptatif
})
```

---

## Niveaux de conscience

Lyra possède 4 niveaux de conscience (0-3) qui déterminent son degré d'introspection et d'adaptation.

### Niveau 0 : Passif

**Comportement :**
- Aucune introspection
- Génération standard uniquement
- Performance maximale (aucun calcul supplémentaire)

**Utilisation :**
- Conversations simples
- Requêtes factuelles
- Performance critique

**Exemple :**
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Quelle est la capitale de la France ?", "consciousness_level": 0}'
```

### Niveau 1 : Observateur

**Comportement :**
- Calcul de métriques épistemologiques
- Aucune action, monitoring uniquement
- Retourne métriques dans la réponse

**Métriques calculées :**
- **Coherence** (0-1) : Densité sémantique du contexte injecté
- **Tension** (0-1) : Stress système (température × longueur réponse)
- **Fit** (0-1) : Alignement longueur attendue/réelle
- **Pressure** (0-1) : Exploration vs exploitation
- **Stability Score** (0-1) : Score composite de stabilité

**Utilisation :**
- Debugging de comportement
- Analyse de performance
- Recherche sur la conscience artificielle

**Exemple :**
```json
{
  "message": "Parle-moi de mécanique quantique",
  "consciousness_level": 1
}

// Réponse inclut :
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

### Niveau 2 : Adaptatif

**Comportement :**
- Hérite du niveau 1 (métriques)
- **Applique automatiquement** les ajustements aux paramètres Bézier (τ_c, ρ, δ_r)
- Boucle de feedback : métriques de l'interaction N-1 adaptent l'interaction N
- Modifications graduelles (5-7.5% par tour)

**Règles d'adaptation :**

1. **Tension élevée (> 0.75)**
   - Réduit τ_c de 5% (diminue température)
   - Raison : Stabiliser le système

2. **Cohérence faible (< 0.3)**
   - Ajuste ρ vers focus
   - Raison : Améliorer pertinence contextuelle

3. **Fit élevé (> 0.8) + Stabilité (> 0.7)**
   - Encourage exploration (augmente τ_c)
   - Raison : Éviter sur-optimisation

4. **Pression élevée (> 0.85)**
   - Réduit τ_c de 7.5% et δ_r de 5%
   - Raison : Alléger charge système

5. **Session longue (> 30 messages) + tension stable**
   - Aucun changement
   - Raison : Convergence atteinte

**Utilisation :**
- Conversations longues et complexes
- Auto-ajustement en temps réel
- Optimisation automatique

**Exemple :**
```json
{
  "message": "Continue la discussion",
  "consciousness_level": 2
}

// Réponse peut inclure :
{
  "consciousness": {
    "suggestion": {
      "reason": "High tension detected",
      "adjustments": {
        "tau_c": -0.05  // Réduit de 5%
      }
    }
  }
}
```

### Niveau 3 : Mémoire sémantique

**Comportement :**
- Hérite du niveau 2 (métriques + adaptation)
- Enregistre chaque message avec embeddings (1024D)
- Rappelle messages similaires par cosine similarity
- Applique décroissance temporelle : `max(0.5, 1.0 - turns_ago * 0.01)`

**Fonctionnement :**

1. **Enregistrement :**
   - Chaque message → embeddings mxbai-embed-large
   - Stockage en mémoire (dict : session_id → entries)
   - Limite : 50 entrées par session

2. **Rappel :**
   - Calcul similarité cosinus avec message actuel
   - Seuil : 0.7 minimum
   - Limite : 3 meilleurs matches
   - Décroissance : -1% par tour écoulé

3. **Injection :**
   - Ajouté au contexte système comme `[MEMORY ECHO]`
   - Format : contenu + metadata (similarité, ancienneté)

**Utilisation :**
- Conversations multi-tours avec continuité
- Questions de suivi sur sujets passés
- Contexte personnel maintenu

**Exemple :**
```python
# Message initial
r1 = requests.post("http://localhost:8000/chat/message", json={
    "message": "Mon chien s'appelle Rex et il adore jouer au frisbee",
    "consciousness_level": 3
})
session_id = r1.json()["session_id"]

# 20 messages plus tard...
r2 = requests.post("http://localhost:8000/chat/message", json={
    "message": "Comment s'appelle mon chien déjà ?",
    "session_id": session_id,
    "consciousness_level": 3
})

# Vérifie memory echoes
echoes = r2.json()["memory_echoes"]
# [{"content": "Mon chien s'appelle Rex...", "similarity": 0.91, "turns_ago": 20}]
```

**Limitations :**
- ⚠️ Mémoire en RAM uniquement (perdue au redémarrage serveur)
- ⚠️ Limite 50 messages par session
- ⚠️ Coût : +100ms par requête (génération embeddings)

### Comparaison des niveaux

| Niveau | Métriques | Adaptation | Mémoire | Latence | Usage |
|--------|-----------|------------|---------|---------|-------|
| 0 | ❌ | ❌ | ❌ | ~1.2s | Production rapide |
| 1 | ✅ | ❌ | ❌ | ~1.3s | Monitoring/debug |
| 2 | ✅ | ✅ | ❌ | ~1.4s | Conversations complexes |
| 3 | ✅ | ✅ | ✅ | ~1.5s | Mémoire long terme |

---

## Profils Bézier

Les profils définissent comment les paramètres évoluent au cours d'une conversation via des courbes de Bézier cubiques.

### Profils disponibles

#### 1. Balanced (défaut)

**Caractéristiques :**
- Température stable autour de 0.7
- Pas de dérive extrême
- Bon pour usage général

**Courbes :**
```yaml
tau_c:  [0, 0.5] → [0.5, 0.45] → [0.55, 0.55] → [1, 0.5]
rho:    [0, 0.5] → [0.33, 0.5] → [0.67, 0.5] → [1, 0.5]
delta_r:[0, 0.3] → [0.33, 0.35] → [0.67, 0.35] → [1, 0.3]
```

**Usage :**
```json
{"message": "...", "profile_name": "balanced"}
```

#### 2. Aggressive

**Caractéristiques :**
- Haute température initiale (0.8-1.0)
- Exploratoire, créatif
- Bon pour brainstorming

**Usage :**
- Génération d'idées
- Écriture créative
- Exploration de concepts

**Exemple :**
```json
{
  "message": "Invente 5 créatures fantastiques originales",
  "profile_name": "aggressive"
}
```

#### 3. Conservative

**Caractéristiques :**
- Basse température (0.3-0.5)
- Précis, factuel
- Bon pour tâches analytiques

**Usage :**
- Code generation
- Résumés factuels
- Calculs mathématiques

**Exemple :**
```json
{
  "message": "Écris une fonction Python pour trier une liste",
  "profile_name": "conservative"
}
```

### Créer un profil personnalisé

Les profils sont définis dans la base de données. Vous pouvez en créer via SQL :

```sql
INSERT INTO bezier_profiles (name, description, tau_c_json, rho_json, delta_r_json)
VALUES (
  'custom_profile',
  'Mon profil personnalisé',
  '[{"t": 0.0, "value": 0.6}, {"t": 0.33, "value": 0.7}, {"t": 0.67, "value": 0.5}, {"t": 1.0, "value": 0.4}]',
  '[{"t": 0.0, "value": 0.5}, {"t": 0.33, "value": 0.5}, {"t": 0.67, "value": 0.5}, {"t": 1.0, "value": 0.5}]',
  '[{"t": 0.0, "value": 0.3}, {"t": 0.33, "value": 0.4}, {"t": 0.67, "value": 0.3}, {"t": 1.0, "value": 0.2}]'
);
```

**Contraintes :**
- Exactement 4 points de contrôle (t=0, t≈0.33, t≈0.67, t=1)
- `t` doit être strictement croissant
- `value` ∈ [0, 1]

### Lister les profils

```bash
curl http://localhost:8000/profiles

# Réponse :
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

### Prévisualiser un profil

```bash
curl "http://localhost:8000/profiles/balanced?preview=20"

# Retourne 20 points échantillonnés de la trajectoire
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

## Gestion des sessions

### Créer une session

**POST /sessions**

```json
{
  "profile_name": "balanced",
  "max_messages": 100,
  "time_mapping": "logarithmic"
}
```

**Réponse :**
```json
{
  "session_id": "abc123...",
  "created_at": "2025-01-14T10:30:00Z",
  "profile_name": "balanced"
}
```

### Récupérer une session

**GET /sessions/{session_id}**

```bash
curl http://localhost:8000/sessions/abc123...
```

**Réponse :**
```json
{
  "session_id": "abc123...",
  "created_at": "2025-01-14T10:30:00Z",
  "message_count": 15,
  "profile_name": "balanced",
  "time_mapping": "logarithmic"
}
```

### Historique de conversation

**GET /sessions/{session_id}/history**

```bash
curl http://localhost:8000/sessions/abc123.../history
```

**Réponse :**
```json
{
  "session_id": "abc123...",
  "messages": [
    {
      "role": "user",
      "content": "Bonjour",
      "timestamp": "2025-01-14T10:30:15Z",
      "message_index": 1
    },
    {
      "role": "assistant",
      "content": "Bonjour ! Comment puis-je vous aider ?",
      "timestamp": "2025-01-14T10:30:17Z",
      "message_index": 2,
      "physics_state": {"t": 0.01, "tau_c": 0.50, ...}
    },
    ...
  ],
  "total_messages": 15
}
```

### Supprimer une session

**DELETE /sessions/{session_id}**

```bash
curl -X DELETE http://localhost:8000/sessions/abc123...
```

**Réponse :**
```json
{
  "success": true,
  "message": "Session deleted"
}
```

---

## Interface Web

Lyra inclut une interface web minimaliste pour tests rapides.

### Accès

Ouvrez votre navigateur : **http://localhost:8000**

### Fonctionnalités

- ✅ Envoi de messages en temps réel
- ✅ Sélection du niveau de conscience (0-3)
- ✅ Sélection du profil Bézier
- ✅ Affichage de l'historique de conversation
- ✅ Métriques de conscience (si niveau ≥ 1)
- ✅ Memory echoes (si niveau 3)
- ✅ Visualisation de l'état physique

### Fichier source

L'interface est un fichier HTML statique : `app/static/index.html`

Vous pouvez la personnaliser selon vos besoins.

---

## Dépannage

### Erreur : "Ollama server not reachable"

**Symptômes :**
```json
{
  "error": "Ollama request failed after 3 attempts"
}
```

**Solutions :**
1. Vérifiez qu'Ollama est lancé :
   ```bash
   ollama list
   ```

2. Vérifiez l'URL dans `config.yaml` :
   ```yaml
   llm:
     base_url: "http://localhost:11434"  # Port par défaut
   ```

3. Testez manuellement :
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Erreur : "Model not found"

**Symptômes :**
```json
{
  "error": "HTTP 404: model 'gpt-oss:20b' not found"
}
```

**Solutions :**
1. Téléchargez le modèle :
   ```bash
   ollama pull gpt-oss:20b
   ```

2. Ou modifiez `config.yaml` pour utiliser un modèle disponible :
   ```yaml
   llm:
     model: "llama3:latest"  # Ou autre modèle installé
   ```

### Erreur : "Database locked"

**Symptômes :**
```
sqlite3.OperationalError: database is locked
```

**Solutions :**
1. Vérifiez qu'aucun autre processus n'utilise la DB :
   ```bash
   lsof data/ispace.db  # Linux/Mac
   ```

2. Activez le mode WAL (déjà fait par défaut) :
   ```sql
   PRAGMA journal_mode=WAL;
   ```

3. Redémarrez le serveur.

### Performance lente

**Symptômes :**
- Latence > 5 secondes par requête

**Solutions :**

1. **Désactivez la conscience si inutile :**
   ```json
   {"consciousness_level": 0}  // Plus rapide
   ```

2. **Réduisez max_history :**
   ```json
   {"max_history": 5}  // Au lieu de 20
   ```

3. **Vérifiez les ressources Ollama :**
   - CPU : Ollama utilise 100% d'un core par défaut
   - RAM : Modèle 20B nécessite ~12GB
   - GPU : Utilisez CUDA si disponible

4. **Optimisez la base de données :**
   ```bash
   curl -X POST http://localhost:8000/admin/vacuum
   ```

### Mémoire sémantique ne fonctionne pas

**Symptômes :**
- `memory_echoes: []` même avec niveau 3

**Causes possibles :**

1. **Similarité trop faible (< 0.7)**
   - Messages trop différents
   - Solution : Reformulez pour être plus explicite

2. **Trop récent (décroissance temporelle)**
   - Attendez quelques tours
   - Solution : Testez avec >5 messages d'écart

3. **Session vide**
   - Première utilisation du session_id
   - Solution : Accumulez d'abord des messages

4. **Serveur redémarré**
   - Mémoire en RAM perdue
   - Solution : À venir (persistance SQLite)

---

## FAQ

### Q : Quelle est la différence entre τ_c, ρ, δ_r et κ ?

**Réponse :**

- **τ_c (tau_c)** : Tension/température
  - Contrôle la créativité (haute) vs déterminisme (basse)
  - Mappé vers temperature Ollama : [0.1, 1.5]

- **ρ (rho)** : Focus/polarité
  - Contrôle répétition vs diversité
  - Mappé vers presence_penalty et frequency_penalty

- **δ_r (delta_r)** : Planification/scheduling
  - Contrôle la densité de contexte injectée
  - Influence le nombre de voisins sémantiques

- **κ (kappa)** : Courbure/style (optionnel)
  - Génère des hints de style dans le prompt
  - Ex : "Be concise" ou "Elaborate deeply"

### Q : Puis-je utiliser un autre LLM qu'Ollama ?

**Réponse :**
Actuellement, seul Ollama est supporté. Pour ajouter un autre backend :

1. Créez un nouveau client dans `app/` (ex : `openai_client.py`)
2. Implémentez la même interface que `OllamaClient`
3. Modifiez `app/main.py` pour injecter le nouveau client

Contribution welcome ! 🚀

### Q : Comment exporter une conversation ?

**Réponse :**
Utilisez l'endpoint d'historique :

```bash
curl http://localhost:8000/sessions/{session_id}/history > conversation.json
```

Ou directement depuis SQLite :

```bash
sqlite3 data/ispace.db "SELECT * FROM events WHERE session_id='...' ORDER BY timestamp"
```

### Q : La mémoire sémantique est-elle persistante ?

**Réponse :**
⚠️ Non, actuellement elle est en RAM uniquement. Redémarrer le serveur efface la mémoire.

**Workaround :**
- Utilisez `consciousness_level: 0-2` pour conversations ne nécessitant pas de mémoire
- Keep-alive le serveur en production

**Roadmap :**
- Phase 4 ajoutera la persistance SQLite pour la mémoire

### Q : Combien de tokens maximum par requête ?

**Réponse :**
Configuré à **4096 tokens** par défaut dans `app/llm_client.py:180`.

Pour modifier :
```python
# app/llm_client.py
"options": {
    "num_predict": 8192  # Augmentez si votre modèle le supporte
}
```

⚠️ Vérifiez les limites de votre modèle Ollama.

### Q : Puis-je utiliser Lyra sans graphe de connaissances ?

**Réponse :**
Oui ! Désactivez l'injection de contexte dans `config.yaml` :

```yaml
context:
  enabled: false
```

Lyra fonctionnera comme un LLM standard avec gestion de sessions.

### Q : Comment ajouter des concepts au graphe sémantique ?

**Réponse :**

**Option 1 : SQL direct**
```sql
INSERT INTO concepts (concept, embedding)
VALUES ('nouveau_concept', NULL);  -- Embedding optionnel

INSERT INTO semantic_relations (source, target, weight)
VALUES ('concept_a', 'nouveau_concept', 0.8);
```

**Option 2 : Script Python**
```python
from database.engine import ISpaceDB
import asyncio

async def add_concept():
    db = ISpaceDB('data/ispace.db')
    # Utilisez les méthodes du db engine
    # (à implémenter selon vos besoins)
```

**Option 3 : Import depuis fichier**
Consultez `scripts/build_global_map.py` pour un exemple d'import batch.

### Q : Les courbes de Bézier sont-elles modifiables en temps réel ?

**Réponse :**
Non, un profil Bézier est fixé pour toute la durée d'une session.

**Workaround :**
- Créez une nouvelle session avec un autre profil
- Ou implémentez la modification via SQL :
  ```sql
  UPDATE sessions SET profile_name='aggressive' WHERE session_id='...';
  ```
  ⚠️ Cela peut créer des discontinuités dans les trajectoires

### Q : Quelle est la latence typique ?

**Réponse :**

| Configuration | Latence moyenne |
|---------------|-----------------|
| Niveau 0, pas de contexte | 1.0-1.5s |
| Niveau 1, avec contexte | 1.3-1.8s |
| Niveau 2, adaptatif | 1.4-2.0s |
| Niveau 3, avec mémoire | 1.5-2.2s |

**Facteurs :**
- Taille du modèle LLM
- CPU vs GPU (Ollama)
- Longueur de l'historique
- Complexité du graphe sémantique

### Q : Puis-je héberger Lyra en production ?

**Réponse :**
Oui, mais considérez ces points :

**À faire avant production :**
1. ✅ Restreindre CORS dans `config.yaml`
   ```yaml
   cors:
     origins:
       - "https://yourdomain.com"
   ```

2. ✅ Activer l'authentification API
   ```yaml
   security:
     api_key_enabled: true
   ```

3. ✅ Configurer le rate limiting
   ```yaml
   security:
     rate_limit_per_minute: 60
   ```

4. ✅ Utiliser un reverse proxy (nginx, Caddy)
5. ✅ Configurer HTTPS
6. ✅ Monitoring (logs, métriques)
7. ✅ Backups automatiques de la DB

**Déploiement Docker recommandé :**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Q : Comment contribuer au projet ?

**Réponse :**
Consultez le [Developer Guide](DEVELOPER_GUIDE.md#contribution) !

Résumé :
1. Fork le dépôt
2. Créez une branche : `git checkout -b feature/ma-feature`
3. Commitez : `git commit -m "Add: ma nouvelle feature"`
4. Push : `git push origin feature/ma-feature`
5. Ouvrez une Pull Request

---

## Support

- 📖 Documentation complète : [docs/fr/](.)
- 🐛 Rapporter un bug : [GitHub Issues](https://github.com/yourusername/lyra_clean_bis/issues)
- 💬 Discussions : [GitHub Discussions](https://github.com/yourusername/lyra_clean_bis/discussions)
- 📧 Email : support@example.com

---

**Prochaine étape :** Consultez le [Developer Guide](DEVELOPER_GUIDE.md) pour contribuer ou personnaliser Lyra.

