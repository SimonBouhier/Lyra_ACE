"""
Consciousness Memory - Niveau 3 (Sophisti qué)

Implémente mémoire sémantique avec decay temporel et rappel vectoriel.

Architecture:
- MemoryEntry: message + embeddings (1024D) + timestamp
- Decay formula: max(0.5, 1.0 - turns_ago * 0.01)
- Recall: cosine similarity filtered par decay
- Injection: [MEMORY ECHO] format dans prompt
"""

from typing import Optional, List, Dict
import time
import math
from dataclasses import dataclass, asdict
from .adaptation import AdaptiveConsciousness


@dataclass
class MemoryEntry:
    """Une entrée en mémoire sémantique"""
    message_id: str          # Identifiant unique
    session_id: str          # Session d'origine
    content: str             # Contenu du message
    embeddings: List[float]  # 1024D embeddings
    timestamp: float         # Epoch timestamp
    turn_number: int         # Numéro du tour dans la session
    
    def to_dict(self):
        """Conversion dict pour sérialisation"""
        return asdict(self)


class SemanticMemory(AdaptiveConsciousness):
    """
    Extension AdaptiveConsciousness (Niveau 3) avec mémoire sémantique.
    
    Stocke messages avec embeddings, rappelle via similarité cosinus,
    applique decay temporel, injuste en contexte.
    """
    
    def __init__(
        self,
        level: int = 3,
        adaptation_rate: float = 0.05,
        max_memory_entries: int = 50,
        decay_rate: float = 0.01,
        similarity_threshold: float = 0.6
    ):
        """
        Initialise mémoire sémantique.
        
        Args:
            level: Niveau conscience (3 pour mémoire)
            adaptation_rate: Taux adaptation (hérité)
            max_memory_entries: Nombre max d'entrées (~1MB par 50 entrées)
            decay_rate: Taux decay par tour (0.01 = 1% par tour)
            similarity_threshold: Seuil cosine similarity pour rappel
        """
        super().__init__(level, adaptation_rate)
        self.max_memory_entries = max_memory_entries
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.memory: Dict[str, List[MemoryEntry]] = {}  # session_id -> entries
    
    def store_memory(
        self,
        session_id: str,
        content: str,
        embeddings: List[float],
        turn_number: int
    ) -> Optional[MemoryEntry]:
        """
        Stocke un message en mémoire.
        
        Args:
            session_id: Identifiant session
            content: Contenu du message
            embeddings: Vecteur 1024D
            turn_number: Numéro du tour
        
        Returns:
            MemoryEntry créée ou None si problème
        """
        if self.level < 3:
            return None
        
        if not content or not embeddings or len(embeddings) != 1024:
            return None
        
        # Créer entrée
        message_id = f"{session_id}_{turn_number}_{int(time.time()*1000)}"
        entry = MemoryEntry(
            message_id=message_id,
            session_id=session_id,
            content=content,
            embeddings=embeddings,
            timestamp=time.time(),
            turn_number=turn_number
        )
        
        # Initialiser session si nécessaire
        if session_id not in self.memory:
            self.memory[session_id] = []
        
        # Ajouter
        self.memory[session_id].append(entry)
        
        # Nettoyer si dépasse max
        if len(self.memory[session_id]) > self.max_memory_entries:
            # Garder les N plus récentes (FIFO)
            self.memory[session_id] = self.memory[session_id][-self.max_memory_entries:]
        
        return entry
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calcule similarité cosinus entre deux vecteurs.
        
        Args:
            vec1: Vecteur 1
            vec2: Vecteur 2
        
        Returns:
            Similarité dans [0, 1]
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Produit scalaire
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Normes
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))  # Clamp [0,1]
    
    def _compute_decay(self, turns_ago: int) -> float:
        """
        Calcule facteur de decay temporel.
        
        Formule : max(0.5, 1.0 - turns_ago * decay_rate)
        
        Args:
            turns_ago: Nombre de tours écoulés
        
        Returns:
            Facteur decay dans [0.5, 1.0]
        """
        decay = 1.0 - (turns_ago * self.decay_rate)
        return max(0.5, min(1.0, decay))
    
    def recall_memory(
        self,
        session_id: str,
        query_embeddings: List[float],
        current_turn: int,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Rappelle messages similaires avec decay temporel.
        
        Args:
            session_id: Session à chercher
            query_embeddings: Vecteur 1024D de la requête
            current_turn: Numéro du tour actuel
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste de dicts {content, similarity, decay, final_score}
        """
        if self.level < 3:
            return []
        
        if session_id not in self.memory:
            return []
        
        entries = self.memory[session_id]
        if not entries:
            return []
        
        # Scorer chaque entrée
        scored = []
        for entry in entries:
            # Similarité cosinus
            similarity = self._cosine_similarity(query_embeddings, entry.embeddings)
            
            # Decay temporel
            turns_ago = current_turn - entry.turn_number
            decay = self._compute_decay(turns_ago)
            
            # Score final = similarité * decay
            final_score = similarity * decay
            
            # Filtrer par threshold
            if final_score >= self.similarity_threshold:
                scored.append({
                    'content': entry.content,
                    'turn_number': entry.turn_number,
                    'similarity': round(similarity, 4),
                    'decay': round(decay, 4),
                    'final_score': round(final_score, 4)
                })
        
        # Trier par score décroissant et prendre top-k
        scored.sort(key=lambda x: x['final_score'], reverse=True)
        return scored[:top_k]
    
    def format_memory_echo(self, recalled: List[Dict]) -> Optional[str]:
        """
        Formate mémoire rappelée en [MEMORY ECHO] pour injection.
        
        Args:
            recalled: Liste de résultats recall_memory()
        
        Returns:
            String formatée pour injection ou None si vide
        """
        if not recalled:
            return None
        
        lines = ["[MEMORY ECHO]"]
        for item in recalled:
            lines.append(f"  - {item['content'][:100]}... (relevance: {item['final_score']})")
        
        return "\n".join(lines)
    
    def get_memory_stats(self, session_id: str) -> Dict:
        """
        Statistiques mémoire pour une session.
        
        Args:
            session_id: Session à analyser
        
        Returns:
            Dict avec entry_count, oldest_turn, newest_turn, etc.
        """
        if session_id not in self.memory:
            return {
                'session_id': session_id,
                'entry_count': 0,
                'memory_status': 'empty'
            }
        
        entries = self.memory[session_id]
        turns = [e.turn_number for e in entries]
        
        return {
            'session_id': session_id,
            'entry_count': len(entries),
            'oldest_turn': min(turns),
            'newest_turn': max(turns),
            'turn_span': max(turns) - min(turns) + 1,
            'memory_status': 'active'
        }
    
    def clear_session_memory(self, session_id: str) -> bool:
        """
        Nettoie la mémoire d'une session.
        
        Args:
            session_id: Session à nettoyer
        
        Returns:
            True si nettoyé, False si session inexistante
        """
        if session_id in self.memory:
            del self.memory[session_id]
            return True
        return False
    
    def dict(self):
        """Conversion dict pour JSON - hérité de parent"""
        return super().dict()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_memory_instance: SemanticMemory | None = None


def get_semantic_memory(
    level: int = 3,
    max_memory_entries: int = 50,
    decay_rate: float = 0.01,
    similarity_threshold: float = 0.6
) -> SemanticMemory:
    """
    Get or create SemanticMemory singleton instance.

    The memory is shared across all requests but sessions are isolated
    via session_id keys in the internal dictionary.

    Usage:
        memory = get_semantic_memory()
        memory.store_memory(session_id, content, embeddings, turn)
        recalled = memory.recall_memory(session_id, query_emb, current_turn)

    Returns:
        SemanticMemory: Singleton instance with session-isolated memory
    """
    global _memory_instance

    if _memory_instance is None:
        _memory_instance = SemanticMemory(
            level=level,
            max_memory_entries=max_memory_entries,
            decay_rate=decay_rate,
            similarity_threshold=similarity_threshold
        )

    return _memory_instance


def clear_semantic_memory() -> None:
    """
    Clear the singleton instance (useful for testing or reset).
    """
    global _memory_instance

    if _memory_instance is not None:
        _memory_instance.memory.clear()
        _memory_instance = None
