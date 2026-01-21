"""
LYRA CLEAN - Consciousness System
==================================

Multi-level consciousness framework for LLM interactions:
- Level 0: Baseline (no consciousness)
- Level 1: Passive metrics (coherence, tension, fit, pressure)
- Level 2: Adaptive consciousness (gradual profile adjustments)
- Level 3: Semantic memory (vector-based recall with temporal decay)
"""

from .metrics import ConsciousnessMetrics, ConsciousnessMonitor
from .adaptation import AdaptiveConsciousness
from .memory import SemanticMemory, MemoryEntry

__all__ = [
    'ConsciousnessMetrics',
    'ConsciousnessMonitor',
    'AdaptiveConsciousness',
    'SemanticMemory',
    'MemoryEntry'
]
