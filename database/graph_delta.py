"""
LYRA-ACE - GRAPH DELTA MANAGEMENT
=================================

Gestion des mutations incrémentielles du graphe sémantique.

Principes:
- Deltas atomiques et auditables
- Rollback possible via historique
- Limite de mutation par batch (5% du graphe)
- Calcul κ hybride (Ollivier + Jaccard)
"""
from __future__ import annotations

import time
import math
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum


class DeltaOperation(str, Enum):
    """Types d'opérations sur le graphe."""
    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    UPDATE_NODE = "update_node"
    UPDATE_EDGE = "update_edge"
    DELETE_NODE = "delete_node"
    DELETE_EDGE = "delete_edge"


@dataclass
class GraphDelta:
    """
    Représente une mutation atomique du graphe.

    Attributes:
        operation: Type d'opération
        source: Concept source (ou unique concept pour opérations nœud)
        target: Concept cible (None pour opérations sur nœuds seuls)
        weight: Nouveau poids (pour add/update edge)
        confidence: Confiance dans ce delta [0, 1]
        model_source: Identifiant du modèle LLM ayant suggéré ce delta
        reason: Justification textuelle optionnelle
        timestamp: Moment de création du delta
    """
    operation: DeltaOperation
    source: str
    target: Optional[str] = None
    weight: Optional[float] = None
    confidence: float = 1.0
    model_source: str = "system"
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    # Champs internes (remplis après application)
    delta_id: Optional[int] = None
    old_weight: Optional[float] = None
    old_kappa: Optional[float] = None
    new_kappa: Optional[float] = None
    applied_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation pour stockage/API."""
        return {k: v.value if isinstance(v, Enum) else v
                for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphDelta":
        """Désérialisation depuis stockage/API."""
        if isinstance(data.get('operation'), str):
            data['operation'] = DeltaOperation(data['operation'])
        return cls(**{k: v for k, v in data.items()
                     if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        """Valide la cohérence du delta."""
        # Les opérations sur arêtes nécessitent source ET target
        edge_ops = {DeltaOperation.ADD_EDGE, DeltaOperation.UPDATE_EDGE, DeltaOperation.DELETE_EDGE}
        if self.operation in edge_ops and not self.target:
            return False

        # ADD et UPDATE nécessitent un poids
        if self.operation in {DeltaOperation.ADD_EDGE, DeltaOperation.UPDATE_EDGE}:
            if self.weight is None:
                return False

        # Confiance dans [0, 1]
        if not (0.0 <= self.confidence <= 1.0):
            return False

        return True


@dataclass
class DeltaBatch:
    """
    Lot de deltas à appliquer ensemble.

    Attributes:
        deltas: Liste des deltas
        session_id: Session associée
        max_mutation_ratio: Ratio maximum de mutation autorisé
    """
    deltas: List[GraphDelta]
    session_id: Optional[str] = None
    max_mutation_ratio: float = 0.05  # 5% par défaut

    def validate_batch_size(self, graph_size: int) -> bool:
        """Vérifie que le batch ne dépasse pas la limite de mutation."""
        max_changes = int(graph_size * self.max_mutation_ratio)
        return len(self.deltas) <= max(max_changes, 1)  # Au moins 1 autorisé


class KappaCalculator:
    """
    Calculateur de courbure κ hybride.

    Formules:
    - Ollivier approx: κ_o = 1/deg(u) + 1/deg(v) - 2/w
    - Jaccard: κ_j = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    - Hybride: κ = α * κ_o + (1-α) * κ_j

    Le coefficient α est ajustable selon le profil (plus structurel ou plus local).
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Coefficient de mélange [0, 1]
                   - α=1.0 : 100% Ollivier (structure globale)
                   - α=0.0 : 100% Jaccard (similarité locale)
                   - α=0.5 : Équilibré (défaut)
        """
        self.alpha = max(0.0, min(1.0, alpha))

    def ollivier_approx(
        self,
        degree_u: int,
        degree_v: int,
        weight: float
    ) -> float:
        """
        Courbure d'Ollivier approximée.

        Formule: κ = 1/deg(u) + 1/deg(v) - 2/w

        Interprétation:
        - κ > 0 : Courbure positive (nœuds "proches", cluster dense)
        - κ < 0 : Courbure négative (nœuds "éloignés", pont entre clusters)
        - κ ≈ 0 : Courbure nulle (structure plate)

        Returns:
            κ ∈ [-2, 2] typiquement, non borné
        """
        if degree_u == 0 or degree_v == 0 or weight == 0:
            return 0.0

        kappa = (1.0 / degree_u) + (1.0 / degree_v) - (2.0 / weight)
        return kappa

    def jaccard_kappa(
        self,
        neighbors_u: set,
        neighbors_v: set
    ) -> float:
        """
        Courbure basée sur l'indice de Jaccard.

        Formule: κ = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

        Interprétation:
        - κ = 1 : Voisinages identiques (forte redondance)
        - κ = 0 : Voisinages disjoints (diversité maximale)

        Returns:
            κ ∈ [0, 1]
        """
        if not neighbors_u and not neighbors_v:
            return 0.0

        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)

        if union == 0:
            return 0.0

        return intersection / union

    def compute_hybrid(
        self,
        degree_u: int,
        degree_v: int,
        weight: float,
        neighbors_u: set,
        neighbors_v: set
    ) -> Dict[str, float]:
        """
        Calcule la courbure hybride avec ses composantes.

        Returns:
            Dict avec:
            - kappa_ollivier: Composante Ollivier
            - kappa_jaccard: Composante Jaccard
            - kappa_hybrid: Valeur finale
            - alpha: Coefficient utilisé
        """
        k_ollivier = self.ollivier_approx(degree_u, degree_v, weight)
        k_jaccard = self.jaccard_kappa(neighbors_u, neighbors_v)

        # Normaliser Ollivier vers [0, 1] pour le mélange
        # Utilise tanh pour compresser vers [-1, 1] puis rescale
        k_ollivier_norm = (math.tanh(k_ollivier) + 1) / 2

        # Mélange hybride
        k_hybrid = self.alpha * k_ollivier_norm + (1 - self.alpha) * k_jaccard

        return {
            "kappa_ollivier": round(k_ollivier, 6),
            "kappa_jaccard": round(k_jaccard, 6),
            "kappa_hybrid": round(k_hybrid, 6),
            "alpha": self.alpha
        }


class DeltaValidationError(Exception):
    """Erreur de validation d'un delta."""
    pass


class MutationLimitExceededError(Exception):
    """Erreur quand le batch dépasse la limite de mutation."""
    pass
