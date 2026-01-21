"""
Consciousness Adaptation - Niveau 2 (Adaptatif)

Implémente adaptation progressive des profils Bezier basée sur métriques.

Règles d'adaptation:
1. Tension > 0.75 → Réduire tau_c (5%)
2. Coherence < 0.3 → Ajuster rho (concentration)
3. Fit > 0.8 + Stabilité > 0.7 → Encourager exploration (delta_r)
4. Pressure > 0.85 → Réduire charge (tau_c 7.5%, delta_r 5%)
5. Session > 30 msgs + Tension stable → No-op
"""

from typing import Optional, Dict
from .metrics import ConsciousnessMetrics, ConsciousnessMonitor


class AdaptiveConsciousness(ConsciousnessMonitor):
    """
    Extension ConsciousnessMonitor (Niveau 2) avec adaptation graduelle.
    
    Suggère ajustements progressifs (5% par défaut) aux paramètres Bezier
    basé sur métriques et historique session.
    """
    
    def __init__(self, level: int = 2, adaptation_rate: float = 0.05):
        """
        Initialise moniteur adaptatif.
        
        Args:
            level: Niveau conscience (2 pour adaptation, hérité par défaut)
            adaptation_rate: Pourcentage d'ajustement par interaction (défaut 5%)
        """
        super().__init__(level)
        self.adaptation_rate = adaptation_rate
    
    def suggest_adjustments(
        self,
        metrics: ConsciousnessMetrics,
        current_profile: Dict[str, float],
        session_length: int = 1
    ) -> Optional[Dict]:
        """
        Suggère ajustements basés sur métriques + historique session.
        
        Args:
            metrics: Métriques épistémiques Phase 1
            current_profile: Profil actuel (tau_c, rho, delta_r)
            session_length: Nombre de messages dans session
        
        Returns:
            dict avec multiplicateurs et raison, ou None si aucun ajustement
        """
        if self.level < 2:
            return None
        
        suggestions = {}
        reasons = []
        
        # Règle 5 : Session longue + tension stable → No-op
        if session_length > 30 and 0.4 < metrics.tension < 0.6:
            return None
        
        # Règle 1 : Tension > 0.75 → Réduire tau_c
        if metrics.tension > 0.75:
            suggestions['tau_c_multiplier'] = 1.0 - self.adaptation_rate
            reasons.append(
                f"High tension ({metrics.tension:.2f}) - reducing tau_c"
            )
        
        # Règle 4 : Pressure > 0.85 → Réduire charge (double)
        if metrics.pressure > 0.85:
            if 'tau_c_multiplier' not in suggestions:
                suggestions['tau_c_multiplier'] = 1.0 - (self.adaptation_rate * 1.5)
            else:
                # Combiner avec règle 1
                suggestions['tau_c_multiplier'] *= (1.0 - (self.adaptation_rate * 0.5))
            
            suggestions['delta_r_multiplier'] = 1.0 - self.adaptation_rate
            reasons.append(
                f"Very high pressure ({metrics.pressure:.2f}) - reducing load"
            )
        
        # Règle 2 : Coherence < 0.3 → Ajuster rho
        if metrics.coherence < 0.3:
            current_rho = current_profile.get('rho', 0.0)
            if current_rho > 0:
                suggestions['rho_shift'] = -self.adaptation_rate
            elif current_rho < 0:
                suggestions['rho_shift'] = self.adaptation_rate
            else:
                suggestions['rho_shift'] = -self.adaptation_rate
            
            reasons.append(
                f"Low coherence ({metrics.coherence:.2f}) - adjusting focus"
            )
        
        # Règle 3 : Fit > 0.8 ET Stabilité > 0.7 → Encourager exploration
        if metrics.fit > 0.8 and metrics.stability_score > 0.7:
            suggestions['delta_r_multiplier'] = 1.0 + (self.adaptation_rate * 0.5)
            reasons.append(
                f"High fit ({metrics.fit:.2f}) & stability ({metrics.stability_score:.2f}) - "
                f"encouraging exploration"
            )
        
        if not suggestions:
            return None
        
        return {
            **suggestions,
            "reason": "; ".join(reasons),
            "triggered_by": {
                "coherence": metrics.coherence,
                "tension": metrics.tension,
                "fit": metrics.fit,
                "pressure": metrics.pressure,
                "stability_score": metrics.stability_score
            }
        }
    
    def apply_adjustments(
        self,
        current_params: Dict[str, float],
        adjustments: Dict
    ) -> Dict[str, float]:
        """
        Applique multiplicateurs aux paramètres avec enforcement bornes.
        
        Args:
            current_params: Paramètres actuels (tau_c, rho, delta_r)
            adjustments: Suggestions d'ajustements (multiplicateurs + shift)
        
        Returns:
            Paramètres modifiés et normalisés
        """
        if not adjustments:
            return current_params
        
        # Initialiser tous les paramètres avec defaults
        modified = {
            'tau_c': current_params.get('tau_c', 1.0),
            'rho': current_params.get('rho', 0.0),
            'delta_r': current_params.get('delta_r', 0.0),
            **{k: v for k, v in current_params.items() if k not in ['tau_c', 'rho', 'delta_r']}
        }
        
        # Appliquer multiplicateurs
        if 'tau_c_multiplier' in adjustments:
            modified['tau_c'] = modified['tau_c'] * adjustments['tau_c_multiplier']
        
        if 'delta_r_multiplier' in adjustments:
            modified['delta_r'] = modified['delta_r'] * adjustments['delta_r_multiplier']
        
        # Appliquer shifts (rho)
        if 'rho_shift' in adjustments:
            modified['rho'] = modified['rho'] + adjustments['rho_shift']
        
        # Enforce bornes
        modified['tau_c'] = max(0.5, min(2.0, modified['tau_c']))
        modified['rho'] = max(-1.0, min(1.0, modified['rho']))
        modified['delta_r'] = max(-1.0, min(1.0, modified['delta_r']))
        
        return modified
    
    def dict(self):
        """Conversion dict pour JSON - hérité de parent"""
        return super().dict()
