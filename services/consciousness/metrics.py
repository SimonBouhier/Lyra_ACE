"""
Consciousness Metrics - Niveau 1 (Passif)

Calcule metriques epistemiques sans modifier comportement :
- Coherence : densite semantique du contexte
- Tension : charge ressentie
- Fit : alignement avec attentes
- Pressure : exploration vs exploitation
"""
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ConsciousnessMetrics:
    """Metriques epistemiques calculees"""
    coherence: float      # [0,1] : Densite semantique
    tension: float        # [0,1] : Charge systeme
    fit: float           # [0,1] : Alignement
    pressure: float      # [0,1] : Exploration
    
    @property
    def stability_score(self) -> float:
        """Score composite de stabilite - normalisé [0,1]"""
        # Formule : 0.5 * (coherence + fit) - 0.5 * tension
        # Normalisation : saturation à [0,1]
        raw_score = 0.5 * (self.coherence + self.fit) - 0.5 * self.tension
        return max(0.0, min(1.0, raw_score))
    
    def dict(self):
        """Conversion dict pour JSON"""
        return {
            "coherence": round(self.coherence, 3),
            "tension": round(self.tension, 3),
            "fit": round(self.fit, 3),
            "pressure": round(self.pressure, 3),
            "stability_score": round(self.stability_score, 3)
        }


class ConsciousnessMonitor:
    """
    Moniteur de conscience (Niveau 1 : Passif)
    
    Calcule metriques epistemiques sans modifier comportement.
    L'overhead doit rester < 5ms.
    """
    
    def __init__(self, level: int = 0):
        """
        Args:
            level: Niveau de conscience (0=off, 1=passive, 2=adaptive, 3=full)
        """
        self.level = level
    
    def compute_metrics(
        self,
        context_weight: float,
        num_concepts: int,
        physics_state: dict,
        response_length: int
    ) -> Optional[ConsciousnessMetrics]:
        """
        Calcule metriques epistemiques
        
        Args:
            context_weight: Poids total du contexte graphe
            num_concepts: Nombre de concepts injectes
            physics_state: Etat physique {tau_c, rho, delta_r}
            response_length: Longueur reponse (en mots)
        
        Returns:
            ConsciousnessMetrics si level >= 1, sinon None
        """
        if self.level < 1:
            return None
        
        # 1. COHERENCE : basé sur densite graphe
        # Plus de concepts avec poids eleve = haute coherence
        coherence = self._compute_coherence(context_weight, num_concepts)
        
        # 2. PRESSURE : basé sur tau_c et delta_r
        # Tau_c eleve + delta_r eleve = haute pression
        pressure = self._compute_pressure(
            physics_state.get("tau_c", 1.0),
            physics_state.get("delta_r", 0.0)
        )
        
        # 3. FIT : basé sur rho et longueur reponse
        # Alignement entre attentes (rho) et production
        fit = self._compute_fit(
            physics_state.get("rho", 0.0),
            response_length
        )
        
        # 4. TENSION : combinaison coherence, pressure
        # Haute si structure faible OU charge forte
        tension = self._compute_tension(coherence, pressure)
        
        return ConsciousnessMetrics(
            coherence=coherence,
            tension=tension,
            fit=fit,
            pressure=pressure
        )
    
    def _compute_coherence(self, weight: float, n_concepts: int) -> float:
        """
        Coherence = densite semantique
        
        Formule : min(1.0, weight / (n_concepts * 0.8))
        
        Rationale :
        - Poids moyen de 0.8 par concept = bonne coherence
        - Sature a 1.0
        """
        if n_concepts == 0:
            return 0.0
        
        avg_weight = weight / n_concepts
        coherence = avg_weight / 0.8  # Normalisation
        return min(1.0, max(0.0, coherence))
    
    def _compute_pressure(self, tau_c: float, delta_r: float) -> float:
        """
        Pressure = charge exploration/exploitation
        
        Formule : 0.3 * |delta_r| + 0.7 * (tau_c / (tau_c + 1.0))
        
        Rationale :
        - Delta_r eleve = exploration temporelle
        - Tau_c eleve = contrainte forte
        """
        normalized_tau = tau_c / (tau_c + 1.0)
        pressure = 0.3 * abs(delta_r) + 0.7 * normalized_tau
        return min(1.0, max(0.0, pressure))
    
    def _compute_fit(self, rho: float, response_length: int) -> float:
        """
        Fit = alignement production vs attentes
        
        Formule : 1.0 - |actual - expected| / expected
        
        Rationale :
        - Rho > 0 : attendu expansif (200+ mots)
        - Rho < 0 : attendu concis (100- mots)
        """
        # Longueur attendue selon rho
        if rho > 0:
            expected_length = 200 + rho * 100  # 200-300 mots
        elif rho < 0:
            expected_length = 150 + rho * 50   # 100-150 mots
        else:
            expected_length = 150
        
        # Ecart normalise
        deviation = abs(response_length - expected_length) / expected_length
        fit = 1.0 - min(1.0, deviation)
        
        return max(0.0, min(1.0, fit))
    
    def _compute_tension(self, coherence: float, pressure: float) -> float:
        """
        Tension = stress systeme
        
        Formule : 0.4 * (1 - coherence) + 0.6 * pressure
        
        Rationale :
        - Structure faible (low coherence) = tension
        - Charge forte (high pressure) = tension
        """
        tension = 0.4 * (1.0 - coherence) + 0.6 * pressure
        return min(1.0, max(0.0, tension))
