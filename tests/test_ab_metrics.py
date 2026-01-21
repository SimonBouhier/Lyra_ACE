"""
TEST A/B - Vérification des métriques Consciousness
===================================================

Teste que les métriques varient réellement en fonction des inputs,
pas des valeurs plates dues aux mocks.

Scenarios:
- A: Faible contexte (tension haute, coherence basse)
- B: Fort contexte (tension basse, coherence haute)
- C: Cas extrêmes (edge cases)
"""

import pytest
from services.consciousness.metrics import ConsciousnessMetrics, ConsciousnessMonitor


class TestMetricsVariability:
    """Vérifie que les métriques varient réellement"""
    
    def test_coherence_varies_with_context_weight(self):
        """A: Coherence doit augmenter avec context_weight"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Faible contexte
        result_low = monitor.compute_metrics(
            context_weight=1.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Fort contexte (10x plus de poids)
        result_high = monitor.compute_metrics(
            context_weight=10.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        print(f"\nCoherence - Faible contexte: {result_low.coherence:.3f}")
        print(f"Coherence - Fort contexte: {result_high.coherence:.3f}")
        
        assert result_high.coherence > result_low.coherence, \
            f"Coherence should increase: {result_low.coherence} vs {result_high.coherence}"
    
    def test_coherence_varies_with_num_concepts(self):
        """B: Coherence doit augmenter avec num_concepts"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Peu de concepts
        result_few = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=1,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Beaucoup de concepts
        result_many = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=10,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        print(f"\nCoherence - 1 concept: {result_few.coherence:.3f}")
        print(f"Coherence - 10 concepts: {result_many.coherence:.3f}")
        
        # Avec même poids mais plus de concepts, coherence basse
        assert result_few.coherence > result_many.coherence, \
            f"Coherence should decrease with more concepts (same weight): {result_few.coherence} vs {result_many.coherence}"
    
    def test_pressure_varies_with_tau_c(self):
        """C: Pressure doit augmenter avec tau_c"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Tau_c bas (détente)
        result_low_tau = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 0.5, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Tau_c haut (tension)
        result_high_tau = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 2.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        print(f"\nPressure - Tau_c=0.5: {result_low_tau.pressure:.3f}")
        print(f"Pressure - Tau_c=2.0: {result_high_tau.pressure:.3f}")
        
        assert result_high_tau.pressure > result_low_tau.pressure, \
            f"Pressure should increase with tau_c: {result_low_tau.pressure} vs {result_high_tau.pressure}"
    
    def test_pressure_varies_with_delta_r(self):
        """D: Pressure doit augmenter avec |delta_r|"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Delta_r nul
        result_zero_dr = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Delta_r élevé
        result_high_dr = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 1.0},
            response_length=150
        )
        
        print(f"\nPressure - Delta_r=0.0: {result_zero_dr.pressure:.3f}")
        print(f"Pressure - Delta_r=1.0: {result_high_dr.pressure:.3f}")
        
        assert result_high_dr.pressure > result_zero_dr.pressure, \
            f"Pressure should increase with delta_r: {result_zero_dr.pressure} vs {result_high_dr.pressure}"
    
    def test_fit_varies_with_response_length(self):
        """E: Fit doit varier avec response_length vs attentes (rho)"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Rho > 0 (attendu expansif ~250 mots)
        # Réponse courte = mauvais fit
        result_bad_fit = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.5, 'delta_r': 0.0},
            response_length=100  # Trop court
        )
        
        # Même rho, réponse longue = bon fit
        result_good_fit = monitor.compute_metrics(
            context_weight=5.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.5, 'delta_r': 0.0},
            response_length=250  # Aligné aux attentes
        )
        
        print(f"\nFit - Réponse courte (100): {result_bad_fit.fit:.3f}")
        print(f"Fit - Réponse alignée (250): {result_good_fit.fit:.3f}")
        
        assert result_good_fit.fit > result_bad_fit.fit, \
            f"Fit should improve with aligned response: {result_bad_fit.fit} vs {result_good_fit.fit}"
    
    def test_tension_varies_with_coherence(self):
        """F: Tension doit diminuer quand coherence augmente"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Faible coherence (faible contexte)
        result_low_coh = monitor.compute_metrics(
            context_weight=0.5,
            num_concepts=10,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Haute coherence (fort contexte)
        result_high_coh = monitor.compute_metrics(
            context_weight=10.0,
            num_concepts=5,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        print(f"\nTension - Faible coherence: {result_low_coh.tension:.3f}")
        print(f"Tension - Haute coherence: {result_high_coh.tension:.3f}")
        
        assert result_high_coh.tension < result_low_coh.tension, \
            f"Tension should decrease with coherence: {result_low_coh.tension} vs {result_high_coh.tension}"
    
    def test_stability_varies(self):
        """G: Stability score doit varier avec coherence et tension"""
        monitor = ConsciousnessMonitor(level=1)
        
        # Scenario stable: haute coherence, basse tension
        result_stable = monitor.compute_metrics(
            context_weight=8.0,
            num_concepts=5,
            physics_state={'tau_c': 0.8, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        
        # Scenario instable: basse coherence, haute tension
        result_unstable = monitor.compute_metrics(
            context_weight=1.0,
            num_concepts=15,
            physics_state={'tau_c': 2.0, 'rho': 0.0, 'delta_r': 0.8},
            response_length=150
        )
        
        print(f"\nStability - Stable scenario: {result_stable.stability_score:.3f}")
        print(f"Stability - Unstable scenario: {result_unstable.stability_score:.3f}")
        
        assert result_stable.stability_score > result_unstable.stability_score, \
            f"Stability should be higher in stable scenario: {result_stable.stability_score} vs {result_unstable.stability_score}"
    
    def test_all_metrics_not_constant(self):
        """H: Vérifier que les métriques varient réellement sur 10 scenarios"""
        monitor = ConsciousnessMonitor(level=1)
        
        scenarios = [
            {'weight': 1.0, 'concepts': 2, 'tau_c': 0.5, 'delta_r': 0.0, 'rho': 0.0, 'length': 100},
            {'weight': 2.0, 'concepts': 3, 'tau_c': 0.7, 'delta_r': 0.1, 'rho': -0.2, 'length': 120},
            {'weight': 3.0, 'concepts': 4, 'tau_c': 0.9, 'delta_r': 0.2, 'rho': -0.1, 'length': 140},
            {'weight': 4.0, 'concepts': 5, 'tau_c': 1.0, 'delta_r': 0.3, 'rho': 0.0, 'length': 150},
            {'weight': 5.0, 'concepts': 6, 'tau_c': 1.1, 'delta_r': 0.4, 'rho': 0.1, 'length': 170},
            {'weight': 6.0, 'concepts': 7, 'tau_c': 1.2, 'delta_r': 0.5, 'rho': 0.2, 'length': 190},
            {'weight': 7.0, 'concepts': 8, 'tau_c': 1.3, 'delta_r': 0.6, 'rho': 0.3, 'length': 210},
            {'weight': 8.0, 'concepts': 9, 'tau_c': 1.4, 'delta_r': 0.7, 'rho': 0.4, 'length': 230},
            {'weight': 9.0, 'concepts': 10, 'tau_c': 1.5, 'delta_r': 0.8, 'rho': 0.5, 'length': 250},
            {'weight': 10.0, 'concepts': 11, 'tau_c': 1.6, 'delta_r': 0.9, 'rho': 0.6, 'length': 270},
        ]
        
        results = []
        print("\nVariability Test - 10 scenarios:")
        print("Scenario | Coherence | Tension | Fit    | Pressure | Stability")
        print("-" * 70)
        
        for i, s in enumerate(scenarios, 1):
            result = monitor.compute_metrics(
                context_weight=s['weight'],
                num_concepts=s['concepts'],
                physics_state={'tau_c': s['tau_c'], 'rho': s['rho'], 'delta_r': s['delta_r']},
                response_length=s['length']
            )
            results.append(result)
            print(f"  {i:2d}    | {result.coherence:7.3f}  | {result.tension:6.3f} | {result.fit:6.3f} | {result.pressure:8.3f} | {result.stability_score:7.3f}")
        
        # Vérifier que les valeurs varient
        coherences = [r.coherence for r in results]
        tensions = [r.tension for r in results]
        fits = [r.fit for r in results]
        pressures = [r.pressure for r in results]
        
        assert len(set(round(c, 3) for c in coherences)) > 1, "Coherence should vary"
        assert len(set(round(t, 3) for t in tensions)) > 1, "Tension should vary"
        assert len(set(round(f, 3) for f in fits)) > 1, "Fit should vary"
        assert len(set(round(p, 3) for p in pressures)) > 1, "Pressure should vary"
        
        print("\n✓ Toutes les métriques varient correctement")


class TestMetricsEdgeCases:
    """Tests des cas limites"""
    
    def test_zero_concepts(self):
        """Cas limite: 0 concepts"""
        monitor = ConsciousnessMonitor(level=1)
        result = monitor.compute_metrics(
            context_weight=0.0,
            num_concepts=0,
            physics_state={'tau_c': 1.0, 'rho': 0.0, 'delta_r': 0.0},
            response_length=150
        )
        assert result.coherence == 0.0, "Coherence should be 0 with 0 concepts"
    
    def test_extreme_values(self):
        """Cas limite: valeurs extrêmes"""
        monitor = ConsciousnessMonitor(level=1)
        result = monitor.compute_metrics(
            context_weight=100.0,
            num_concepts=100,
            physics_state={'tau_c': 10.0, 'rho': 1.0, 'delta_r': 1.0},
            response_length=1000
        )
        
        # Toutes les métriques doivent être en [0,1]
        assert 0.0 <= result.coherence <= 1.0
        assert 0.0 <= result.tension <= 1.0
        assert 0.0 <= result.fit <= 1.0
        assert 0.0 <= result.pressure <= 1.0
        assert 0.0 <= result.stability_score <= 1.0
        
        print(f"\nExtreme values test passed:")
        print(f"  Coherence: {result.coherence:.3f}")
        print(f"  Tension: {result.tension:.3f}")
        print(f"  Fit: {result.fit:.3f}")
        print(f"  Pressure: {result.pressure:.3f}")
        print(f"  Stability: {result.stability_score:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
