"""
LYRA CLEAN - BEZIER PHYSICS ENGINE
==================================

Deterministic trajectory engine using cubic Bezier curves.

Replaces:
- lyra_core/policies.py (unstable PID feedback)
- lyra_nemeton/nemeton_controller.py (reactive mutations)

Philosophy:
- NO reactive feedback (unstable oscillations)
- YES ballistic trajectories (predictable evolution)
- Parameters follow smooth curves defined at session start

Mathematical Foundation:
- Cubic Bezier: B(t) = (1-t)³P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³ P₃
- t ∈ [0, 1]: Normalized time parameter (session progress)
- Control points [P₀, P₁, P₂, P₃]: Define curve shape

Author: Refactored from Lyra_Uni_3 legacy
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BezierPoint:
    """
    Control point for Bezier curve.

    Attributes:
        t: Time parameter [0, 1]
        value: Parameter value (e.g., τc, ρ, δr)
    """
    t: float
    value: float

    def __post_init__(self):
        """Validate ranges."""
        if not (0.0 <= self.t <= 1.0):
            raise ValueError(f"t must be in [0, 1], got {self.t}")


class CubicBezier:
    """
    Cubic Bezier curve interpolator.

    Mathematical form:
        B(t) = (1-t)³P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³ P₃

    where P₀, P₁, P₂, P₃ are control points.

    Usage:
        curve = CubicBezier([
            BezierPoint(0.0, 1.0),   # Start: τc = 1.0
            BezierPoint(0.3, 1.2),   # Control point 1
            BezierPoint(0.7, 0.9),   # Control point 2
            BezierPoint(1.0, 1.0)    # End: τc = 1.0
        ])

        tau_c = curve.evaluate(0.5)  # Get value at t=0.5 (mid-session)
    """

    def __init__(self, control_points: List[BezierPoint]):
        """
        Initialize Bezier curve.

        Args:
            control_points: List of 4 control points [P₀, P₁, P₂, P₃]

        Raises:
            ValueError: If not exactly 4 points provided
        """
        if len(control_points) != 4:
            raise ValueError(f"Cubic Bezier requires exactly 4 control points, got {len(control_points)}")

        # Sort by t (should already be ordered, but enforce)
        self.points = sorted(control_points, key=lambda p: p.t)

        # Validate monotonicity (t must increase)
        for i in range(1, len(self.points)):
            if self.points[i].t <= self.points[i-1].t:
                raise ValueError(f"Control points must have strictly increasing t values")

        # Validate endpoints
        if not math.isclose(self.points[0].t, 0.0, abs_tol=1e-6):
            raise ValueError(f"First control point must have t=0.0, got {self.points[0].t}")

        if not math.isclose(self.points[3].t, 1.0, abs_tol=1e-6):
            raise ValueError(f"Last control point must have t=1.0, got {self.points[3].t}")

    def evaluate(self, t: float) -> float:
        """
        Evaluate Bezier curve at parameter t.

        Args:
            t: Time parameter ∈ [0, 1]

        Returns:
            Interpolated value

        Mathematical formula:
            B(t) = (1-t)³P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³ P₃
        """
        # Clamp t to [0, 1]
        t = max(0.0, min(1.0, t))

        # Extract values
        p0, p1, p2, p3 = [p.value for p in self.points]

        # Bernstein polynomial coefficients
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3

        # Compute weighted sum
        return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

    def derivative(self, t: float) -> float:
        """
        Evaluate first derivative dB/dt at parameter t.

        Useful for:
        - Detecting rate of change
        - Identifying inflection points
        - Adaptive sampling

        Returns:
            Rate of change at t
        """
        t = max(0.0, min(1.0, t))

        p0, p1, p2, p3 = [p.value for p in self.points]

        # Derivative formula for cubic Bezier
        # dB/dt = 3(1-t)²(P₁-P₀) + 6(1-t)t(P₂-P₁) + 3t²(P₃-P₂)
        return (
            3 * (1 - t) ** 2 * (p1 - p0) +
            6 * (1 - t) * t * (p2 - p1) +
            3 * t ** 2 * (p3 - p2)
        )

    @classmethod
    def from_json(cls, points_json: List[List[float]]) -> "CubicBezier":
        """
        Create curve from JSON representation.

        Args:
            points_json: List of [t, value] pairs
                         Example: [[0, 1.0], [0.3, 1.2], [0.7, 0.9], [1, 1.0]]

        Returns:
            CubicBezier instance
        """
        control_points = [BezierPoint(t, value) for t, value in points_json]
        return cls(control_points)


@dataclass(frozen=True)
class PhysicsState:
    """
    Immutable physics state at time t.

    Replaces mutable State from legacy code.

    Attributes:
        t: Normalized time ∈ [0, 1]
        tau_c: Tension/temperature parameter
        rho: Focus/polarity parameter
        delta_r: Scheduling parameter
        kappa: Curvature parameter (optional)
    """
    t: float
    tau_c: float
    rho: float
    delta_r: float
    kappa: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            't': self.t,
            'tau_c': self.tau_c,
            'rho': self.rho,
            'delta_r': self.delta_r,
            'kappa': self.kappa
        }


class BezierEngine:
    """
    Physics engine for deterministic parameter trajectories.

    Key Concept:
    - At session start, select a profile (e.g., "creative", "safe")
    - Profile defines Bezier curves for τc, ρ, δr
    - Parameters evolve smoothly along curves (no reactive jumps)
    - Predictable, stable, tunable behavior

    Usage:
        engine = BezierEngine(profile_data)
        state = engine.compute_state(t=0.5)  # Mid-session
        print(f"τc = {state.tau_c}, ρ = {state.rho}")
    """

    def __init__(
        self,
        tau_c_curve: CubicBezier,
        rho_curve: CubicBezier,
        delta_r_curve: CubicBezier,
        kappa_curve: CubicBezier | None = None
    ):
        """
        Initialize physics engine with Bezier curves.

        Args:
            tau_c_curve: Tension trajectory
            rho_curve: Focus trajectory
            delta_r_curve: Scheduling trajectory
            kappa_curve: Curvature trajectory (optional, default constant 0.5)
        """
        self.tau_c_curve = tau_c_curve
        self.rho_curve = rho_curve
        self.delta_r_curve = delta_r_curve
        self.kappa_curve = kappa_curve

    def compute_state(self, t: float) -> PhysicsState:
        """
        Compute physics state at time t.

        Args:
            t: Normalized time ∈ [0, 1]
               - 0.0 = Session start
               - 0.5 = Mid-session
               - 1.0 = Session end (or asymptotic limit)

        Returns:
            Immutable PhysicsState with all parameters

        Example:
            # At 30% through conversation
            state = engine.compute_state(0.3)

            # Use parameters for LLM generation
            temperature = map_tau_to_temperature(state.tau_c)
            presence_penalty = map_rho_to_penalty(state.rho)
        """
        tau_c = self.tau_c_curve.evaluate(t)
        rho = self.rho_curve.evaluate(t)
        delta_r = self.delta_r_curve.evaluate(t)

        kappa = self.kappa_curve.evaluate(t) if self.kappa_curve else 0.5

        return PhysicsState(
            t=t,
            tau_c=tau_c,
            rho=rho,
            delta_r=delta_r,
            kappa=kappa
        )

    def sample_trajectory(self, num_points: int = 20) -> List[PhysicsState]:
        """
        Sample trajectory at evenly spaced points.

        Useful for:
        - Visualization
        - Pre-computation
        - Debugging

        Args:
            num_points: Number of samples ∈ [0, 1]

        Returns:
            List of PhysicsState snapshots
        """
        if num_points < 2:
            raise ValueError("num_points must be >= 2")

        samples = []
        for i in range(num_points):
            t = i / (num_points - 1)  # Linspace [0, 1]
            samples.append(self.compute_state(t))

        return samples

    @classmethod
    def from_profile(cls, profile_data: Dict[str, Any]) -> "BezierEngine":
        """
        Create engine from database profile.

        Args:
            profile_data: Dict with keys:
                - tau_c_curve: List of [t, value] pairs
                - rho_curve: List of [t, value] pairs
                - delta_r_curve: List of [t, value] pairs
                - kappa_curve: (optional) List of [t, value] pairs

        Returns:
            Initialized BezierEngine

        Example:
            profile = await db.get_profile("creative")
            engine = BezierEngine.from_profile(profile)
        """
        tau_c_curve = CubicBezier.from_json(profile_data['tau_c_curve'])
        rho_curve = CubicBezier.from_json(profile_data['rho_curve'])
        delta_r_curve = CubicBezier.from_json(profile_data['delta_r_curve'])

        kappa_curve = None
        if profile_data.get('kappa_curve'):
            kappa_curve = CubicBezier.from_json(profile_data['kappa_curve'])

        return cls(tau_c_curve, rho_curve, delta_r_curve, kappa_curve)


# ============================================================================
# TIME PARAMETER MAPPING (Session progress → t ∈ [0, 1])
# ============================================================================

class TimeMapper:
    """
    Maps conversation progress to normalized time parameter t.

    Strategies:
    - Linear: t = n / max_messages
    - Logarithmic: t = log(n+1) / log(max+1)  (slower early, faster late)
    - Sigmoid: t = 1 / (1 + exp(-k*(n - n₀)))  (smooth acceleration)

    Default: Logarithmic (natural conversation feel)
    """

    @staticmethod
    def linear(message_count: int, max_messages: int = 100) -> float:
        """
        Linear mapping: t = n / max.

        Args:
            message_count: Current message count
            max_messages: Expected max messages in session

        Returns:
            t ∈ [0, 1]
        """
        return min(1.0, message_count / max_messages)

    @staticmethod
    def logarithmic(message_count: int, max_messages: int = 100) -> float:
        """
        Logarithmic mapping: slower early progress.

        Args:
            message_count: Current message count
            max_messages: Expected max messages

        Returns:
            t ∈ [0, 1]

        Example:
            msg=1  → t=0.10
            msg=10 → t=0.46
            msg=50 → t=0.85
            msg=100→ t=1.00
        """
        if message_count == 0:
            return 0.0

        return min(1.0, math.log(message_count + 1) / math.log(max_messages + 1))

    @staticmethod
    def sigmoid(
        message_count: int,
        midpoint: int = 50,
        steepness: float = 0.1
    ) -> float:
        """
        Sigmoid mapping: smooth S-curve.

        Args:
            message_count: Current message count
            midpoint: Inflection point (t=0.5)
            steepness: Curve steepness

        Returns:
            t ∈ [0, 1]
        """
        return 1.0 / (1.0 + math.exp(-steepness * (message_count - midpoint)))


# ============================================================================
# PARAMETER MAPPERS (Physics → Ollama API)
# ============================================================================

def map_tau_to_temperature(tau_c: float, base_temp: float = 0.8) -> float:
    """
    Map tension τc to Ollama temperature.

    Formula: temperature = base_temp / τc

    Physics interpretation:
    - τc high → temperature low → focused responses
    - τc low → temperature high → creative responses

    Args:
        tau_c: Tension parameter (typically [0.5, 2.0])
        base_temp: Base temperature

    Returns:
        Temperature ∈ [0.1, 1.5]
    """
    tau_c = max(0.1, min(2.5, tau_c))
    temp = base_temp / tau_c
    return max(0.1, min(1.5, temp))


def map_rho_to_penalties(rho: float) -> Dict[str, float]:
    """
    Map focus ρ to Ollama penalties.

    Physics interpretation:
    - ρ > 0 → expansive (explore new concepts)
    - ρ < 0 → conservative (stick to known patterns)
    - ρ = 0 → neutral

    Args:
        rho: Focus parameter ∈ [-1, 1]

    Returns:
        Dict with keys: presence_penalty, frequency_penalty
    """
    rho = max(-1.0, min(1.0, rho))

    return {
        "presence_penalty": -0.6 * rho,      # Negative ρ → discourage new tokens
        "frequency_penalty": 0.3 * abs(rho)  # High |ρ| → avoid repetition
    }


def map_kappa_to_style_hints(kappa: float) -> List[str]:
    """
    Map curvature κ to style constraints.

    Physics interpretation:
    - κ high → structured, formal
    - κ low → exploratory, loose

    Args:
        kappa: Curvature ∈ [0, 1]

    Returns:
        List of style hint strings
    """
    hints = []

    if kappa > 0.7:
        hints.append("Structure responses in clear sections")
        hints.append("Use numbered lists when appropriate")
        hints.append("Maintain formal tone")
    elif kappa < 0.3:
        hints.append("Feel free to explore tangents")
        hints.append("Use analogies and metaphors")
        hints.append("Conversational tone encouraged")
    else:
        hints.append("Balance structure and exploration")

    return hints
