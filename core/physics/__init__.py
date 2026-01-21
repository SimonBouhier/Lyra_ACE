"""
Physics engine for deterministic parameter trajectories.

Exports:
- BezierEngine: Main trajectory engine
- PhysicsState: Immutable state at time t
- TimeMapper: Message count → t ∈ [0, 1]
"""
from .bezier import (
    BezierEngine,
    PhysicsState,
    CubicBezier,
    BezierPoint,
    TimeMapper,
    map_tau_to_temperature,
    map_rho_to_penalties,
    map_kappa_to_style_hints
)

__all__ = [
    'BezierEngine',
    'PhysicsState',
    'CubicBezier',
    'BezierPoint',
    'TimeMapper',
    'map_tau_to_temperature',
    'map_rho_to_penalties',
    'map_kappa_to_style_hints'
]
