"""
LYRA-ACE - GRAPH MUTATION API
=============================

Endpoints pour les opérations sur le graphe sémantique.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional

from app.models import (
    GraphDeltaRequest, GraphDeltaResponse, KappaResponse, ErrorResponse
)
from database import (
    get_db, ISpaceDB, GraphDelta, DeltaOperation,
    DeltaValidationError, MutationLimitExceededError
)


router = APIRouter(prefix="/graph", tags=["graph"])


async def get_database() -> ISpaceDB:
    """Dependency: Database instance."""
    return await get_db()


@router.post("/delta", response_model=GraphDeltaResponse)
async def apply_delta(
    request: GraphDeltaRequest,
    session_id: Optional[str] = Query(None, description="Session ID for audit"),
    kappa_alpha: float = Query(0.5, ge=0.0, le=1.0, description="Kappa hybrid coefficient"),
    db: ISpaceDB = Depends(get_database)
):
    """
    Applique un delta au graphe sémantique.

    Le delta est enregistré dans l'historique pour permettre le rollback.

    Example:
        POST /graph/delta?session_id=abc-123&kappa_alpha=0.5
        {
            "operation": "add_edge",
            "source": "entropy",
            "target": "chaos",
            "weight": 0.75,
            "confidence": 0.9
        }
    """
    try:
        delta = GraphDelta(
            operation=DeltaOperation(request.operation),
            source=request.source,
            target=request.target,
            weight=request.weight,
            confidence=request.confidence,
            model_source=request.model_source,
            reason=request.reason
        )

        result = await db.apply_delta(delta, session_id, kappa_alpha)

        return GraphDeltaResponse(
            delta_id=result.delta_id,
            operation=result.operation.value,
            source=result.source,
            target=result.target,
            old_weight=result.old_weight,
            new_weight=result.weight,
            old_kappa=result.old_kappa,
            new_kappa=result.new_kappa,
            applied_at=result.applied_at
        )

    except DeltaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MutationLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/kappa/{source}/{target}", response_model=KappaResponse)
async def compute_kappa(
    source: str,
    target: str,
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Hybrid coefficient"),
    store_history: bool = Query(False, description="Store in kappa_history table"),
    db: ISpaceDB = Depends(get_database)
):
    """
    Calcule la courbure κ hybride pour une arête.

    Formules:
    - Ollivier: κ_o = 1/deg(u) + 1/deg(v) - 2/w
    - Jaccard: κ_j = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    - Hybride: κ = α * κ_o_norm + (1-α) * κ_j

    Example:
        GET /graph/kappa/entropy/information?alpha=0.6
    """
    result = await db.compute_kappa_live(source, target, alpha, store_history)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Edge {source} -> {target} not found"
        )

    return KappaResponse(
        source=source,
        target=target,
        kappa_ollivier=result["kappa_ollivier"],
        kappa_jaccard=result["kappa_jaccard"],
        kappa_hybrid=result["kappa_hybrid"],
        alpha=result["alpha"]
    )


@router.get("/deltas")
async def get_delta_history(
    session_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    include_rolled_back: bool = Query(False),
    db: ISpaceDB = Depends(get_database)
):
    """
    Récupère l'historique des deltas.

    Example:
        GET /graph/deltas?session_id=abc-123&limit=50
    """
    deltas = await db.get_delta_history(session_id, limit, include_rolled_back)
    return {"deltas": deltas, "count": len(deltas)}


@router.post("/rollback")
async def rollback_deltas(
    session_id: str = Query(..., description="Session ID"),
    to_timestamp: Optional[float] = Query(None, description="Rollback to this timestamp"),
    delta_ids: Optional[List[int]] = Query(None, description="Specific delta IDs to rollback"),
    db: ISpaceDB = Depends(get_database)
):
    """
    Annule des deltas (restaure l'état précédent).

    Example:
        POST /graph/rollback?session_id=abc-123&to_timestamp=1704067200
    """
    if not to_timestamp and not delta_ids:
        raise HTTPException(
            status_code=400,
            detail="Either to_timestamp or delta_ids must be provided"
        )

    count = await db.rollback_deltas(session_id, to_timestamp, delta_ids)
    return {"rolled_back": count, "session_id": session_id}


@router.get("/stats")
async def get_mutation_stats(db: ISpaceDB = Depends(get_database)):
    """
    Statistiques sur les mutations du graphe.

    Example:
        GET /graph/stats
    """
    stats = await db.get_graph_mutation_stats()
    return stats
