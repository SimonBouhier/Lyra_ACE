"""
LYRA-ACE - MULTI-MODEL API
==========================

Endpoints pour la génération multi-modèles.
"""
from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models import (
    MultiModelRequest, MultiModelResponse,
    MultiModelResponseItem, ConsensusMetricsModel
)
from app.llm_client import get_multi_model_client, MultiModelClient
from database import get_db, ISpaceDB
from core.physics import BezierEngine, TimeMapper


router = APIRouter(prefix="/multimodel", tags=["multimodel"])


async def get_database() -> ISpaceDB:
    return await get_db()


async def get_multi_client() -> MultiModelClient:
    return await get_multi_model_client()


@router.get("/models")
async def list_models(
    refresh: bool = False,
    client: MultiModelClient = Depends(get_multi_client)
):
    """
    Liste les modèles disponibles sur Ollama.

    Example:
        GET /multimodel/models
    """
    models = await client.list_available_models(refresh)
    return {"models": models, "count": len(models)}


@router.post("/generate", response_model=MultiModelResponse)
async def generate_multi(
    request: MultiModelRequest,
    db: ISpaceDB = Depends(get_database),
    client: MultiModelClient = Depends(get_multi_client)
):
    """
    Génère des réponses avec plusieurs modèles et calcule le consensus.

    Example:
        POST /multimodel/generate
        {
            "text": "Explain entropy",
            "models": ["llama3.1:8b", "mistral"],
            "profile": "analytical"
        }
    """
    # Vérifier que les modèles sont disponibles
    available = await client.list_available_models()
    unavailable = [m for m in request.models if m not in available]
    if unavailable:
        raise HTTPException(
            status_code=400,
            detail=f"Models not available: {unavailable}. Available: {available}"
        )

    # Session
    session_id = request.session_id or str(uuid.uuid4())

    # Charger le profil Bézier
    profile = await db.get_profile(request.profile)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Profile '{request.profile}' not found")

    engine = BezierEngine.from_profile(profile)

    # Obtenir le nombre de messages pour le time mapping
    async with db.connection() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?",
            (session_id,)
        )
        msg_count = (await cursor.fetchone())[0]

    t = TimeMapper.logarithmic(msg_count, max_messages=100)
    physics_state = engine.compute_state(t)

    # Préparer les messages
    messages = [{"role": "user", "content": request.text}]

    # Générer avec chaque modèle
    responses = await client.generate_sequential(
        messages=messages,
        models=request.models,
        physics_state=physics_state,
        stop_on_first_success=request.stop_on_first_success
    )

    # Calculer le consensus
    consensus = client.compute_consensus(responses)

    # Sélectionner la meilleure réponse
    best = client.select_best_response(responses, consensus)

    if not best:
        raise HTTPException(
            status_code=500,
            detail="All models failed to generate a response"
        )

    # Formater la réponse
    response_items = {
        model: MultiModelResponseItem(
            model=r.model,
            text=r.text,
            latency_ms=r.latency_ms,
            tokens=r.tokens,
            success=r.success,
            error=r.error
        )
        for model, r in responses.items()
    }

    return MultiModelResponse(
        best_response=best.text,
        best_model=best.model,
        responses=response_items,
        consensus=ConsensusMetricsModel(**consensus.to_dict()),
        session_id=session_id,
        physics_state=physics_state.to_dict()
    )
