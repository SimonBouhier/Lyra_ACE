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
    MultiModelResponseItem, ConsensusMetricsModel,
    BatchMultiModelRequest, BatchMultiModelResponse, BatchModelResponseItem
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


@router.post("/batch-generate", response_model=BatchMultiModelResponse)
async def batch_generate_multi(
    request: BatchMultiModelRequest,
    db: ISpaceDB = Depends(get_database),
    client: MultiModelClient = Depends(get_multi_client)
):
    """
    Génère des réponses pour plusieurs questions avec plusieurs modèles.

    STRATÉGIE VRAM-OPTIMALE:
    - Charge modèle 1 → traite TOUTES les questions → décharge
    - Charge modèle 2 → traite TOUTES les questions → décharge
    - etc.

    Cela minimise l'utilisation VRAM en gardant un seul modèle chargé à la fois.

    Example:
        POST /multimodel/batch-generate
        {
            "questions": [
                "What is entropy?",
                "Explain photosynthesis",
                "What is gravity?"
            ],
            "models": ["llama3.1:8b", "mistral:7b"],
            "profile": "analytical",
            "system_prompt": "Answer concisely in 2-3 sentences."
        }

    Returns:
        BatchMultiModelResponse with responses organized by model
    """
    import time
    start_time = time.time()

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
        raise HTTPException(
            status_code=400,
            detail=f"Profile '{request.profile}' not found"
        )

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

    # Générer avec batching séquentiel (VRAM-optimal)
    responses = await client.batch_generate_sequential(
        questions=request.questions,
        models=request.models,
        physics_state=physics_state,
        system_prompt=request.system_prompt
    )

    total_duration_ms = (time.time() - start_time) * 1000

    # Formater la réponse
    formatted_responses = {}
    for model, model_responses in responses.items():
        formatted_responses[model] = [
            BatchModelResponseItem(
                question_index=i,
                text=r.text,
                latency_ms=r.latency_ms,
                tokens=r.tokens,
                success=r.success,
                error=r.error
            )
            for i, r in enumerate(model_responses)
        ]

    return BatchMultiModelResponse(
        responses=formatted_responses,
        models_processed=len(request.models),
        questions_processed=len(request.questions),
        total_duration_ms=total_duration_ms,
        vram_managed=True,
        session_id=session_id,
        physics_state=physics_state.to_dict()
    )
