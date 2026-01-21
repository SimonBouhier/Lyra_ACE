"""
LYRA CLEAN - CHAT API ROUTES
============================

Chat completion endpoints with physics-driven generation.
"""
from __future__ import annotations

import time
import uuid
import traceback
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from app.models import ChatRequest, ChatResponse, ErrorResponse
from app.llm_client import get_ollama_client, OllamaClient
from app.embeddings import get_embeddings
from database import get_db, ISpaceDB
from core.physics import BezierEngine, TimeMapper, PhysicsState
from services import ContextInjector, ConversationMemory, build_system_prompt
from services.consciousness.metrics import ConsciousnessMonitor
from services.consciousness.adaptation import AdaptiveConsciousness
from services.consciousness.memory import get_semantic_memory


router = APIRouter(prefix="/chat", tags=["chat"])


# ============================================================================
# DEPENDENCIES
# ============================================================================

async def get_database() -> ISpaceDB:
    """Dependency: Database instance."""
    return await get_db()


async def get_llm() -> OllamaClient:
    """Dependency: Ollama client instance."""
    return await get_ollama_client()


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    db: ISpaceDB = Depends(get_database),
    llm: OllamaClient = Depends(get_llm)
):
    """
    Send chat message and get LLM response.

    Workflow:
    1. Create/load session
    2. Compute physics state from Bezier trajectory
    3. Extract semantic context from graph
    4. Build enriched prompt with conversation history
    5. Generate LLM response with physics parameters
    6. Log interaction to database
    7. Return response with metadata

    Example Request:
        POST /chat/message
        {
            "text": "What is entropy in information theory?",
            "session_id": "abc-123",  # optional
            "profile": "creative",
            "enable_context": true
        }

    Example Response:
        {
            "text": "Entropy in information theory...",
            "session_id": "abc-123",
            "physics_state": {"t": 0.46, "tau_c": 1.25, "rho": 0.35, "delta_r": 0.12},
            "context": {"keywords": ["entropy", "information"], "latency_ms": 15},
            "latency": {"context_extraction": 15, "llm_generation": 1200, "total": 1250},
            "tokens": {"prompt": 250, "completion": 180, "total": 430}
        }
    """
    start_time = time.time()

    try:
        # ====================================================================
        # STEP 1: SESSION MANAGEMENT
        # ====================================================================

        session_id = request.session_id or str(uuid.uuid4())

        # Check if session exists
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            session_exists = await cursor.fetchone() is not None

        if not session_exists:
            # Create new session
            await db.create_session(
                session_id=session_id,
                profile=request.profile
            )

        # ====================================================================
        # STEP 2: PHYSICS STATE COMPUTATION
        # ====================================================================

        # Load Bezier profile
        profile = await db.get_profile(request.profile)
        if not profile:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{request.profile}' not found"
            )

        engine = BezierEngine.from_profile(profile)

        # Get message count for time mapping
        async with db.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM events
                WHERE session_id = ? AND event_type IN ('user_message', 'assistant_message')
                """,
                (session_id,)
            )
            message_count = (await cursor.fetchone())[0]

        # Map message count → t ∈ [0, 1]
        t = TimeMapper.logarithmic(message_count, max_messages=100)
        physics_state = engine.compute_state(t)

        # ====================================================================
        # STEP 2B: ADAPTIVE CONSCIOUSNESS (Level 2+)
        # ====================================================================
        # Apply adaptations from previous interaction (feedback loop)

        adaptive_adjustments = None
        if request.consciousness_level >= 2:
            try:
                # Retrieve last metrics from session
                last_metrics = await db.get_last_consciousness_metrics(session_id)

                if last_metrics:
                    adapter = AdaptiveConsciousness(level=request.consciousness_level)
                    adaptive_adjustments = adapter.suggest_adjustments(
                        metrics=last_metrics,
                        current_profile={
                            'tau_c': physics_state.tau_c,
                            'rho': physics_state.rho,
                            'delta_r': physics_state.delta_r
                        },
                        session_length=message_count
                    )

                    if adaptive_adjustments:
                        # Apply adjustments to physics_state
                        adapted_params = adapter.apply_adjustments(
                            {
                                'tau_c': physics_state.tau_c,
                                'rho': physics_state.rho,
                                'delta_r': physics_state.delta_r
                            },
                            adaptive_adjustments
                        )
                        # Create new PhysicsState with adapted parameters
                        physics_state = PhysicsState(
                            t=physics_state.t,
                            tau_c=adapted_params['tau_c'],
                            rho=adapted_params['rho'],
                            delta_r=adapted_params['delta_r'],
                            kappa=physics_state.kappa
                        )
                        print(f"[Consciousness] Adaptive adjustments applied: {adaptive_adjustments.get('reason', 'N/A')}")

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Adaptive consciousness failed: {e}")

        # ====================================================================
        # STEP 3: CONVERSATION HISTORY
        # ====================================================================

        memory = ConversationMemory(db)
        conversation_history = await memory.get_recent_messages(
            session_id=session_id,
            max_messages=request.max_history
        )
        print(f"[Chat] Session {session_id[:8]}... - Retrieved {len(conversation_history)} history messages (max_history={request.max_history})")

        # ====================================================================
        # STEP 4: SEMANTIC CONTEXT EXTRACTION
        # ====================================================================

        injector = ContextInjector(db)

        system_prompt = build_system_prompt(physics_state)

        enriched_prompt = await injector.build_enriched_prompt(
            user_prompt=request.text,
            system_prompt=system_prompt,
            physics_state=physics_state,
            conversation_history=conversation_history,
            enable_context=request.enable_context
        )

        context_latency_ms = enriched_prompt["latency_ms"]

        # ====================================================================
        # STEP 4B: SEMANTIC MEMORY RECALL (PHASE 3)
        # ====================================================================

        memory_echo = None
        if request.consciousness_level >= 3:
            try:
                # Use singleton instance (session-isolated via session_id keys)
                semantic_memory = get_semantic_memory()

                # Encode query with embeddings
                query_embeddings = await get_embeddings(request.text)
                
                # Recall similar messages from session
                recalled = semantic_memory.recall_memory(
                    session_id=session_id,
                    query_embeddings=query_embeddings,
                    current_turn=message_count,
                    top_k=3
                )
                
                # Format for injection into prompt
                if recalled:
                    memory_echo = semantic_memory.format_memory_echo(recalled)
                    
                    # Inject into system context
                    if memory_echo:
                        enriched_prompt["messages"].insert(-1, {
                            "role": "system",
                            "content": memory_echo
                        })
            except Exception as e:
                # Log but don't fail the chat if memory fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Semantic memory recall failed: {e}")

        # ====================================================================
        # STEP 5: LLM GENERATION
        # ====================================================================

        llm_start = time.time()

        llm_response = await llm.chat(
            messages=enriched_prompt["messages"],
            physics_state=physics_state
        )

        llm_latency_ms = llm_response["latency_ms"]
        response_text = llm_response["text"]
        tokens = llm_response["tokens"]

        # ====================================================================
        # STEP 6: DATABASE LOGGING
        # ====================================================================

        # Log user message
        await db.append_event(
            session_id=session_id,
            event_type="user_message",
            role="user",
            content=request.text,
            injected_concepts=(
                enriched_prompt["context"]["query_keywords"]
                if enriched_prompt["context"]
                else None
            ),
            graph_weight=(
                enriched_prompt["context"]["total_weight"]
                if enriched_prompt["context"]
                else 0.0
            )
        )

        # Log assistant message
        event_id = await db.append_event(
            session_id=session_id,
            event_type="assistant_message",
            role="assistant",
            content=response_text,
            latency_ms=llm_latency_ms
        )

        # Log trajectory point
        await db.log_trajectory_point(
            session_id=session_id,
            t_param=physics_state.t,
            tau_c=physics_state.tau_c,
            rho=physics_state.rho,
            delta_r=physics_state.delta_r,
            kappa=physics_state.kappa,
            event_id=event_id
        )

        # ====================================================================
        # STEP 6B: STORE IN SEMANTIC MEMORY (PHASE 3)
        # ====================================================================

        if request.consciousness_level >= 3:
            try:
                # Use singleton instance (session-isolated via session_id keys)
                semantic_memory = get_semantic_memory()

                # Encode and store user message
                message_embeddings = await get_embeddings(request.text)
                semantic_memory.store_memory(
                    session_id=session_id,
                    content=request.text,
                    embeddings=message_embeddings,
                    turn_number=message_count
                )
            except Exception as e:
                # Log but don't fail if memory storage fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Semantic memory storage failed: {e}")

        # ====================================================================
        # STEP 7: CONSCIOUSNESS METRICS (PHASE 1+)
        # ====================================================================
        
        consciousness_metrics = None
        if request.consciousness_level >= 1:
            monitor = ConsciousnessMonitor(level=request.consciousness_level)
            consciousness_metrics = monitor.compute_metrics(
                context_weight=(enriched_prompt.get("context") or {}).get("total_weight", 0.0),
                num_concepts=len((enriched_prompt.get("context") or {}).get("neighbor_concepts", [])),
                physics_state={
                    "tau_c": physics_state.tau_c,
                    "rho": physics_state.rho,
                    "delta_r": physics_state.delta_r
                },
                response_length=len(response_text.split())
            )

        # ====================================================================
        # STEP 7B: STORE CONSCIOUSNESS METRICS (Level 2+)
        # ====================================================================
        # Store metrics for next interaction's adaptive feedback loop

        if consciousness_metrics and request.consciousness_level >= 2:
            try:
                await db.store_consciousness_metrics(
                    session_id=session_id,
                    turn_number=message_count,
                    metrics=consciousness_metrics.dict(),
                    adjustments=adaptive_adjustments
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to store consciousness metrics: {e}")

        # ====================================================================
        # STEP 8: BUILD RESPONSE
        # ====================================================================

        total_latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            text=response_text,
            session_id=session_id,
            model=llm_response.get("model"),
            physics_state=physics_state.to_dict(),
            context=enriched_prompt.get("context"),
            latency={
                "context_extraction": context_latency_ms,
                "llm_generation": llm_latency_ms,
                "total": total_latency_ms
            },
            tokens=tokens,
            consciousness=consciousness_metrics.dict() if consciousness_metrics else None,
            memory_echo=memory_echo
        )

    except HTTPException:
        raise

    except Exception as e:
        # Print full traceback for debugging
        print("[ERROR] Chat endpoint exception:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health")
async def chat_health(llm: OllamaClient = Depends(get_llm)):
    """
    Check chat endpoint health (LLM connectivity).

    Returns:
        {
            "status": "healthy",
            "llm": {"connected": true, "model": "gpt-oss:20b"}
        }
    """
    health = await llm.health_check()

    status = "healthy" if health["connected"] else "unhealthy"

    return {
        "status": status,
        "llm": health
    }
