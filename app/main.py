"""
LYRA CLEAN - MAIN API SERVER
============================

Production-ready FastAPI application.

Features:
- Async endpoints with connection pooling
- CORS middleware for web clients
- Health checks
- Graceful shutdown
- Structured logging

Usage:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, sessions, graph, multimodel
from app.models import HealthResponse, StatsResponse, ErrorResponse
from app.llm_client import get_ollama_client, close_ollama_client, get_multi_model_client, close_multi_model_client
from database import get_db, close_db


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global startup time for uptime tracking
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
    - Initialize database
    - Initialize Ollama client
    - Load configurations

    Shutdown:
    - Close database connections
    - Close HTTP clients
    """
    global _startup_time

    try:
        # STARTUP
        print("="*80)
        print(" LYRA CLEAN - STARTING")
        print("="*80)

        _startup_time = time.time()

        # Initialize database
        print("[Startup] Initializing database...")
        db = await get_db()
        stats = await db.get_stats()
        print(f"[Startup] Database ready: {stats['concepts']} concepts, {stats['relations']} relations")

        # Initialize Ollama client
        print("[Startup] Initializing Ollama client...")
        llm = await get_ollama_client()
        health = await llm.health_check()

        if health["connected"]:
            print(f"[Startup] Ollama ready: {health['model']} ({len(health['models'])} models available)")
        else:
            print(f"[Startup] ⚠️  Ollama not connected: {health.get('error', 'Unknown error')}")

        # Initialize MultiModel client
        print("[Startup] Initializing MultiModel client...")
        multi_client = await get_multi_model_client()
        models = await multi_client.list_available_models()
        print(f"[Startup] MultiModel ready: {len(models)} models available")

        print("="*80)
        print(" LYRA CLEAN - READY")
        print(" API Documentation: http://localhost:8000/docs")
        print("="*80)

        yield  # Application runs here

    except Exception as e:
        print(f"\n[ERROR] Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # SHUTDOWN
        print("\n" + "="*80)
        print(" LYRA CLEAN - SHUTTING DOWN")
        print("="*80)

        try:
            print("[Shutdown] Closing database...")
            await close_db()
        except Exception as e:
            print(f"[Shutdown Error] Database: {e}")

        try:
            print("[Shutdown] Closing Ollama client...")
            await close_ollama_client()
        except Exception as e:
            print(f"[Shutdown Error] Ollama: {e}")

        try:
            print("[Shutdown] Closing MultiModel client...")
            await close_multi_model_client()
        except Exception as e:
            print(f"[Shutdown Error] MultiModel: {e}")

        print("[Shutdown] Cleanup complete")
        print("="*80)


# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Lyra Clean API",
    description="Physics-driven semantic LLM system with Bezier trajectories",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware (allow web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MIDDLEWARE (Request Logging)
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all HTTP requests with timing.

    Format: [METHOD] /path - 200 OK (123ms)
    """
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    print(f"[{request.method}] {request.url.path} - {response.status_code} ({duration_ms:.1f}ms)")

    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 Not Found."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="not_found",
            message=f"Endpoint {request.url.path} not found"
        ).model_dump()
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 Internal Server Error."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            details={"exception": str(exc)}
        ).model_dump()
    )


# ============================================================================
# ROUTERS
# ============================================================================

# Include API routers
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(graph.router)      # Lyra-ACE: Graph mutations
app.include_router(multimodel.router)  # Lyra-ACE: Multi-model generation


# ============================================================================
# STATIC FILES & WEB UI
# ============================================================================

# Mount static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", tags=["ui"])
async def root():
    """
    Serve the web interface (Lyra Lite UI).

    Returns:
        HTML page for the chat interface
    """
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Web UI not found", "message": "index.html is missing from static directory"}
        )


@app.get("/api", tags=["system"])
async def api_root():
    """
    Root endpoint with API information.

    Returns:
        {
            "service": "Lyra Clean API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    """
    return {
        "service": "Lyra Clean API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat/message",
            "sessions": "/sessions",
            "profiles": "/profiles",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Comprehensive health check.

    Checks:
    - Database connectivity
    - Ollama connectivity
    - System resources

    Example:
        GET /health

    Returns:
        {
            "status": "healthy",
            "database": {"connected": true, "concepts": 15234},
            "ollama": {"connected": true, "model": "gpt-oss:20b"},
            "version": "1.0.0",
            "uptime_seconds": 3600.5
        }
    """
    # Database health
    try:
        db = await get_db()
        db_stats = await db.get_stats()
        db_health = {
            "connected": True,
            "concepts": db_stats["concepts"],
            "relations": db_stats["relations"],
            "sessions": db_stats["sessions"],
            "db_size_mb": db_stats["db_size_mb"]
        }
    except Exception as e:
        db_health = {
            "connected": False,
            "error": str(e)
        }

    # Ollama health
    try:
        llm = await get_ollama_client()
        ollama_health = await llm.health_check()
    except Exception as e:
        ollama_health = {
            "connected": False,
            "error": str(e)
        }

    # Overall status
    if db_health["connected"] and ollama_health["connected"]:
        status = "healthy"
    elif db_health["connected"] or ollama_health["connected"]:
        status = "degraded"
    else:
        status = "unhealthy"

    uptime = time.time() - _startup_time

    return HealthResponse(
        status=status,
        database=db_health,
        ollama=ollama_health,
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.get("/stats", response_model=StatsResponse, tags=["system"])
async def system_stats():
    """
    System statistics and performance metrics.

    Example:
        GET /stats

    Returns:
        {
            "database": {
                "concepts": 15234,
                "relations": 245678,
                "sessions": 42,
                "events": 1523
            },
            "performance": {
                "uptime_seconds": 3600.5
            }
        }
    """
    db = await get_db()
    db_stats = await db.get_stats()

    uptime = time.time() - _startup_time

    return StatsResponse(
        database=db_stats,
        performance={
            "uptime_seconds": uptime
        }
    )


# ============================================================================
# MAIN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Dev mode
        log_level="info"
    )
