"""
Embedding generation via Ollama mxbai-embed-large (1024D vectors)
==================================================================

Provides async embeddings for semantic memory and similarity operations.
"""

import httpx
from typing import List
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024


async def get_embeddings(text: str) -> List[float]:
    """
    Génère embeddings 1024D via Ollama mxbai-embed-large.
    
    Args:
        text: Texte à encoder
        
    Returns:
        Liste de 1024 floats (normalized L2 distance)
        
    Raises:
        httpx.HTTPError: Si erreur de connexion Ollama
        ValueError: Si le modèle n'est pas disponible
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"Invalid text input: {text}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                },
                follow_redirects=True
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = data.get("embedding", [])
            
            if not embeddings:
                raise ValueError("No embeddings returned from Ollama")
            
            if len(embeddings) != EMBEDDING_DIM:
                logger.warning(
                    f"Expected {EMBEDDING_DIM}D embedding, got {len(embeddings)}D"
                )
            
            return embeddings
            
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_URL}: {e}")
        raise
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.error(
                f"Model {EMBEDDING_MODEL} not found. Install via: ollama pull {EMBEDDING_MODEL}"
            )
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Génère embeddings pour plusieurs textes (optimisé batch).
    
    Args:
        texts: Liste de textes à encoder
        
    Returns:
        Liste de listes de floats (même taille que texts)
    """
    embeddings = []
    for text in texts:
        emb = await get_embeddings(text)
        embeddings.append(emb)
    
    return embeddings
