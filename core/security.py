"""
LYRA-ACE - SECURITY MODULE
==========================

Gestion securisee des secrets et API keys.

Principes:
- Jamais de secrets en dur dans le code
- Variables d'environnement obligatoires
- Validation au demarrage
- Logging sans exposition des valeurs

Usage:
    from core.security import get_api_key, validate_environment

    # Au demarrage
    validate_environment()

    # Pour obtenir une cle
    ollama_url = get_api_key("OLLAMA_URL", default="http://localhost:11434")
"""
from __future__ import annotations

import os
import re
import hashlib
import secrets
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from functools import lru_cache


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SecretConfig:
    """Configuration d'un secret."""
    name: str
    required: bool = False
    default: Optional[str] = None
    validator: Optional[callable] = None
    description: str = ""


# Secrets attendus par l'application
EXPECTED_SECRETS: List[SecretConfig] = [
    SecretConfig(
        name="OLLAMA_URL",
        required=False,
        default="http://localhost:11434",
        description="URL du serveur Ollama"
    ),
    SecretConfig(
        name="OLLAMA_MODEL",
        required=False,
        default="gpt-oss:20b",
        description="Modele Ollama par defaut"
    ),
    SecretConfig(
        name="MISTRAL_API_KEY",
        required=False,
        default=None,
        description="Cle API Mistral (optionnel)"
    ),
    SecretConfig(
        name="OPENAI_API_KEY",
        required=False,
        default=None,
        description="Cle API OpenAI (optionnel)"
    ),
    SecretConfig(
        name="LYRA_SECRET_KEY",
        required=False,
        default=None,
        description="Cle secrete pour signatures (generee si absente)"
    ),
    SecretConfig(
        name="LYRA_DB_ENCRYPTION_KEY",
        required=False,
        default=None,
        description="Cle de chiffrement SQLite (optionnel, pour sqlcipher)"
    ),
]


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_url(value: str) -> bool:
    """Valide une URL."""
    pattern = r'^https?://[\w\-\.]+(:\d+)?(/.*)?$'
    return bool(re.match(pattern, value))


def validate_api_key(value: str) -> bool:
    """Valide le format d'une cle API (longueur minimale)."""
    return len(value) >= 16


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_api_key(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Recupere une cle API depuis les variables d'environnement.

    Args:
        name: Nom de la variable d'environnement
        default: Valeur par defaut si non definie
        required: Leve une exception si absent et required=True

    Returns:
        Valeur de la cle ou default

    Raises:
        EnvironmentError: Si required=True et cle absente
    """
    value = os.environ.get(name)

    if value is None:
        if required:
            raise EnvironmentError(
                f"Required environment variable '{name}' is not set. "
                f"Please set it before starting the application."
            )
        return default

    return value


def get_secret(name: str) -> Optional[str]:
    """
    Recupere un secret configure.

    Cherche d'abord dans les variables d'environnement,
    puis utilise la valeur par defaut si definie.

    Args:
        name: Nom du secret

    Returns:
        Valeur du secret ou None
    """
    # Trouver la config
    config = next((s for s in EXPECTED_SECRETS if s.name == name), None)

    if config is None:
        # Secret non configure, chercher directement
        return os.environ.get(name)

    value = os.environ.get(name)
    if value is None:
        value = config.default

    # Valider si validator present
    if value and config.validator:
        if not config.validator(value):
            raise ValueError(f"Invalid value for secret '{name}'")

    return value


def validate_environment() -> Dict[str, Any]:
    """
    Valide que toutes les variables d'environnement requises sont presentes.

    Returns:
        Dict avec le statut de chaque secret:
        - name: Nom du secret
        - present: True si defini
        - valid: True si valide
        - has_default: True si default disponible

    Raises:
        EnvironmentError: Si un secret requis est manquant
    """
    results = []
    missing_required = []

    for config in EXPECTED_SECRETS:
        value = os.environ.get(config.name)
        present = value is not None
        has_default = config.default is not None

        # Valider
        valid = True
        if present and config.validator:
            valid = config.validator(value)

        result = {
            "name": config.name,
            "present": present,
            "valid": valid,
            "has_default": has_default,
            "required": config.required,
            "description": config.description
        }
        results.append(result)

        if config.required and not present and not has_default:
            missing_required.append(config.name)

    if missing_required:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_required)}"
        )

    return {"secrets": results, "valid": True}


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    Masque un secret pour le logging.

    Args:
        value: Valeur a masquer
        visible_chars: Nombre de caracteres visibles a la fin

    Returns:
        Valeur masquee (ex: "****abcd")
    """
    if not value:
        return "(empty)"

    if len(value) <= visible_chars:
        return "*" * len(value)

    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


@lru_cache(maxsize=1)
def get_or_generate_secret_key() -> str:
    """
    Recupere ou genere la cle secrete de l'application.

    Si LYRA_SECRET_KEY n'est pas definie, genere une cle
    aleatoire (valide uniquement pour cette instance).

    Returns:
        Cle secrete de 64 caracteres hex
    """
    key = os.environ.get("LYRA_SECRET_KEY")

    if key:
        return key

    # Generer une cle aleatoire (warning: non persistee)
    import warnings
    warnings.warn(
        "LYRA_SECRET_KEY not set. Using generated key (not persistent across restarts). "
        "Set LYRA_SECRET_KEY environment variable for production.",
        RuntimeWarning
    )
    return secrets.token_hex(32)


# ============================================================================
# HASHING & TOKENS
# ============================================================================

def hash_value(value: str, salt: Optional[str] = None) -> str:
    """
    Hash une valeur avec SHA-256.

    Args:
        value: Valeur a hasher
        salt: Salt optionnel (utilise secret_key si absent)

    Returns:
        Hash hexadecimal
    """
    if salt is None:
        salt = get_or_generate_secret_key()[:16]

    salted = f"{salt}{value}"
    return hashlib.sha256(salted.encode()).hexdigest()


def generate_session_token() -> str:
    """
    Genere un token de session securise.

    Returns:
        Token de 64 caracteres hex
    """
    return secrets.token_hex(32)


def verify_session_token(token: str) -> bool:
    """
    Verifie la validite basique d'un token.

    Args:
        token: Token a verifier

    Returns:
        True si format valide
    """
    if not token:
        return False

    # Token doit etre hex de 64 caracteres
    if len(token) != 64:
        return False

    try:
        int(token, 16)
        return True
    except ValueError:
        return False


# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================

def is_production() -> bool:
    """
    Verifie si l'application tourne en production.

    Regarde la variable LYRA_ENV ou detecte automatiquement.
    """
    env = os.environ.get("LYRA_ENV", "development").lower()
    return env in ("production", "prod")


def is_debug() -> bool:
    """
    Verifie si le mode debug est active.
    """
    return os.environ.get("LYRA_DEBUG", "").lower() in ("1", "true", "yes")


def get_config_path() -> str:
    """
    Retourne le chemin du fichier de configuration.
    """
    return os.environ.get("LYRA_CONFIG_PATH", "config.yaml")


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_security():
    """
    Initialise le module de securite au demarrage.

    - Valide les variables d'environnement
    - Genere les cles manquantes
    - Log le statut (sans exposer les valeurs)
    """
    from database.pool import get_logger
    logger = get_logger("security")

    try:
        status = validate_environment()

        for secret in status["secrets"]:
            log_data = {
                "secret": secret["name"],
                "present": secret["present"],
                "has_default": secret["has_default"]
            }

            if secret["present"]:
                logger.info("secret_configured", **log_data)
            elif secret["has_default"]:
                logger.debug("secret_using_default", **log_data)
            elif secret["required"]:
                logger.error("secret_missing_required", **log_data)

        # Initialiser la cle secrete
        _ = get_or_generate_secret_key()

        logger.info("security_initialized", production=is_production(), debug=is_debug())

    except EnvironmentError as e:
        logger.error("security_initialization_failed", error=str(e))
        raise
