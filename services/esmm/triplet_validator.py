"""
ESMM - TRIPLET VALIDATOR
========================

Validation stricte des triplets extraits par LLM.

Fonctionnalités:
- Validation Pydantic avec règles strictes
- Nettoyage et normalisation automatique
- Détection des patterns invalides
- Support multilingue (FR/EN)

Author: Lyra-ACE ESMM Protocol
"""
from __future__ import annotations

import re
import unicodedata
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator, model_validator

from .prompts import CANONICAL_RELATIONS, normalize_relation, is_canonical_relation

logger = logging.getLogger(__name__)


# Patterns invalides à rejeter
INVALID_PATTERNS = [
    # Pronoms non résolus
    r"^(il|elle|ils|elles|on|ça|ceci|cela|ce|it|this|that|they|he|she)$",
    # Mots trop génériques
    r"^(chose|truc|machin|thing|stuff|something|anything)$",
    # Mots vides
    r"^(le|la|les|un|une|des|de|du|the|a|an)$",
    # Caractères spéciaux seuls
    r"^[<>{}[\]|\\\/]+$",
    # Nombres seuls
    r"^\d+$",
]

# Regex compilées pour performance
COMPILED_INVALID = [re.compile(p, re.IGNORECASE) for p in INVALID_PATTERNS]


@dataclass
class ValidationResult:
    """Résultat de validation d'un triplet."""
    valid: bool
    triplet: Optional[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    corrected: bool = False


class ExtractedTriplet(BaseModel):
    """
    Modèle Pydantic pour un triplet extrait.

    Validation stricte avec normalisation automatique.
    """
    subject: str = Field(..., min_length=2, max_length=100)
    relation: str = Field(..., min_length=2, max_length=50)
    object: str = Field(..., min_length=2, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator('subject', 'object', mode='before')
    @classmethod
    def clean_concept(cls, v: str) -> str:
        """Nettoie et normalise un concept."""
        if not isinstance(v, str):
            v = str(v)

        # Strip et lowercase
        v = v.strip().lower()

        # Supprimer les guillemets
        v = v.strip('"\'`')

        # Normaliser les espaces
        v = re.sub(r'\s+', ' ', v)

        # Supprimer les caractères de contrôle
        v = ''.join(c for c in v if unicodedata.category(c) != 'Cc')

        return v

    @field_validator('subject', 'object')
    @classmethod
    def validate_concept(cls, v: str) -> str:
        """Valide qu'un concept n'est pas un pattern invalide."""
        # Vérifier les patterns invalides
        for pattern in COMPILED_INVALID:
            if pattern.match(v):
                raise ValueError(f"Invalid concept pattern: '{v}'")

        # Vérifier la longueur après nettoyage
        if len(v) < 2:
            raise ValueError(f"Concept too short: '{v}'")

        if len(v) > 100:
            raise ValueError(f"Concept too long: '{v}'")

        # Vérifier les caractères interdits
        if re.search(r'[<>{}[\]|\\]', v):
            raise ValueError(f"Invalid characters in concept: '{v}'")

        return v

    @field_validator('relation', mode='before')
    @classmethod
    def normalize_and_validate_relation(cls, v: str) -> str:
        """Normalise la relation vers sa forme canonique."""
        if not isinstance(v, str):
            v = str(v)

        v = v.strip().lower()

        # Normaliser vers forme canonique
        normalized = normalize_relation(v)

        return normalized

    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v) -> float:
        """Convertit et valide la confiance."""
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                v = 0.5  # Default

        # Clamp entre 0 et 1
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode='after')
    def validate_triplet(self) -> 'ExtractedTriplet':
        """Validation globale du triplet."""
        # Sujet et objet ne doivent pas être identiques
        if self.subject == self.object:
            raise ValueError("Subject and object cannot be identical")

        # Confiance minimum pour triplets valides
        if self.confidence < 0.3:
            raise ValueError(f"Confidence too low: {self.confidence}")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence
        }


class TripletValidator:
    """
    Validateur de triplets avec parsing JSON robuste.

    Gère les sorties LLM mal formatées et applique
    des corrections automatiques quand possible.
    """

    def __init__(self, min_confidence: float = 0.5):
        """
        Args:
            min_confidence: Confiance minimum pour accepter un triplet
        """
        self.min_confidence = min_confidence
        self._stats = {
            "total_processed": 0,
            "valid": 0,
            "invalid": 0,
            "corrected": 0
        }

    def parse_llm_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse la sortie LLM en liste de triplets bruts.

        Gère plusieurs formats:
        - JSON array valide
        - JSON avec markdown code blocks
        - JSON avec texte avant/après
        - Lignes JSON séparées

        Args:
            output: Sortie brute du LLM

        Returns:
            Liste de dictionnaires triplets
        """
        if not output or not output.strip():
            return []

        output = output.strip()

        # Essayer de trouver un JSON array
        # 1. D'abord, extraire du markdown code block
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output)
        if code_block_match:
            output = code_block_match.group(1).strip()

        # 2. Trouver le premier [ et le dernier ]
        start = output.find('[')
        end = output.rfind(']')

        if start != -1 and end != -1 and end > start:
            json_str = output[start:end + 1]
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # 3. Essayer de parser ligne par ligne (JSONL)
        triplets = []
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        triplets.append(obj)
                except json.JSONDecodeError:
                    continue

        if triplets:
            return triplets

        # 4. Essayer de parser tout comme un objet unique
        try:
            obj = json.loads(output)
            if isinstance(obj, dict):
                return [obj]
            elif isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        logger.warning(f"[TripletValidator] Failed to parse LLM output: {output[:100]}...")
        return []

    def validate_triplet(self, raw_triplet: Dict[str, Any]) -> ValidationResult:
        """
        Valide un triplet brut et retourne le résultat.

        Args:
            raw_triplet: Dictionnaire triplet brut

        Returns:
            ValidationResult avec statut et erreurs
        """
        self._stats["total_processed"] += 1
        errors = []
        warnings = []

        # Vérifier les champs requis
        required = ["subject", "relation", "object"]
        missing = [f for f in required if f not in raw_triplet]
        if missing:
            errors.append(f"Missing fields: {missing}")
            self._stats["invalid"] += 1
            return ValidationResult(
                valid=False,
                triplet=None,
                errors=errors,
                warnings=warnings
            )

        # Ajouter confiance par défaut si manquante
        if "confidence" not in raw_triplet:
            raw_triplet["confidence"] = 0.7
            warnings.append("Default confidence 0.7 applied")

        # Essayer de valider avec Pydantic
        try:
            validated = ExtractedTriplet(**raw_triplet)

            # Vérifier la confiance minimum
            if validated.confidence < self.min_confidence:
                warnings.append(f"Low confidence: {validated.confidence}")

            # Vérifier si la relation a été normalisée
            original_relation = raw_triplet.get("relation", "").lower()
            if original_relation != validated.relation:
                warnings.append(f"Relation normalized: '{original_relation}' -> '{validated.relation}'")

            self._stats["valid"] += 1
            if warnings:
                self._stats["corrected"] += 1

            return ValidationResult(
                valid=True,
                triplet=validated.to_dict(),
                errors=[],
                warnings=warnings,
                corrected=len(warnings) > 0
            )

        except ValueError as e:
            errors.append(str(e))
            self._stats["invalid"] += 1
            return ValidationResult(
                valid=False,
                triplet=None,
                errors=errors,
                warnings=warnings
            )

    def validate_batch(
        self,
        raw_triplets: List[Dict[str, Any]],
        filter_invalid: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[ValidationResult]]:
        """
        Valide un batch de triplets.

        Args:
            raw_triplets: Liste de triplets bruts
            filter_invalid: Si True, ne retourne que les valides

        Returns:
            Tuple (triplets valides, tous les résultats de validation)
        """
        results = []
        valid_triplets = []

        for raw in raw_triplets:
            result = self.validate_triplet(raw)
            results.append(result)

            if result.valid and (not filter_invalid or result.triplet):
                valid_triplets.append(result.triplet)

        logger.info(
            f"[TripletValidator] Batch: {len(valid_triplets)}/{len(raw_triplets)} valid"
        )

        return valid_triplets, results

    def validate_llm_output(
        self,
        llm_output: str,
        filter_invalid: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[ValidationResult]]:
        """
        Parse et valide la sortie complète d'un LLM.

        Args:
            llm_output: Sortie brute du LLM
            filter_invalid: Si True, ne retourne que les valides

        Returns:
            Tuple (triplets valides, tous les résultats)
        """
        raw_triplets = self.parse_llm_output(llm_output)

        if not raw_triplets:
            return [], []

        return self.validate_batch(raw_triplets, filter_invalid)

    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques de validation."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Remet les statistiques à zéro."""
        self._stats = {
            "total_processed": 0,
            "valid": 0,
            "invalid": 0,
            "corrected": 0
        }


def validate_triplet_quick(
    subject: str,
    relation: str,
    object_: str,
    confidence: float = 0.7
) -> Optional[Dict[str, Any]]:
    """
    Validation rapide d'un triplet unique.

    Args:
        subject: Sujet
        relation: Relation
        object_: Objet
        confidence: Confiance

    Returns:
        Dictionnaire triplet validé ou None si invalide
    """
    try:
        triplet = ExtractedTriplet(
            subject=subject,
            relation=relation,
            object=object_,
            confidence=confidence
        )
        return triplet.to_dict()
    except ValueError:
        return None


def extract_and_validate(
    llm_output: str,
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fonction de commodité pour extraction + validation.

    Args:
        llm_output: Sortie brute du LLM
        min_confidence: Confiance minimum

    Returns:
        Liste de triplets valides
    """
    validator = TripletValidator(min_confidence=min_confidence)
    valid_triplets, _ = validator.validate_llm_output(llm_output)
    return valid_triplets
