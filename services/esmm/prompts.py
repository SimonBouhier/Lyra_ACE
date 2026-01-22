"""
ESMM - FEW-SHOT PROMPTS FOR TRIPLET EXTRACTION
================================================

Templates de prompts avec exemples positifs et négatifs pour
améliorer la qualité de l'extraction de triplets sémantiques.

Stratégie:
1. Exemples positifs: triplets valides et bien formés
2. Exemples négatifs: erreurs communes à éviter
3. Relations canoniques: liste blanche stricte
4. Format de sortie: JSON structuré

Author: Lyra-ACE ESMM Protocol
"""
from __future__ import annotations

from typing import List, Dict, Any

# Relations canoniques autorisées
CANONICAL_RELATIONS = [
    "cause",           # A causes B
    "caused_by",       # A is caused by B
    "is_a",            # A is a type of B
    "has_a",           # A has B as component
    "part_of",         # A is part of B
    "has_part",        # A has B as part
    "related_to",      # General relation
    "similar_to",      # Similarity
    "opposite_of",     # Opposition/antonym
    "implies",         # Logical implication
    "contradicts",     # Logical contradiction
    "supports",        # Evidence support
    "requires",        # Dependency
    "produces",        # Production/output
    "consumes",        # Input/consumption
    "enables",         # Enabling relation
    "prevents",        # Prevention
    "follows",         # Temporal sequence
    "precedes",        # Temporal precedence
    "contains",        # Containment
]

# Template principal pour extraction de triplets
TRIPLET_EXTRACTION_PROMPT = """Tu es un extracteur de connaissances expert. Tu extrais des triplets sémantiques (Sujet, Relation, Objet) du texte fourni.

## RELATIONS CANONIQUES AUTORISÉES (UTILISE UNIQUEMENT CELLES-CI)
{canonical_relations}

## EXEMPLES VALIDES (À IMITER)

Texte: "L'entropie augmente dans un système isolé car l'énergie se disperse."
Triplets:
[
  {{"subject": "entropie", "relation": "related_to", "object": "système isolé", "confidence": 0.9}},
  {{"subject": "énergie", "relation": "cause", "object": "entropie", "confidence": 0.85}}
]

Texte: "La photosynthèse est un processus biologique qui produit de l'oxygène à partir de CO2."
Triplets:
[
  {{"subject": "photosynthèse", "relation": "is_a", "object": "processus biologique", "confidence": 0.95}},
  {{"subject": "photosynthèse", "relation": "produces", "object": "oxygène", "confidence": 0.9}},
  {{"subject": "photosynthèse", "relation": "consumes", "object": "CO2", "confidence": 0.9}}
]

Texte: "Le théorème de Pythagore implique que le carré de l'hypoténuse égale la somme des carrés des côtés."
Triplets:
[
  {{"subject": "théorème de Pythagore", "relation": "implies", "object": "relation carrés côtés", "confidence": 0.95}},
  {{"subject": "hypoténuse", "relation": "part_of", "object": "triangle rectangle", "confidence": 0.8}}
]

## EXEMPLES INVALIDES (À NE JAMAIS FAIRE)

❌ Triplets trop vagues:
  {{"subject": "chose", "relation": "fait", "object": "autre chose"}} - Trop générique
  {{"subject": "ça", "relation": "est", "object": "important"}} - Pronoms non résolus

❌ Relations non canoniques:
  {{"subject": "A", "relation": "interagit_avec", "object": "B"}} - Utilise "related_to" à la place
  {{"subject": "X", "relation": "mène_à", "object": "Y"}} - Utilise "cause" ou "implies"

❌ Opinions et subjectivité:
  {{"subject": "Einstein", "relation": "pensait", "object": "temps relatif"}} - Opinion, pas fait
  {{"subject": "théorie", "relation": "semble", "object": "correcte"}} - Incertain

❌ Informations incomplètes:
  {{"subject": "il", "relation": "cause", "object": "effet"}} - "il" non résolu
  {{"subject": "", "relation": "is_a", "object": "concept"}} - Sujet vide

## RÈGLES STRICTES

1. **Concepts concrets**: Les sujets et objets doivent être des termes précis (2-100 caractères)
2. **Relations canoniques**: Utilise UNIQUEMENT les relations de la liste ci-dessus
3. **Confiance**: Attribue une confiance entre 0.5 (incertain) et 1.0 (certain)
4. **Pas de pronoms**: Résous les références avant d'extraire
5. **Factuel uniquement**: Pas d'opinions, croyances ou spéculations
6. **Français ou Anglais**: Les concepts peuvent être dans les deux langues

## FORMAT DE SORTIE (JSON STRICT)
```json
[
  {{"subject": "concept_sujet", "relation": "relation_canonique", "object": "concept_objet", "confidence": 0.0-1.0}},
  ...
]
```

Si aucun triplet valide n'est extractible, retourne: []

## TEXTE À ANALYSER
{text}

## TRIPLETS EXTRAITS (JSON uniquement, pas d'explication)
"""

# Template pour validation de triplets existants
TRIPLET_VALIDATION_PROMPT = """Tu es un validateur de triplets sémantiques. Évalue si les triplets suivants sont valides et bien formés.

## CRITÈRES DE VALIDATION
1. Sujet et objet non vides (2-100 caractères)
2. Relation dans la liste canonique: {canonical_relations}
3. Pas de pronoms non résolus (il, elle, ça, ceci, cela)
4. Pas d'opinions ou de spéculations
5. Confiance appropriée (0.5-1.0)

## TRIPLETS À VALIDER
{triplets}

## FORMAT DE SORTIE
Pour chaque triplet, indique:
```json
[
  {{"index": 0, "valid": true/false, "reason": "explication si invalide", "corrected": {{...}} ou null}}
]
```
"""

# Template pour génération de relations à partir de concepts
RELATION_GENERATION_PROMPT = """Tu génères des relations sémantiques entre deux concepts donnés.

## RELATIONS CANONIQUES DISPONIBLES
{canonical_relations}

## EXEMPLES
Concepts: "cause" et "effet"
Relations:
[
  {{"relation": "opposite_of", "confidence": 0.9, "bidirectional": true}},
  {{"relation": "implies", "confidence": 0.7, "bidirectional": false}}
]

Concepts: "photosynthèse" et "oxygène"
Relations:
[
  {{"relation": "produces", "confidence": 0.95, "bidirectional": false}}
]

## CONCEPTS À ANALYSER
Concept A: {concept_a}
Concept B: {concept_b}

## RELATIONS POSSIBLES (JSON uniquement)
"""

# Template pour extraction de concepts d'un texte
CONCEPT_EXTRACTION_PROMPT = """Tu extrais les concepts clés d'un texte pour construire un graphe sémantique.

## RÈGLES
1. Extrais les noms, termes techniques, et concepts abstraits
2. Normalise en minuscules sans accents
3. Ignore les mots courants (le, la, de, un, etc.)
4. Fusionne les variantes (ex: "entropies" → "entropie")
5. Maximum 20 concepts par texte

## EXEMPLES

Texte: "L'entropie est une mesure du désordre dans un système thermodynamique."
Concepts: ["entropie", "mesure", "desordre", "systeme thermodynamique"]

Texte: "La théorie de la relativité d'Einstein révolutionna la physique moderne."
Concepts: ["theorie relativite", "einstein", "physique moderne", "revolution"]

## TEXTE À ANALYSER
{text}

## CONCEPTS EXTRAITS (JSON array uniquement)
"""


def get_triplet_extraction_prompt(text: str) -> str:
    """
    Génère le prompt complet pour extraction de triplets.

    Args:
        text: Texte source à analyser

    Returns:
        Prompt formaté avec exemples
    """
    relations_str = ", ".join(CANONICAL_RELATIONS)
    return TRIPLET_EXTRACTION_PROMPT.format(
        canonical_relations=relations_str,
        text=text
    )


def get_triplet_validation_prompt(triplets: List[Dict[str, Any]]) -> str:
    """
    Génère le prompt pour validation de triplets.

    Args:
        triplets: Liste de triplets à valider

    Returns:
        Prompt formaté
    """
    import json
    relations_str = ", ".join(CANONICAL_RELATIONS)
    triplets_str = json.dumps(triplets, ensure_ascii=False, indent=2)
    return TRIPLET_VALIDATION_PROMPT.format(
        canonical_relations=relations_str,
        triplets=triplets_str
    )


def get_relation_generation_prompt(concept_a: str, concept_b: str) -> str:
    """
    Génère le prompt pour trouver des relations entre deux concepts.

    Args:
        concept_a: Premier concept
        concept_b: Deuxième concept

    Returns:
        Prompt formaté
    """
    relations_str = ", ".join(CANONICAL_RELATIONS)
    return RELATION_GENERATION_PROMPT.format(
        canonical_relations=relations_str,
        concept_a=concept_a,
        concept_b=concept_b
    )


def get_concept_extraction_prompt(text: str) -> str:
    """
    Génère le prompt pour extraction de concepts.

    Args:
        text: Texte source

    Returns:
        Prompt formaté
    """
    return CONCEPT_EXTRACTION_PROMPT.format(text=text)


def is_canonical_relation(relation: str) -> bool:
    """
    Vérifie si une relation est dans la liste canonique.

    Args:
        relation: Nom de la relation

    Returns:
        True si canonique
    """
    return relation.lower() in [r.lower() for r in CANONICAL_RELATIONS]


def normalize_relation(relation: str) -> str:
    """
    Normalise une relation vers sa forme canonique.

    Tente de mapper les relations non-canoniques vers des équivalents.

    Args:
        relation: Relation à normaliser

    Returns:
        Relation canonique ou "related_to" par défaut
    """
    relation = relation.lower().strip()

    # Mappings courants
    mappings = {
        # Causalité
        "causes": "cause",
        "leads_to": "cause",
        "results_in": "cause",
        "mene_a": "cause",
        "provoque": "cause",
        "entraine": "cause",

        # Causalité inverse
        "is_caused_by": "caused_by",
        "results_from": "caused_by",
        "comes_from": "caused_by",

        # Hiérarchie
        "type_of": "is_a",
        "kind_of": "is_a",
        "instance_of": "is_a",
        "est_un": "is_a",
        "est_une": "is_a",

        # Composition
        "composed_of": "has_part",
        "consists_of": "has_part",
        "includes": "has_part",
        "component_of": "part_of",
        "element_of": "part_of",
        "membre_de": "part_of",

        # Similarité
        "like": "similar_to",
        "resembles": "similar_to",
        "similar": "similar_to",
        "equivalent_to": "similar_to",

        # Opposition
        "contrary_to": "opposite_of",
        "antonym_of": "opposite_of",
        "contraire_de": "opposite_of",

        # Production
        "creates": "produces",
        "generates": "produces",
        "makes": "produces",
        "outputs": "produces",
        "produit": "produces",

        # Consommation
        "uses": "consumes",
        "needs": "requires",
        "depends_on": "requires",
        "necessite": "requires",

        # Séquence
        "before": "precedes",
        "after": "follows",
        "then": "follows",
        "next": "follows",
        "avant": "precedes",
        "apres": "follows",

        # Implication
        "means": "implies",
        "suggests": "implies",
        "indicates": "implies",

        # Relations génériques
        "associated_with": "related_to",
        "connected_to": "related_to",
        "linked_to": "related_to",
        "relates_to": "related_to",
        "lie_a": "related_to",
    }

    if relation in CANONICAL_RELATIONS:
        return relation

    if relation in mappings:
        return mappings[relation]

    # Fallback
    return "related_to"
