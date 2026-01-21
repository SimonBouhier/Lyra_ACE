"""
Tests pour le système de deltas du graphe.
"""
import pytest
import asyncio
import time
from database import get_db, GraphDelta, DeltaOperation, DeltaBatch
from database.graph_delta import KappaCalculator, MutationLimitExceededError


@pytest.fixture
async def db():
    """Fixture pour la base de données de test."""
    database = await get_db("data/test_ispace.db")
    yield database
    # Cleanup après tests


@pytest.mark.asyncio
async def test_apply_add_edge_delta(db):
    """Test: Ajout d'une arête via delta."""
    delta = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="test_concept_a",
        target="test_concept_b",
        weight=0.75,
        confidence=0.9,
        model_source="test"
    )

    # D'abord ajouter les noeuds
    await db.apply_delta(GraphDelta(
        operation=DeltaOperation.ADD_NODE,
        source="test_concept_a"
    ))
    await db.apply_delta(GraphDelta(
        operation=DeltaOperation.ADD_NODE,
        source="test_concept_b"
    ))

    # Puis l'arête
    result = await db.apply_delta(delta, session_id="test_session")

    assert result.delta_id is not None
    assert result.new_kappa is not None
    assert result.applied_at is not None


@pytest.mark.asyncio
async def test_rollback_restores_state(db):
    """Test: Le rollback restaure l'état précédent."""
    # Ajouter une arête
    delta = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="rollback_test_a",
        target="rollback_test_b",
        weight=0.5
    )

    await db.apply_delta(GraphDelta(operation=DeltaOperation.ADD_NODE, source="rollback_test_a"))
    await db.apply_delta(GraphDelta(operation=DeltaOperation.ADD_NODE, source="rollback_test_b"))

    timestamp_before = time.time()
    result = await db.apply_delta(delta, session_id="rollback_session")

    # Vérifier que l'arête existe
    neighbors = await db.get_neighbors("rollback_test_a")
    assert any(n["target"] == "rollback_test_b" for n in neighbors)

    # Rollback
    count = await db.rollback_deltas("rollback_session", to_timestamp=timestamp_before)
    assert count >= 1

    # Vérifier que l'arête n'existe plus
    neighbors = await db.get_neighbors("rollback_test_a")
    assert not any(n["target"] == "rollback_test_b" for n in neighbors)


@pytest.mark.asyncio
async def test_kappa_recalculated_after_edge(db):
    """Test: kappa est recalculé après ajout d'arête."""
    delta = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="kappa_test_a",
        target="kappa_test_b",
        weight=0.8
    )

    await db.apply_delta(GraphDelta(operation=DeltaOperation.ADD_NODE, source="kappa_test_a"))
    await db.apply_delta(GraphDelta(operation=DeltaOperation.ADD_NODE, source="kappa_test_b"))

    result = await db.apply_delta(delta)

    assert result.new_kappa is not None
    assert 0.0 <= result.new_kappa <= 1.0


@pytest.mark.asyncio
async def test_mutation_limit_respected(db):
    """Test: La limite de 5% est respectée."""
    stats = await db.get_stats()
    graph_size = stats["concepts"] + stats["relations"]
    max_allowed = int(graph_size * 0.05)

    # Créer un batch trop grand
    deltas = [
        GraphDelta(
            operation=DeltaOperation.ADD_NODE,
            source=f"limit_test_{i}"
        )
        for i in range(max_allowed + 10)
    ]

    batch = DeltaBatch(deltas=deltas, session_id="limit_test")

    with pytest.raises(MutationLimitExceededError):
        await db.apply_delta_batch(batch)


def test_kappa_calculator_ollivier():
    """Test: Formule Ollivier."""
    calc = KappaCalculator(alpha=1.0)

    # Cas simple: deg=2, weight=1
    k = calc.ollivier_approx(degree_u=2, degree_v=2, weight=1.0)
    # kappa = 1/2 + 1/2 - 2/1 = 1 - 2 = -1
    assert k == -1.0


def test_kappa_calculator_jaccard():
    """Test: Formule Jaccard."""
    calc = KappaCalculator(alpha=0.0)

    neighbors_u = {"a", "b", "c"}
    neighbors_v = {"b", "c", "d"}

    k = calc.jaccard_kappa(neighbors_u, neighbors_v)
    # Intersection: {b, c} = 2
    # Union: {a, b, c, d} = 4
    # Jaccard = 2/4 = 0.5
    assert k == 0.5


def test_kappa_calculator_hybrid():
    """Test: Formule hybride."""
    calc = KappaCalculator(alpha=0.5)

    result = calc.compute_hybrid(
        degree_u=3,
        degree_v=3,
        weight=0.8,
        neighbors_u={"a", "b"},
        neighbors_v={"b", "c"}
    )

    assert "kappa_ollivier" in result
    assert "kappa_jaccard" in result
    assert "kappa_hybrid" in result
    assert 0.0 <= result["kappa_hybrid"] <= 1.0


def test_graph_delta_validation():
    """Test: Validation des deltas."""
    # Delta valide pour add_edge
    valid_delta = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="a",
        target="b",
        weight=0.5
    )
    assert valid_delta.validate() is True

    # Delta invalide: add_edge sans target
    invalid_delta = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="a",
        weight=0.5
    )
    assert invalid_delta.validate() is False

    # Delta invalide: add_edge sans weight
    invalid_delta2 = GraphDelta(
        operation=DeltaOperation.ADD_EDGE,
        source="a",
        target="b"
    )
    assert invalid_delta2.validate() is False

    # Delta valide pour add_node (pas besoin de target ni weight)
    valid_node = GraphDelta(
        operation=DeltaOperation.ADD_NODE,
        source="new_concept"
    )
    assert valid_node.validate() is True


def test_delta_batch_validation():
    """Test: Validation de la taille des batches."""
    deltas = [
        GraphDelta(operation=DeltaOperation.ADD_NODE, source=f"node_{i}")
        for i in range(10)
    ]

    batch = DeltaBatch(deltas=deltas, max_mutation_ratio=0.05)

    # Avec un graphe de 100 elements, 5% = 5, donc 10 deltas devraient echouer
    assert batch.validate_batch_size(100) is False

    # Avec un graphe de 500 elements, 5% = 25, donc 10 deltas devraient passer
    assert batch.validate_batch_size(500) is True
