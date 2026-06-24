import math

import numpy as np
import pytest

from src.individual import create_individual, reset_individual_id_counter
from src.lineage import LineageTracker


def test_design_document_example_reproduces_lc_075():
    reset_individual_id_counter()
    founders = [
        create_individual(np.arange(4), fitness=10.0),
        create_individual(np.arange(4), fitness=20.0),
        create_individual(np.arange(4), fitness=30.0),
        create_individual(np.arange(4), fitness=40.0),
    ]
    tracker = LineageTracker()

    qualities = tracker.compute_founder_quality(founders)
    ancestry_x = {
        founders[0].id: 0.5,
        founders[1].id: 0.25,
        founders[2].id: 0.25,
        founders[3].id: 0.0,
    }

    assert qualities[founders[0].id] == pytest.approx(1.0)
    assert qualities[founders[1].id] == pytest.approx(2.0 / 3.0)
    assert qualities[founders[2].id] == pytest.approx(1.0 / 3.0)
    assert qualities[founders[3].id] == pytest.approx(0.0)
    assert tracker.compute_lc(ancestry_x) == pytest.approx(0.75)


def test_offspring_ancestry_is_normalized_after_merge():
    reset_individual_id_counter()
    parent_a = create_individual([0, 1, 2], ancestry={1: 0.6, 2: 0.4})
    parent_b = create_individual([2, 1, 0], ancestry={2: 0.5, 3: 0.5})
    tracker = LineageTracker({1: 1.0, 2: 0.5, 3: 0.0})

    ancestry = tracker.compute_offspring_ancestry(parent_a, parent_b)

    assert ancestry == pytest.approx({1: 0.3, 2: 0.45, 3: 0.25})
    assert sum(ancestry.values()) == pytest.approx(1.0, abs=1e-10)


def test_pruning_renormalizes_remaining_ancestry():
    tracker = LineageTracker(prune_threshold=0.01)

    pruned = tracker.prune_ancestry({1: 0.99, 2: 0.005, 3: 0.005})

    assert pruned == pytest.approx({1: 1.0})
    assert sum(pruned.values()) == pytest.approx(1.0, abs=1e-10)


def test_founder_individual_gets_self_ancestry():
    reset_individual_id_counter()
    founder = create_individual([0, 1, 2])

    assert founder.ancestry == {founder.id: 1.0}


def test_multigeneration_chain_propagates_ap_recursively():
    reset_individual_id_counter()
    founder_a = create_individual([0, 1, 2], fitness=1.0)
    founder_b = create_individual([0, 2, 1], fitness=2.0)
    founder_c = create_individual([1, 0, 2], fitness=3.0)
    tracker = LineageTracker(prune_threshold=0.0)
    tracker.compute_founder_quality([founder_a, founder_b, founder_c])

    child_ab = create_individual(
        [2, 0, 1],
        parent_ids=(founder_a.id, founder_b.id),
        ancestry=tracker.compute_offspring_ancestry(founder_a, founder_b),
    )
    child_abc = create_individual(
        [2, 1, 0],
        parent_ids=(child_ab.id, founder_c.id),
        ancestry=tracker.compute_offspring_ancestry(child_ab, founder_c),
    )
    child_back_to_a = create_individual(
        [1, 2, 0],
        parent_ids=(child_abc.id, founder_a.id),
        ancestry=tracker.compute_offspring_ancestry(child_abc, founder_a),
    )

    assert child_ab.ancestry == pytest.approx({founder_a.id: 0.5, founder_b.id: 0.5})
    assert child_abc.ancestry == pytest.approx(
        {founder_a.id: 0.25, founder_b.id: 0.25, founder_c.id: 0.5}
    )
    assert child_back_to_a.ancestry == pytest.approx(
        {founder_a.id: 0.625, founder_b.id: 0.125, founder_c.id: 0.25}
    )
    assert sum(child_back_to_a.ancestry.values()) == pytest.approx(1.0, abs=1e-10)


def test_entropy_and_effective_founders_metrics():
    tracker = LineageTracker()
    ancestry = {1: 0.5, 2: 0.25, 3: 0.25}

    assert tracker.compute_ancestry_entropy(ancestry) == pytest.approx(1.5)
    assert tracker.compute_effective_founders(ancestry) == pytest.approx(1.0 / 0.375)
    assert math.isfinite(tracker.compute_effective_founders(ancestry))
