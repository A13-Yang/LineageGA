import numpy as np

from src.crossover import order_crossover, pmx_crossover


def _assert_legal(child, reference):
    assert len(child) == len(reference)
    assert sorted(child.tolist()) == sorted(reference.tolist())
    assert len(set(child.tolist())) == len(child)


def test_order_crossover_produces_legal_permutations():
    rng = np.random.default_rng(123)
    for _ in range(100):
        parent_a = rng.permutation(30)
        parent_b = rng.permutation(30)
        child = order_crossover(parent_a, parent_b, rng=rng)

        _assert_legal(child, parent_a)


def test_pmx_crossover_produces_legal_permutations():
    rng = np.random.default_rng(456)
    for _ in range(100):
        parent_a = rng.permutation(30)
        parent_b = rng.permutation(30)
        child = pmx_crossover(parent_a, parent_b, rng=rng)

        _assert_legal(child, parent_a)
