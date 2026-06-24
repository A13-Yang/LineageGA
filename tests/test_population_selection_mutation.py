import numpy as np

from src.fitness import build_distance_matrix, calculate_tour_length
from src.mutation import swap_mutation
from src.population import calculate_diversity, create_initial_population, elitism
from src.selection import select_parents


def test_initial_population_contains_legal_permutations():
    population = create_initial_population(10, 20, rng=np.random.default_rng(1))

    assert len(population) == 20
    assert len({individual.id for individual in population}) == 20
    for individual in population:
        assert sorted(individual.genes.tolist()) == list(range(10))


def test_swap_mutation_keeps_permutation_legal():
    genes = np.arange(12)
    mutated = swap_mutation(genes, 1.0, rng=np.random.default_rng(2))

    assert sorted(mutated.tolist()) == genes.tolist()
    assert not np.array_equal(mutated, genes)


def test_elitism_and_tournament_selection_use_min_fitness():
    population = create_initial_population(5, 6, rng=np.random.default_rng(3))
    for fitness, individual in enumerate(population):
        individual.fitness = float(fitness)

    assert elitism(population, 2) == population[:2]
    parent_a, parent_b = select_parents(
        population,
        "tournament",
        k=3,
        rng=np.random.default_rng(4),
    )

    assert parent_a in population
    assert parent_b in population


def test_elite_poor_and_random_selection_strategies():
    population = create_initial_population(5, 10, rng=np.random.default_rng(6))
    for fitness, individual in enumerate(population):
        individual.fitness = float(fitness)

    elite_a, elite_b = select_parents(
        population,
        "elite",
        elite_ratio=0.2,
        rng=np.random.default_rng(7),
    )
    poor_a, poor_b = select_parents(
        population,
        "poor",
        elite_ratio=0.2,
        rng=np.random.default_rng(8),
    )
    random_a, random_b = select_parents(
        population,
        "random",
        rng=np.random.default_rng(9),
    )

    assert {elite_a.fitness, elite_b.fitness} <= {0.0, 1.0}
    assert {poor_a.fitness, poor_b.fitness} <= {8.0, 9.0}
    assert random_a in population
    assert random_b in population
    assert random_a is not random_b


def test_diversity_is_zero_for_identical_edge_sets():
    cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    distances = build_distance_matrix(cities)
    population = create_initial_population(4, 2, rng=np.random.default_rng(5))
    for individual in population:
        individual.genes = np.array([0, 1, 2, 3])
        individual.fitness = calculate_tour_length(individual.genes, distances)

    assert calculate_diversity(population) == 0.0
