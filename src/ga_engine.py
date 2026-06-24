"""Complete GA engine with lineage tracking for TSP experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from src.crossover import order_crossover_pair, pmx_crossover_pair, uniform_crossover_pair
from src.fitness import calculate_tour_length
from src.individual import Individual, create_individual, reset_individual_id_counter
from src.lineage import LineageTracker
from src.mutation import bitflip_mutation, swap_mutation
from src.population import (
    calculate_bitstring_diversity,
    calculate_diversity,
    create_bitstring_population,
    create_initial_population,
    elitism,
)
from src.selection import select_parents


@dataclass(frozen=True)
class GAConfig:
    """Configuration for one GA run."""

    distance_matrix: np.ndarray | None = None
    population_size: int = 100
    n_generations: int = 300
    crossover_type: str = "ox"
    mutation_rate: float = 0.02
    n_elites: int = 2
    selection_strategy: str = "tournament"
    tournament_size: int = 3
    elite_ratio: float = 0.2
    ancestry_prune_threshold: float = 0.001
    snapshot_interval: int | None = 1
    problem_type: str = "tsp"
    chromosome_length: int | None = None
    fitness_function: Callable[[np.ndarray], float] | None = None
    maximize: bool | None = None


@dataclass(frozen=True)
class IndividualSnapshot:
    """Compact per-generation individual record."""

    id: int
    generation: int
    fitness: float
    lc: float
    parent_ids: tuple[int, ...]
    ancestry: dict[int, float]
    genes: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GenerationRecord:
    """Aggregate metrics recorded for one generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity: float
    avg_lc: float
    lc_fitness_correlation: float
    upset_offspring_rate: float
    best_individual_id: int
    individuals: list[IndividualSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentResult:
    """Result returned by one complete GA run."""

    config: GAConfig
    seed: int | None
    history: list[GenerationRecord]
    final_best_individual: Individual
    final_population: list[Individual]
    elapsed_time: float
    founder_qualities: dict[int, float]


def is_upset_offspring(
    child_fitness: float,
    parent_a: Individual,
    parent_b: Individual,
    *,
    maximize: bool = False,
) -> bool:
    """Return True when a child beats the average fitness of its parents."""
    parent_mean = (parent_a.fitness + parent_b.fitness) / 2.0
    return child_fitness > parent_mean if maximize else child_fitness < parent_mean


class GAEngine:
    """Run a complete lineage-aware GA."""

    def __init__(self, config: GAConfig) -> None:
        self.config = config
        self._validate_config()

    def run(self, seed: int | None = None) -> ExperimentResult:
        """Execute one full GA run and return generation history."""
        start_time = perf_counter()
        reset_individual_id_counter()
        rng = np.random.default_rng(seed)

        population = self._create_initial_population(rng)
        self._evaluate_population(population)

        lineage_tracker = LineageTracker(
            prune_threshold=self.config.ancestry_prune_threshold,
        )
        founder_qualities = lineage_tracker.compute_founder_quality(
            population,
            maximize=self._maximize,
        )
        for individual in population:
            individual.ancestry = {individual.id: 1.0}
            individual.lc = lineage_tracker.compute_lc(individual.ancestry)

        history = [
            self._build_record(
                generation=0,
                population=population,
                upset_offspring_rate=0.0,
            )
        ]

        for generation in range(1, self.config.n_generations + 1):
            population, upset_rate = self._next_generation(
                population,
                generation=generation,
                lineage_tracker=lineage_tracker,
                rng=rng,
            )
            history.append(
                self._build_record(
                    generation=generation,
                    population=population,
                    upset_offspring_rate=upset_rate,
                )
            )

        final_best = self._best_individual(population)
        return ExperimentResult(
            config=self.config,
            seed=seed,
            history=history,
            final_best_individual=final_best,
            final_population=list(population),
            elapsed_time=perf_counter() - start_time,
            founder_qualities=founder_qualities,
        )

    def _next_generation(
        self,
        population: list[Individual],
        *,
        generation: int,
        lineage_tracker: LineageTracker,
        rng: np.random.Generator,
    ) -> tuple[list[Individual], float]:
        elites = [
            elite.copy_with(id=elite.id, generation=generation)
            for elite in elitism(
                population,
                self.config.n_elites,
                maximize=self._maximize,
            )
        ]
        offspring_needed = self.config.population_size - len(elites)
        offspring: list[Individual] = []
        upset_count = 0
        offspring_count = 0

        while len(offspring) < offspring_needed:
            parent_a, parent_b = select_parents(
                population,
                self.config.selection_strategy,
                rng=rng,
                k=self.config.tournament_size,
                elite_ratio=self.config.elite_ratio,
                maximize=self._maximize,
            )
            child_gene_pair = self._crossover(parent_a.genes, parent_b.genes, rng=rng)
            for child_genes in child_gene_pair:
                if len(offspring) >= offspring_needed:
                    break

                ancestry = lineage_tracker.compute_offspring_ancestry(parent_a, parent_b)
                lc = lineage_tracker.compute_lc(ancestry)
                pre_mutation_fitness = self._evaluate_genes(child_genes)
                if is_upset_offspring(
                    pre_mutation_fitness,
                    parent_a,
                    parent_b,
                    maximize=self._maximize,
                ):
                    upset_count += 1
                offspring_count += 1

                mutated_genes = self._mutate(
                    child_genes,
                    rng=rng,
                )
                offspring.append(
                    create_individual(
                        mutated_genes,
                        fitness=self._evaluate_genes(mutated_genes),
                        generation=generation,
                        parent_ids=(parent_a.id, parent_b.id),
                        ancestry=ancestry,
                        lc=lc,
                    )
                )

        upset_rate = 0.0 if offspring_count == 0 else upset_count / offspring_count
        return elites + offspring, float(upset_rate)

    def _evaluate_population(self, population: list[Individual]) -> None:
        for individual in population:
            individual.fitness = self._evaluate_genes(individual.genes)

    def _create_initial_population(self, rng: np.random.Generator) -> list[Individual]:
        if self._problem_type == "tsp":
            assert self.config.distance_matrix is not None
            return create_initial_population(
                self.config.distance_matrix.shape[0],
                self.config.population_size,
                rng=rng,
                generation=0,
            )

        assert self.config.chromosome_length is not None
        return create_bitstring_population(
            self.config.chromosome_length,
            self.config.population_size,
            rng=rng,
            generation=0,
        )

    def _evaluate_genes(self, genes: np.ndarray) -> float:
        if self._problem_type == "tsp":
            assert self.config.distance_matrix is not None
            return float(calculate_tour_length(genes, self.config.distance_matrix))
        if self.config.fitness_function is None:
            raise ValueError("fitness_function is required for binary problems")
        return float(self.config.fitness_function(genes))

    def _mutate(
        self,
        genes: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self._problem_type == "tsp":
            return swap_mutation(genes, self.config.mutation_rate, rng=rng)
        return bitflip_mutation(genes, self.config.mutation_rate, rng=rng)

    def _crossover(
        self,
        parent_a_genes: np.ndarray,
        parent_b_genes: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        crossover_type = self.config.crossover_type.lower()
        if crossover_type == "ox":
            return order_crossover_pair(parent_a_genes, parent_b_genes, rng=rng)
        if crossover_type == "pmx":
            return pmx_crossover_pair(parent_a_genes, parent_b_genes, rng=rng)
        if crossover_type in {"uniform", "ux"}:
            return uniform_crossover_pair(parent_a_genes, parent_b_genes, rng=rng)
        raise ValueError(f"unsupported crossover_type: {self.config.crossover_type}")

    def _build_record(
        self,
        *,
        generation: int,
        population: list[Individual],
        upset_offspring_rate: float,
    ) -> GenerationRecord:
        fitnesses = np.asarray([individual.fitness for individual in population], dtype=float)
        lcs = np.asarray([individual.lc for individual in population], dtype=float)
        best = self._best_individual(population)
        snapshots = (
            [self._snapshot(individual) for individual in population]
            if self._should_snapshot(generation)
            else []
        )

        return GenerationRecord(
            generation=generation,
            best_fitness=float(fitnesses.max() if self._maximize else fitnesses.min()),
            avg_fitness=float(fitnesses.mean()),
            worst_fitness=float(fitnesses.min() if self._maximize else fitnesses.max()),
            diversity=self._diversity(population),
            avg_lc=float(lcs.mean()),
            lc_fitness_correlation=_pearson_correlation(lcs, fitnesses),
            upset_offspring_rate=upset_offspring_rate,
            best_individual_id=best.id,
            individuals=snapshots,
        )

    def _snapshot(self, individual: Individual) -> IndividualSnapshot:
        return IndividualSnapshot(
            id=individual.id,
            generation=individual.generation,
            fitness=individual.fitness,
            lc=individual.lc,
            parent_ids=individual.parent_ids,
            ancestry=dict(individual.ancestry),
            genes=tuple(int(gene) for gene in individual.genes.tolist()),
        )

    def _should_snapshot(self, generation: int) -> bool:
        interval = self.config.snapshot_interval
        if interval is None:
            return generation == 0 or generation == self.config.n_generations
        if interval <= 0:
            return False
        return generation % interval == 0 or generation == self.config.n_generations

    def _validate_config(self) -> None:
        problem_type = self._problem_type
        if problem_type == "tsp":
            if self.config.distance_matrix is None:
                raise ValueError("distance_matrix is required for problem_type='tsp'")
            distances = np.asarray(self.config.distance_matrix, dtype=float)
            if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
                raise ValueError("distance_matrix must be a square 2D matrix")
        elif problem_type in {"nk", "trap"}:
            if self.config.chromosome_length is None or self.config.chromosome_length < 1:
                raise ValueError("chromosome_length must be positive for binary problems")
            if self.config.fitness_function is None:
                raise ValueError("fitness_function is required for binary problems")
        else:
            raise ValueError(f"unsupported problem_type: {self.config.problem_type}")

        if self.config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.config.n_generations < 0:
            raise ValueError("n_generations must not be negative")
        if not 0.0 <= self.config.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.config.n_elites < self.config.population_size:
            raise ValueError("n_elites must be at least 0 and less than population_size")
        if self.config.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if not 0.0 < self.config.elite_ratio <= 1.0:
            raise ValueError("elite_ratio must be in the interval (0, 1]")
        if self.config.ancestry_prune_threshold < 0.0:
            raise ValueError("ancestry_prune_threshold must not be negative")
        crossover_type = self.config.crossover_type.lower()
        if problem_type == "tsp" and crossover_type not in {"ox", "pmx"}:
            raise ValueError("TSP supports only ox and pmx crossover")
        if problem_type in {"nk", "trap"} and crossover_type not in {"uniform", "ux"}:
            raise ValueError("binary problems support only uniform crossover")

    @property
    def _problem_type(self) -> str:
        return self.config.problem_type.lower()

    @property
    def _maximize(self) -> bool:
        if self.config.maximize is not None:
            return bool(self.config.maximize)
        return self._problem_type in {"nk", "trap"}

    def _best_individual(self, population: list[Individual]) -> Individual:
        selector = max if self._maximize else min
        return selector(population, key=lambda individual: individual.fitness)

    def _diversity(self, population: list[Individual]) -> float:
        if self._problem_type == "tsp":
            return calculate_diversity(population)
        return calculate_bitstring_diversity(population)


def _pearson_correlation(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2 or np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return 0.0
    correlation = float(np.corrcoef(xs, ys)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0
