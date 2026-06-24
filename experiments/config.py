"""Phase 3 experiment configuration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from src.ga_engine import GAConfig
from src.nk_landscape import NKLandscape
from src.trap_function import TrapFunction
from src.tsplib_parser import parse_tsp


DEFAULT_DATA_DIR = Path("data/tsplib")
DEFAULT_RESULT_DIR = Path("data/results")
DEFAULT_CONVERGENCE_THRESHOLDS = (0.05, 0.10, 0.15)


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for one strategy/instance experiment cell."""

    experiment_code: str
    strategy: str
    instance: str
    problem_type: str = "tsp"
    problem_params: dict[str, Any] = field(default_factory=dict)
    population_size: int = 120
    n_generations: int = 300
    crossover_type: str = "ox"
    mutation_rate: float = 0.08
    n_elites: int = 2
    selection_strategy: str = "tournament"
    tournament_size: int = 5
    elite_ratio: float = 0.2
    ancestry_prune_threshold: float = 0.001
    snapshot_interval: int | None = 50
    tsp_path: str | None = None
    opt_tour_path: str | None = None
    result_dir: str = str(DEFAULT_RESULT_DIR)
    convergence_thresholds: tuple[float, ...] = DEFAULT_CONVERGENCE_THRESHOLDS

    @property
    def run_group(self) -> str:
        """Stable group name used in output filenames."""
        return f"{self.strategy}_{self.instance}"

    @property
    def is_maximization(self) -> bool:
        """Return True when larger objective values are better."""
        return self.problem_type.lower() in {"nk", "trap"}

    @property
    def resolved_tsp_path(self) -> Path:
        """Return the TSPLIB instance path."""
        if self.tsp_path is not None:
            return Path(self.tsp_path)
        return DEFAULT_DATA_DIR / f"{self.instance}.tsp"

    @property
    def resolved_opt_tour_path(self) -> Path:
        """Return the TSPLIB optimal-tour path."""
        if self.opt_tour_path is not None:
            return Path(self.opt_tour_path)
        return DEFAULT_DATA_DIR / f"{self.instance}.opt.tour"

    @property
    def resolved_result_dir(self) -> Path:
        """Return the directory where Phase 3 artifacts are written."""
        return Path(self.result_dir)

    def validate(self) -> None:
        """Validate scalar parameters before starting a run."""
        problem_type = self.problem_type.lower()
        if problem_type not in {"tsp", "nk", "trap"}:
            raise ValueError(f"unsupported problem_type: {self.problem_type}")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.n_generations < 0:
            raise ValueError("n_generations must not be negative")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= self.n_elites < self.population_size:
            raise ValueError("n_elites must be at least 0 and less than population_size")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if not 0.0 < self.elite_ratio <= 1.0:
            raise ValueError("elite_ratio must be in the interval (0, 1]")
        if self.ancestry_prune_threshold < 0.0:
            raise ValueError("ancestry_prune_threshold must not be negative")
        if any(threshold < 0.0 for threshold in self.convergence_thresholds):
            raise ValueError("convergence thresholds must not be negative")
        crossover_type = self.crossover_type.lower()
        if problem_type == "tsp" and crossover_type not in {"ox", "pmx"}:
            raise ValueError("TSP experiments support ox or pmx crossover")
        if problem_type in {"nk", "trap"} and crossover_type not in {"uniform", "ux"}:
            raise ValueError("binary experiments support uniform crossover")

    def to_ga_config(self) -> GAConfig:
        """Load/build the benchmark instance and build the core GA engine config."""
        self.validate()
        common = {
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "crossover_type": self.crossover_type,
            "mutation_rate": self.mutation_rate,
            "n_elites": self.n_elites,
            "selection_strategy": self.selection_strategy,
            "tournament_size": self.tournament_size,
            "elite_ratio": self.elite_ratio,
            "ancestry_prune_threshold": self.ancestry_prune_threshold,
            "snapshot_interval": self.snapshot_interval,
            "problem_type": self.problem_type,
            "maximize": self.is_maximization,
        }

        if self.problem_type.lower() == "tsp":
            instance = parse_tsp(self.resolved_tsp_path)
            return GAConfig(distance_matrix=instance.distance_matrix, **common)

        if self.problem_type.lower() == "nk":
            landscape = self._nk_landscape()
            return GAConfig(
                chromosome_length=landscape.n,
                fitness_function=landscape.fitness,
                **common,
            )

        trap = self._trap_function()
        return GAConfig(
            chromosome_length=trap.n_bits,
            fitness_function=trap.fitness,
            **common,
        )

    def _nk_landscape(self) -> NKLandscape:
        params = dict(self.problem_params)
        return NKLandscape(
            n=int(params.get("n", 20)),
            k=int(params.get("k", 2)),
            seed=None if params.get("seed") is None else int(params.get("seed", 0)),
        )

    def _trap_function(self) -> TrapFunction:
        params = dict(self.problem_params)
        return TrapFunction(
            block_size=int(params.get("block_size", 5)),
            n_blocks=int(params.get("n_blocks", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to JSON-compatible primitives."""
        values = asdict(self)
        values["convergence_thresholds"] = list(self.convergence_thresholds)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        """Deserialize an experiment config from JSON-compatible primitives."""
        normalized = dict(values)
        if "convergence_thresholds" in normalized:
            normalized["convergence_thresholds"] = tuple(
                float(value) for value in normalized["convergence_thresholds"]
            )
        if normalized.get("problem_params") is None:
            normalized["problem_params"] = {}
        return cls(**normalized)

    def to_json(self, path: str | Path) -> None:
        """Write the config as stable, human-readable JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load a config from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_experiment_config(
    *,
    experiment_code: str,
    strategy: str,
    selection_strategy: str,
    instance: str = "eil51",
    **overrides: Any,
) -> ExperimentConfig:
    """Build an ExperimentConfig while allowing focused parameter overrides."""
    values = {
        "experiment_code": experiment_code,
        "strategy": strategy,
        "selection_strategy": selection_strategy,
        "instance": instance,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def config_a(instance: str = "eil51", **overrides: Any) -> ExperimentConfig:
    """A: standard tournament breeding."""
    return make_experiment_config(
        experiment_code="A",
        strategy="tournament",
        selection_strategy="tournament",
        instance=instance,
        **overrides,
    )


def config_b(instance: str = "eil51", **overrides: Any) -> ExperimentConfig:
    """B: elite-pool breeding."""
    return make_experiment_config(
        experiment_code="B",
        strategy="elite",
        selection_strategy="elite",
        instance=instance,
        **overrides,
    )


def config_c(instance: str = "eil51", **overrides: Any) -> ExperimentConfig:
    """C: poor-pool breeding."""
    return make_experiment_config(
        experiment_code="C",
        strategy="poor",
        selection_strategy="poor",
        instance=instance,
        **overrides,
    )


def config_d(instance: str = "eil51", **overrides: Any) -> ExperimentConfig:
    """D: random breeding."""
    return make_experiment_config(
        experiment_code="D",
        strategy="random",
        selection_strategy="random",
        instance=instance,
        **overrides,
    )


def predefined_configs(
    instances: tuple[str, ...] = ("eil51", "kroA100"),
    **overrides: Any,
) -> list[ExperimentConfig]:
    """Return the full Phase 3 strategy x instance matrix."""
    factories = (config_a, config_b, config_c, config_d)
    return [
        factory(instance=instance, **overrides)
        for instance in instances
        for factory in factories
    ]


def predefined_nk_configs(
    *,
    n: int = 20,
    k_values: tuple[int, ...] = (0, 2, 5, 10, 19),
    landscape_seed: int = 2026,
    **overrides: Any,
) -> list[ExperimentConfig]:
    """Return the Phase 5 NK strategy x K matrix."""
    factories = (config_a, config_b, config_c, config_d)
    configs: list[ExperimentConfig] = []
    for k in k_values:
        values = {
            "problem_type": "nk",
            "problem_params": {"n": n, "k": k, "seed": landscape_seed + k},
            "crossover_type": "uniform",
            "mutation_rate": 1.0 / n,
            "instance": f"nk_N{n}_K{k}",
        }
        values.update(overrides)
        for factory in factories:
            configs.append(factory(**values))
    return configs


def predefined_trap_configs(
    *,
    block_size: int = 5,
    n_blocks_values: tuple[int, ...] = (5, 10),
    **overrides: Any,
) -> list[ExperimentConfig]:
    """Return the Phase 5 Trap strategy x scale matrix."""
    factories = (config_a, config_b, config_c, config_d)
    configs: list[ExperimentConfig] = []
    for n_blocks in n_blocks_values:
        n_bits = block_size * n_blocks
        values = {
            "problem_type": "trap",
            "problem_params": {"block_size": block_size, "n_blocks": n_blocks},
            "crossover_type": "uniform",
            "mutation_rate": 1.0 / n_bits,
            "instance": f"trap_{n_blocks}",
        }
        values.update(overrides)
        for factory in factories:
            configs.append(factory(**values))
    return configs


def predefined_phase05_configs(
    *,
    nk_n: int = 20,
    nk_k_values: tuple[int, ...] = (0, 2, 5, 10, 19),
    nk_landscape_seed: int = 2026,
    trap_block_size: int = 5,
    trap_n_blocks_values: tuple[int, ...] = (5, 10),
    **overrides: Any,
) -> list[ExperimentConfig]:
    """Return the complete Phase 5 extension matrix."""
    return predefined_nk_configs(
        n=nk_n,
        k_values=nk_k_values,
        landscape_seed=nk_landscape_seed,
        **overrides,
    ) + predefined_trap_configs(
        block_size=trap_block_size,
        n_blocks_values=trap_n_blocks_values,
        **overrides,
    )
