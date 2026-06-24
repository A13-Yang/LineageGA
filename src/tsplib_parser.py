"""Small TSPLIB parser for Phase 1 EUC_2D TSP instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fitness import build_distance_matrix


@dataclass(frozen=True)
class TSPInstance:
    """Parsed TSPLIB instance."""

    name: str
    dimension: int
    edge_weight_type: str
    cities: np.ndarray
    distance_matrix: np.ndarray


def _split_header(line: str) -> tuple[str, str] | None:
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip().upper(), value.strip()

    parts = line.split(maxsplit=1)
    if len(parts) == 2 and parts[0].isalpha():
        return parts[0].strip().upper(), parts[1].strip()
    return None


def parse_tsp(path: str | Path) -> TSPInstance:
    """Parse a TSPLIB .tsp file that uses EUC_2D coordinates."""
    path = Path(path)
    metadata: dict[str, str] = {}
    coordinates: list[tuple[int, float, float]] = []
    in_node_section = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "NODE_COORD_SECTION":
            in_node_section = True
            continue

        if not in_node_section:
            header = _split_header(line)
            if header is not None:
                key, value = header
                metadata[key] = value
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"invalid NODE_COORD_SECTION line: {line!r}")
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        coordinates.append((node_id, x, y))

    edge_weight_type = metadata.get("EDGE_WEIGHT_TYPE", "").upper()
    if edge_weight_type != "EUC_2D":
        raise ValueError(f"only EUC_2D is supported, got {edge_weight_type!r}")

    dimension = int(metadata.get("DIMENSION", len(coordinates)))
    if len(coordinates) != dimension:
        raise ValueError(f"expected {dimension} cities, parsed {len(coordinates)}")

    coordinates.sort(key=lambda item: item[0])
    expected_ids = list(range(1, dimension + 1))
    actual_ids = [item[0] for item in coordinates]
    if actual_ids != expected_ids:
        raise ValueError("Phase 1 parser expects city IDs to be contiguous from 1")

    cities = np.asarray([(x, y) for _, x, y in coordinates], dtype=float)
    return TSPInstance(
        name=metadata.get("NAME", path.stem),
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        cities=cities,
        distance_matrix=build_distance_matrix(cities),
    )


def parse_opt_tour(path: str | Path, *, zero_based: bool = True) -> np.ndarray:
    """Parse a TSPLIB .opt.tour file and return the tour as city indices."""
    path = Path(path)
    in_tour_section = False
    tour: list[int] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "TOUR_SECTION":
            in_tour_section = True
            continue
        if not in_tour_section:
            continue

        for token in line.split():
            value = int(token)
            if value == -1:
                in_tour_section = False
                break
            tour.append(value - 1 if zero_based else value)

    if not tour:
        raise ValueError(f"no TOUR_SECTION found in {path}")
    return np.asarray(tour, dtype=int)


def load_distance_matrix(path: str | Path) -> np.ndarray:
    """Parse a .tsp file and return only its precomputed distance matrix."""
    return parse_tsp(path).distance_matrix
