import numpy as np

from src.fitness import build_distance_matrix, calculate_tour_length
from src.tsplib_parser import parse_opt_tour, parse_tsp


def test_square_tour_length_is_closed_path():
    cities = np.array(
        [
            [0, 0],
            [3, 0],
            [3, 4],
            [0, 4],
        ],
        dtype=float,
    )
    distances = build_distance_matrix(cities)

    assert calculate_tour_length(np.array([0, 1, 2, 3]), distances) == 14.0


def test_eil51_optimal_tour_length():
    instance = parse_tsp("data/tsplib/eil51.tsp")
    tour = parse_opt_tour("data/tsplib/eil51.opt.tour")

    assert instance.cities.shape == (51, 2)
    assert instance.distance_matrix.shape == (51, 51)
    assert calculate_tour_length(tour, instance.distance_matrix) == 426.0
