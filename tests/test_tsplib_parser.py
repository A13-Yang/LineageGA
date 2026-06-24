from src.fitness import calculate_tour_length
from src.tsplib_parser import parse_opt_tour, parse_tsp


def test_parse_eil51_instance_and_optimal_tour():
    instance = parse_tsp("data/tsplib/eil51.tsp")
    tour = parse_opt_tour("data/tsplib/eil51.opt.tour")

    assert instance.name == "eil51"
    assert instance.dimension == 51
    assert instance.cities.shape == (51, 2)
    assert instance.distance_matrix.shape == (51, 51)
    assert calculate_tour_length(tour, instance.distance_matrix) == 426.0


def test_parse_kroa100_instance_and_optimal_tour():
    instance = parse_tsp("data/tsplib/kroA100.tsp")
    tour = parse_opt_tour("data/tsplib/kroA100.opt.tour")

    assert instance.name == "kroA100"
    assert instance.dimension == 100
    assert instance.cities.shape == (100, 2)
    assert instance.distance_matrix.shape == (100, 100)
    assert calculate_tour_length(tour, instance.distance_matrix) == 21282.0
