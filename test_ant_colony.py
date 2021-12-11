import numpy as np
import AntColonyAlgorithm
import pytest

city_coordinates_dictionary = \
    {0: {'x': 0.0, 'y': 0.0},
     1: {'x': 1.0, 'y': 0.0},
     2: {'x': 2.0, 'y': 0.0},
     3: {'x': 2.0, 'y': 1.0}}

distance_matrix_expected = np.array([[0, 1.0, 2, np.sqrt(5)],
                                     [1, 0, 1, np.sqrt(2)], [2, 1, 0, 1],
                                     [np.sqrt(5), np.sqrt(2), 1, 0]])
algorithm = AntColonyAlgorithm.AntColonyAlgorithm(city_coordinates_dictionary)

tau = np.array([[0., 0.0001, 0.0001, 0.0001], [0.0001, 0., 0.0001, 0.0001],
                [0.0001, 0.0001, 0., 0.0001], [0.0001, 0.0001, 0.0001, 0.]])

eta = np.array([[0., 1., 0.5, 0.4472136], [1., 0., 1., 0.70710678],
                [0.5, 1., 0., 1.], [0.4472136, 0.70710678, 1., 0.]])

path_expected = [0, 1, 2, 3, 0]


def test_get_distance_matrix_from_city_coordinates():
    distance_matrix = algorithm.get_distance_matrix_from_city_coordinates(
        city_coordinates_dictionary)
    for i in range(4):
        for j in range(4):
            assert distance_matrix[i, j] == pytest.approx(
                distance_matrix_expected[i, j])


def test_probability_from_i_to_j():
    assert algorithm.probability_from_i_to_j(
        1, 0, tau, eta, [0]) == pytest.approx(0.5135543424717)
    assert algorithm.probability_from_i_to_j(0, 1, tau, eta,
                                             [0, 1]) == pytest.approx(0)
    sum_prob = 0
    for j in range(4):
        sum_prob += algorithm.probability_from_i_to_j(j, 0, tau, eta, [0])
    assert sum_prob == pytest.approx(1)


def test_move_ant_k():
    # note: if we don't seed it, then it will be a flaky test due to random nature of the function.
    np.random.seed(300)
    path = algorithm.move_ant_k(0, tau=tau, eta=eta)
    print(path)
    assert all([path_expected[i] == path[i] for i in range(len(path))])
    assert (len(algorithm.move_ant_k(0, tau,
                                     eta)) == algorithm.num_of_cities + 1)


def test_get_tour_length():
    assert algorithm.get_tour_length([0, 1, 2, 3]) == pytest.approx(3)
    assert algorithm.get_tour_length([0, 1, 2, 3,
                                      0]) == pytest.approx(np.sqrt(5) + 3)
    assert algorithm.get_tour_length([0, 3, 2,
                                      1]) == pytest.approx(np.sqrt(5) + 2)


def test_run_algorithm():
    path_lengths, paths = algorithm.run_ant_colony_algorithm()
    assert path_lengths[-1] == pytest.approx(np.sqrt(5) + 3), "The algorithm should've converged to the optimal path."
    assert paths[-1] == path_expected, "The algorithm should've converged to the optimal path."
