import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class AntColonyAlgorithm:
    def __init__(self,
                 city_coordinates_dictionary: Dict[int, Dict[str, float]],
                 alpha: float = 1,
                 beta: float = 1,
                 num_of_ants: int = 20,
                 num_of_iterations: int = 20,
                 decay_rate: float = 0.1,
                 Q: float = 0.1):
        self.city_coordinates_dictionary = city_coordinates_dictionary
        self.distance_matrix = self.get_distance_matrix_from_city_coordinates(
            city_coordinates_dictionary)
        self.num_of_iterations = num_of_iterations
        self.alpha = alpha
        self.beta = beta
        self.num_of_ants = num_of_ants
        self.num_of_cities = self.distance_matrix.shape[0]
        self.decay_rate = decay_rate
        self.Q = Q

    def get_distance_matrix_from_city_coordinates(
            self, city_coordinates_dictionary: Dict[int, Dict[str, float]]):
        """ city_coordinates_dictionary: a dictionary where key is the city index 
            and value are the city coordinates. Example:
            city_coordinates_dictionary = \
            {0: {'x': 0.0, 'y': 0.0},
             1: {'x': 1.0, 'y': 0.0},
             2: {'x': 2.0, 'y': 0.0},
             3: {'x': 2.0, 'y': 1.0}}
        """
        # initialization of distance matrix from the city coordinates
        self.num_of_cities = len(city_coordinates_dictionary)
        distance_matrix_from_city_dict = np.zeros(
            (self.num_of_cities, self.num_of_cities))

        for j in range(self.num_of_cities):
            for i in range(self.num_of_cities):
                distance_matrix_from_city_dict[j][i] = np.sqrt(
                    (city_coordinates_dictionary[j]['x'] - city_coordinates_dictionary[i]['x']) ** 2 + \
                    (city_coordinates_dictionary[j]['y'] - city_coordinates_dictionary[i]['y']) ** 2)

        self.distance_matrix = distance_matrix_from_city_dict

        return distance_matrix_from_city_dict

    def probability_from_i_to_j(self, j: int, i: int, tau, eta, path: List[int]) -> float:
        # assert(i in path)
        if j in set(path):
            return 0

        p = tau[j][i] ** self.alpha * eta[j][i] ** self.beta

        normalization = 0
        for s in set(range(self.num_of_cities)) - set(path):
            normalization += tau[s][i] ** self.alpha * eta[s][i] ** self.beta

        p = p / normalization

        return p

    def move_ant_k(self, starting_city: int, tau, eta) -> List[int]:

        path = [starting_city]
        for _ in range(self.num_of_cities + 1):

            if len(path) == self.num_of_cities:
                path.append(starting_city)
                break
            else:
                available_cities = list(
                    set(range(self.num_of_cities)) - set(path))
                next_city = np.random.choice(available_cities, p= \
                    [self.probability_from_i_to_j(j, path[-1], tau, eta, path) for j in available_cities])
                path.append(next_city)

        # print("Current path: " + str(path))
        return path

    def get_tour_length(self, path: list) -> float:
        """
        Get the length of the path.
        """
        path_length = 0
        for m in range(len(path) - 1):
            path_length += self.distance_matrix[path[m + 1], path[m]]

        return path_length

    def get_Delta_tau_k(self, path: List[int], path_length: float) -> float:
        Delta_tau_k = np.zeros((self.num_of_cities, self.num_of_cities))
        path_as_tuples: List[Tuple[int, int]] = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for j in range(self.num_of_cities):
            for i in range(self.num_of_cities):
                # check if the ant travelled through (i,j)
                if (i, j) in path_as_tuples or (j, i) in path_as_tuples:
                    #                 print((i,j))
                    Delta_tau_k[j][i] = self.Q / path_length

        return Delta_tau_k

    def run_ant_colony_algorithm(self):
        # pheromone matrix
        tau = np.full((self.num_of_cities, self.num_of_cities), 0.0001)
        tau = tau - np.diag(np.diag(tau))

        helper_matrix = np.diag(np.ones(self.num_of_cities))
        eta = 1 / (self.distance_matrix + helper_matrix) - helper_matrix

        paths = []
        path_lengths = []
        for _ in range(self.num_of_iterations):

            delta_tau = np.zeros(tau.shape)
            # for each ant
            for _ in range(self.num_of_ants):
                # move ant_k through all the cities
                path = self.move_ant_k(0, tau, eta)
                # calculate the tour length
                path_length = self.get_tour_length(path)

                delta_tau_k = self.get_Delta_tau_k(path, path_length)
                delta_tau += delta_tau_k

            paths.append(path)
            path_lengths.append(path_length)

            # update the pheromone matrix tau once all the ants completed their journey
            tau = (1 - self.decay_rate) * tau + delta_tau

        # print("Final path: " + str(path))
        # print("Length of final path: " + str(path_length))
        return path_lengths, paths

    def get_path_as_tuples(self, path: List[int]) -> list:
        path_as_tuples = []
        for i in range(len(path) - 1):
            path_as_tuples.append((path[i], path[i + 1]))
        return path_as_tuples

    def plot_paths(self, paths: List[List[int]], city_coordinates_dictionary=None):
        if city_coordinates_dictionary is not None:
            coordinates_dict = city_coordinates_dictionary
        else:
            coordinates_dict = self.city_coordinates_dictionary

        cities_xs = [el['x'] for el in list(coordinates_dict.values())]
        cities_ys = [el['y'] for el in list(coordinates_dict.values())]
        plt.scatter(cities_xs, cities_ys)

        for i, label in enumerate(list(coordinates_dict.keys())):
            plt.annotate(label, (cities_xs[i], cities_ys[i]))

        for i, path in enumerate(paths):
            xs = []
            ys = []
            for el in path:
                xs.append(coordinates_dict[el]['x'])
                ys.append(coordinates_dict[el]['y'])

            plt.plot(xs, ys, label=str(i), alpha=i / len(paths))

        plt.legend()
        plt.show()

    def optimize_parameters(self,
                            alphas=None,
                            betas=None,
                            nums_of_ants=None,
                            nums_of_iterations=None,
                            decay_rates=None,
                            Qs=None):

        # initialize grid search (if not specified by the user)
        if alphas is not None:
            pass
        else:
            alphas = [1.]

        if betas is not None:
            pass
        else:
            betas = [1.]

        if nums_of_ants is not None:
            pass
        else:
            nums_of_ants = [20, 50]

        if nums_of_iterations is not None:
            pass
        else:
            nums_of_iterations = [10, 100, 250]

        if decay_rates is not None:
            pass
        else:
            decay_rates = [0.1, 0.5, 0.8]

        if Qs is not None:
            pass
        else:
            Qs = [0.1]

        params_and_lengths = []
        for alpha in alphas:
            for beta in betas:
                for num_of_ants in nums_of_ants:
                    for num_of_iterations in nums_of_iterations:
                        for decay_rate in decay_rates:
                            for Q in Qs:
                                params = {
                                    'alpha': alpha,
                                    'beta': beta,
                                    'num_of_ants': num_of_ants,
                                    'num_of_iterations': num_of_iterations,
                                    'decay_rate': decay_rate,
                                    'Q': Q
                                }
                    algorithm = self.__class__(
                        self.city_coordinates_dictionary,
                        alpha=params['alpha'],
                        beta=params['beta'],
                        num_of_ants=params['num_of_ants'],
                        num_of_iterations=params['num_of_iterations'],
                        decay_rate=params['decay_rate'],
                        Q=params['Q'])
                    path_lengths, _ = algorithm.run_ant_colony_algorithm()
                    params_and_lengths.append({
                        'params': params,
                        'length': path_lengths[-1]
                    })

        return params_and_lengths
