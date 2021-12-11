import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(subprocess.getoutput("git rev-parse --show-toplevel"))

import AntColonyAlgorithm

try:
    data = pd.read_csv("dataset/distance_matrix.txt", sep='r\s+', header=None)
    best_path_benchmark = np.array(
        [1, 13, 2, 15, 9, 5, 7, 3, 12, 14, 10, 8, 6, 4, 11])
    best_path_benchmark = best_path_benchmark - np.ones(
        len(best_path_benchmark))  # because my index from 0
    print("Best path should be: " + str(best_path_benchmark))
except:
    print("Could not load the benchmark file.")

if __name__ == "__main__":
    try:
        city_coordinates = pd.read_csv("datasets/coordinates.txt",
                                       sep=r'\s+',
                                       header=None)
        city_coordinates.columns = ["x", "y"]
        city_coordinates_dictionary = city_coordinates.to_dict("index")
    except FileNotFoundError:
        print("Could not find the file.")
    except:
        print("Something went wrong.")

    algorithm = AntColonyAlgorithm.AntColonyAlgorithm(
        city_coordinates_dictionary,
        num_of_iterations=10,
        num_of_ants=10,
        decay_rate=0.7)

    # region Simple run of the algorithm
    # path_lengths, paths = algorithm.run_ant_colony_algorithm()
    #
    # algorithm.plot_paths(paths[-5:], city_coordinates_dictionary)
    # plt.plot(path_lengths)
    # plt.title(f"Number of cities: {city_coordinates.shape[0]}")
    # plt.show()
    # endregion

    # region Optimize parameters
    params_and_lengths = algorithm.optimize_parameters()

    print(params_and_lengths)

    tuples_to_plot = [(el['params']['num_of_iterations'], el['params']['decay_rate'], el['length']) for el in
                      params_and_lengths]

    plt.scatter(*zip(*tuples_to_plot))
    # endregion
