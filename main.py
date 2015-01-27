import sys
import traceback

from ants import *

def print_header(dataset_path):
	print("\n------------------------------------------------------------")
	print("                     Results %s                               " % (dataset_path,))
	print("------------------------------------------------------------")

def print_results(Q, beta, alpha, num_iterations, num_ants, evaporation_factor, best_path, best_path_cost):	
	print("\nFor Q = %s, alpha = %s, beta = %s , num_iterations = %s, num_ants = %s, evaporation_factor = %s:" % (Q, beta, alpha, num_iterations, num_ants, evaporation_factor))
	print("Best found path is:")
	print(" ".join(map(lambda x: x[0], best_path)))
	print("\nBest found path cost = %s\n" % (best_path_cost,))

# 
def get_distance_matrix(dataset):
	return list(map(lambda x: list(map(lambda y: int(y), x.split())), dataset))

if __name__ == "__main__":   

	data_path = sys.argv[1]

	distance_matrix = None
	with open(data_path, 'r') as f:
		lines = f.readlines()
		distance_matrix = get_distance_matrix(lines)

	print_header(data_path)

	Q, beta, alpha, num_iterations, num_ants, evaporation_factor = 10, 0.2, 0.3, 10, 5, 0.3
	result = simulation(num_iterations, num_ants, Q, distance_matrix, alpha, beta, evaporation_factor)


	best_path_cost, best_path = min(result, key=lambda x: x[0])
	print_results(Q, beta, alpha, num_iterations, num_ants, evaporation_factor, best_path, best_path_cost)