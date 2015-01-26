import random 
import numpy

def map_matrix(f, matrix):
	return [[ f(matrix[i][j]) for j in range(len(matrix[i]))] for i in range(len(matrix))]

# Ants must visit each node >= 1 times
def make_generic_ant(Q, distance_matrix, alpha, beta):
	# numpy.vectorize was having issues when it was presented as lambda with if else expression(Q / x) ** beta
	H = numpy.matrix(map_matrix(lambda x: (Q / x) ** beta if x != 0 else 0, distance_matrix))
	num_nodes = len(distance_matrix)
	distance_matrix = numpy.matrix(distance_matrix, dtype=numpy.float)
	# Returns a 4 value tuple: matrix pheromone', Q, cost_of_tour, visited_edges:
	# pheromone'[i][j] = Q / cost_of_tour if i->j edge was visited otherwise 0
	def ant(starting_node, pheromone_matrix):
		pheromone_matrix = numpy.matrix(pheromone_matrix, dtype=numpy.float)
		visited_nodes = [False for _ in range(num_nodes)]
		pheromone_sum_of_visited = 0
		cost_of_tour = 0
		visited_nodes[starting_node] = True
		current_node = starting_node
		visited_edges = []
		# init random number gen
		random_generator = random.Random()

		while not all(visited_nodes):
			for next_node in filter(lambda node: node != current_node, range(num_nodes)):
				node_probability = pheromone_matrix[current_node, next_node] * H[current_node, next_node]
				node_probability = node_probability / (pheromone_sum_of_visited + node_probability)
				node_acceptance_treshold = random_generator.random()

				if node_probability >= node_acceptance_treshold:
					visited_nodes[next_node] = True
					pheromone_sum_of_visited += pheromone_matrix[current_node, next_node] * H[current_node, next_node]
					
					#print ((current_node, next_node), "->", distance_matrix[current_node][next_node])
					
					cost_of_tour += distance_matrix[current_node, next_node]
					visited_edges.append((current_node, next_node))
					current_node = next_node
					break
		# cycle creating edge
		visited_edges.append( (current_node, starting_node) )
		cost_of_tour += distance_matrix[current_node, starting_node]

		pheromone_matrix_2 = numpy.vectorize(lambda *args: 0, otypes=[numpy.float])(pheromone_matrix)
		for (i,j) in visited_edges:
			pheromone_matrix_2[i, j] = Q / cost_of_tour
		return (pheromone_matrix_2, Q, cost_of_tour, visited_edges)
	return ant

# ant_constructor = make_generic_ant(10, [[0.0, 32.0, 21.0], [321.0, 0.0, 121.0], [32.0, 1.0, 0.0]], 0.23, 0.43)
# ant1 = ant_constructor(0, [[0,2,2],[1,0,1],[3,1,0]])
# ant2 = ant_constructor(1, [[0,2,2],[1,0,1],[3,1,0]])
# ant3 = ant_constructor(2, [[0,2,2],[1,0,1],[3,1,0]])

# print(ant1)
# print(ant2)
# print(ant3)


from multiprocessing import Pool

# Returns (updated pheromone_matrix, (best_total_cost, best_tour))
def one_simulation_step(ant_constructor, num_of_ants, pheromone_matrix):
	num_nodes = len(pheromone_matrix)
	starting_nodes = [random.randint(0, num_nodes-1) for _ in range(num_of_ants)]

	#worker_pool = Pool(num_of_ants)

	data = list(map(lambda starting_node: ant_constructor(starting_node, pheromone_matrix), starting_nodes))

	pheromone_matrices = map(lambda x: x[0], data)
	pheromone_matrix = sum(pheromone_matrices)
	total_costs_and_edges = map(lambda x: (x[2], x[3]), data)
	cost, edges = min(total_costs_and_edges, key=lambda x: x[0])
	
	return pheromone_matrix, cost, edges

def evaporate_pheromone(pheromone_matrix, evaporation_factor):

	evaporate = numpy.vectorize(lambda pheromone_ij: pheromone_ij * (1 - evaporation_factor))
	return evaporate(pheromone_matrix)


def simulation(num_iterations, num_of_ants, Q, distance_matrix, alpha, beta, evaporation_factor):
	results = []
	ant_contructor = make_generic_ant(Q, distance_matrix, alpha, beta)
	num_nodes = len(distance_matrix)
	pheromone_matrix = [[random.random() for j in range(num_nodes)] for i in range(num_nodes)]
	
	for _ in range(num_iterations):
		pheromone_matrix, best_total_cost, best_tour = one_simulation_step(ant_contructor, num_of_ants, pheromone_matrix)
		results.append( (best_total_cost, best_tour) )
		evaporate_pheromone(pheromone_matrix, evaporation_factor)
	return results

# simulation1 = simulation(10, 4, 10, [[0, 12, 32], [28, 0, 32], [1, 1, 0]], 0.1, 0.05, 0.4)

# print(simulation1)
