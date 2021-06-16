"""RoleSim_v1.py: Code for performing role detection in a community graph. Works with undirected graphs."""

__author__ = "Abhishek Mahadevan Raju"
__credits__ = ["Himanshi Allahabadi"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Abhishek Mahadevan Raju"
__status__ = "Prototype"

# Imports

import networkx as nx
import numpy as np
from numpy import genfromtxt
import pandas as pd

from itertools import combinations
from networkx.algorithms.matching import max_weight_matching
import itertools

import time

def bootstrap():
    # The code was originally meant to analyze a graph of movies from the IMDb dataset, and thus there might be a 
    # few variables defined for this purpose. However, the code should work independent of the dataset.

    '''
    Create 2 dictionaries that convert your graph unique IDs into
    a series of numbers starting at 0.
    '''
    # node_to_index_dict
    # index_to_node_dict


    """Processing of RoleSim begins below"""

    actual_data_graph = nx.Graph()

    # actual_data_graph = Graph created using integers for indices, and corresponding edges.

    actual_data_graph.remove_nodes_from(list(nx.isolates(actual_data_graph)))

    print(actual_data_graph.number_of_nodes())
    print(actual_data_graph.number_of_edges())
    print(actual_data_graph.nodes)
    print(actual_data_graph.edges)

    '''
    The following section creates a sorted list of nodes ordered by degree
    '''


    unsorted_degrees = []
    unsorted_nodes = []

    for node in list(actual_data_graph.nodes):
        unsorted_nodes.append(node)
        unsorted_degrees.append(actual_data_graph.degree(node))

    # print(unsorted_degrees)
    sorted_node_degree_indices = np.argsort(-np.array(unsorted_degrees), kind ='mergesort')
    # print(sorted_node_degree_indices)
    sorted_degrees = np.array(unsorted_degrees)[sorted_node_degree_indices]
    # print(sorted_degrees)
    sorted_nodes = np.array(unsorted_nodes)[sorted_node_degree_indices]
    # print(sorted_nodes)
    return sorted_nodes, actual_data_graph



'''
The following section creates a sorted list of neighbours for the sorted list of nodes
'''

def neigh(sorted_nodes, actual_data_graph):
    neighbor_degree_sorted_list = {}

    for node in sorted_nodes:
        neighbors = list(actual_data_graph.neighbors(node))
        neighbor_degrees = [actual_data_graph.degree(n) for n in neighbors]
        sorted_neighbor_degrees_indices = np.argsort(neighbor_degrees, kind ='mergesort')
        sorted_neighbors = np.array(neighbors)[sorted_neighbor_degrees_indices]
        sorted_neighbor_degrees = np.array(neighbor_degrees)[sorted_neighbor_degrees_indices]
        neighbor_degree_sorted_list[node] = (sorted_neighbors, sorted_neighbor_degrees)

    # print(neighbor_degree_sorted_list)

    sorted_node_list = list(sorted_nodes)
    comb = combinations(sorted_node_list, 2)
    return neighbor_degree_sorted_list

'''
Function to find the maximal matching weight between the neighboring nodes of the two nodes
'''

def get_maximal_matching_weight_from_existing_graph_edges(graph, node_u, node_v, neighbor_degree_sorted_list):
    neighbors_u = list(neighbor_degree_sorted_list[node_u][0])
    neighbors_v = list(neighbor_degree_sorted_list[node_v][0])
    combined = [neighbors_u, neighbors_v]
    all_possible_edges = list(itertools.product(*combined))
    subgraph = nx.Graph()

    for edge in all_possible_edges:
        if graph.has_edge(*edge):
            subgraph.add_edge(*edge)

    weight = max_weight_matching(subgraph)
    return weight


# print(get_maximal_matching_weight_from_existing_graph_edges(actual_data_graph, 310, 1430))
# print(len(get_maximal_matching_weight_from_existing_graph_edges(actual_data_graph, 310,1430)))

'''
Iceberg pruning and initialization step for the actual RoleSim algorithm
'''
# The following hyperparameters are being set to prune a certain threshold of nodes from the given graph.
# This step might take up to 10 minutes depending upon the size of the graph.

theta = 0.9999
alpha = 0.4
beta = 0.2
theta_bar = (theta - beta)/(1 - beta)

similarity_temp_graph_iceberg = nx.Graph()

def insert_node_pair_and_similarity(graph, node_u, node_v, beta, maximal_matching_weight, similarity_temp_graph_iceberg):
    similarity_initial_value = (1 - beta) * (maximal_matching_weight/graph.degree(node_u))  + beta
#     print("Adding edge to similarity matrix -", node_u, " to ", node_v, " with value ", similarity_initial_value)
    similarity_temp_graph_iceberg.add_edge(node_u, node_v, weight=similarity_initial_value)

def permutations(sorted_node_list, actual_data_graph):
    comb = combinations(sorted_node_list, 2)
    graph = actual_data_graph
    return comb, graph

# PART - 1 of Iceberg - creating initial similarity matrix using the pruned combinations.

# A summary of intermediate data stored so far.
# -----------------------------------------------
# sorted_nodes      //List of nodes sorted by descending degree value
# sorted_degrees    //List of degrees of corresponding nodes above
# neighbor_degree_sorted_list //Tuple of neighbouring nodes for the node that has ID as the index of the tuple. 

def cleaning(comb, graph, neighbor_degree_sorted_list):
    pruned_by_one = 0
    pruned_by_two = 0
    pruned_by_three = 0
    inserted_into_graph = 0

    # Begin the process of pruning using the Iceberg idea
    for i in list(comb):
        node_u = i[0]
        node_v = i[1]

        # Rule 1
        if (theta_bar*graph.degree(node_u) > graph.degree(node_v)) or (graph.degree(node_v) > graph.degree(node_u)):
        #   print("Pruned by rule 1", node_u, node_v)
            pruned_by_one += 1
            continue
        else :

            # Rule 3
            neighbor_1_degree_u = neighbor_degree_sorted_list[node_u][1][0]
            neighbor_1_degree_v = neighbor_degree_sorted_list[node_v][1][0]
            m11 = (1 - beta)* (min(neighbor_1_degree_u, neighbor_1_degree_v) / max(neighbor_1_degree_u, neighbor_1_degree_v)) + beta
            if neighbor_1_degree_v <= neighbor_1_degree_u and ((graph.degree(node_v) - 1 + m11) < (theta_bar*graph.degree(node_u))):
    #             print("Pruned by rule 3", node_u, node_v)
                pruned_by_three += 1
                continue

        # Rule 2
        maximal_matching_weight = len(get_maximal_matching_weight_from_existing_graph_edges(graph, node_u, node_v))

        if maximal_matching_weight >= theta_bar*graph.degree(node_u):
            inserted_into_graph += 1
            insert_node_pair_and_similarity(graph, node_u, node_v, beta, maximal_matching_weight, similarity_temp_graph_iceberg)
        else:
            pruned_by_two += 1
    #         print("Pruned by rule 2", node_u, node_v, " with ", maximal_matching_weight)

    print(inserted_into_graph, ' nodes were inserted in the graph.')
    print(pruned_by_one, ' nodes were pruned by rule 1.')
    print(pruned_by_two, ' nodes were pruned by rule 2.')
    print(pruned_by_three, ' nodes were pruned by rule 3.')

    print(similarity_temp_graph_iceberg.number_of_nodes())
    print(similarity_temp_graph_iceberg.number_of_edges())
    print(list(similarity_temp_graph_iceberg.nodes()))
    print(list(similarity_temp_graph_iceberg.edges()))


# PART - 2 of Iceberg - filling up the other combinations of the similarity graph using the auxiliary formula provided by the authors.

def auxiliary(actual_data_graph):
    # Attempting a graph-based architecture for processing iterations of RoleSim. Required deep-copy of graph.
    final_similarity_iceberg_graph = nx.Graph(similarity_temp_graph_iceberg)

    # print(final_similarity_iceberg_graph.number_of_nodes())
    # print(final_similarity_iceberg_graph.number_of_edges())

    list_of_nodes_selected = list(final_similarity_iceberg_graph.nodes)
    # print(list_of_nodes_selected)
    # print(final_similarity_iceberg_graph.edges(data=True))

    skipped_present_edges = 0
    for edgepair in combinations(list_of_nodes_selected, 2):
        u = edgepair[0]
        v = edgepair[1]
        if final_similarity_iceberg_graph.has_edge(*edgepair):
            skipped_present_edges+=1
        else:
            degree_u = actual_data_graph.degree(u)
            degree_v = actual_data_graph.degree(v)
            denominator = max(degree_u, degree_v)
            numerator = min(degree_u, degree_v)
            weight_of_edgepair = ( alpha * (1 - beta)* (numerator / denominator) ) + beta
            final_similarity_iceberg_graph.add_edge(*edgepair, weight=weight_of_edgepair)

    print('Found current edges with weights, ',skipped_present_edges, ' of them.')
    print('Current count of edges, ', final_similarity_iceberg_graph.number_of_edges())
    
    return final_similarity_iceberg_graph

# Code that returns a bunch of maximal matchings
# Black boxish, but basically works by taking different starting points, which aren't greedy.

# Reference URL - https://stackoverflow.com/questions/51933046/python-networkx-get-unique-matching-combinations

def all_maximal_matchings(T):

    maximal_matchings = []
    partial_matchings = [{(u,v)} for (u,v) in T.edges()]

    while partial_matchings:
        # get current partial matching
        m = partial_matchings.pop()
        nodes_m = set(itertools.chain(*m))

        extended = False
        for (u,v) in T.edges():
            if u not in nodes_m and v not in nodes_m:
                extended = True
                # copy m, extend it and add it to the list of partial matchings
                m_extended = set(m)
                m_extended.add((u,v))
                partial_matchings.append(m_extended)

        if not extended and m not in maximal_matchings:
            maximal_matchings.append(m)

    return maximal_matchings


# Test code for above function

# T = nx.Graph()

# T.add_edge('A','B', weight = 0.1)
# T.add_edge('A','C', weight = 0.2)
# T.add_edge('B','D', weight = 0.9)
# T.add_edge('D','A', weight = 200)

# print(all_maximal_matchings(T))

# test = all_maximal_matchings(T)

list_of_nodes_selected_sorted = sorted(list_of_nodes_selected)
# print(list_of_nodes_selected_sorted)

# Parameters for RoleSim are the same as the ones for Iceberg RoleSim initialization.
# The additional parameter delta is the only one required.

delta = 0.01

'''
Beginning Iterative RoleSim algorithm. The algorithm is exponential, and takes increasingly longer time as the number of nodes increases.
Attempt to prune the graph to within 600 nodes (takes 10 minutes per iteration). If nodes are around 250, it takes much lesser time.
'''

iterative_rolesim_graph_initial = nx.Graph(final_similarity_iceberg_graph)

'''
Auxiliary functions for the use of the main RoleSim algorithm.
'''

def check_if_converged_graph(current_iteration, previous_iteration):
    current_sum = current_iteration.size(weight='weight')
    previous_sum = previous_iteration.size(weight='weight')
    diff = np.fabs(current_sum-previous_sum)
    print("Difference between iterations", diff)
    print("Compared Difference between iterations", diff / current_iteration.number_of_edges())
    if diff > (0.01 *current_iteration.number_of_edges()):
        return True
    else:
        return False

def get_weight_of_new_iteration_graph_from_formula(previous_iteration, node_u, node_v, neighbor_degree_sorted_list):
    neighbors_u = list(neighbor_degree_sorted_list[node_u][0])
    #     print(neighbors_u)
    neighbors_v = list(neighbor_degree_sorted_list[node_v][0])
    #     print(neighbors_v)

    neighbors_u = list(set(neighbors_u) & set(list_of_nodes_selected_sorted))
    neighbors_v = list(set(neighbors_v) & set(list_of_nodes_selected_sorted))
    #     print(neighbors_u)
    #     print(neighbors_v)
    combined = [neighbors_u, neighbors_v]
    #     print(combined)
    all_possible_edges = list(itertools.product(*combined))
    subgraph = nx.Graph()
    degree_u = len(neighbors_u)
    degree_v = len(neighbors_v)
    for edge in all_possible_edges:
    #     if temp_matrix_weight_matching_matrix.has_edge(*edge):
        if edge[0] != edge[1]:
            node_x = edge[0]
            node_y = edge[1]
            subgraph.add_edge(*edge, weight = previous_iteration[node_x][node_y]['weight'])
    #     print('Created subgraph')
    #     print(subgraph.nodes())
    #     print(subgraph.edges(data=True))
    #     print('---')

    trialing_using_nx_max_matching = True
    if trialing_using_nx_max_matching:
        # max_weight_matching is a function of networkX that performs the maximal weight matching.
        # By using the edge weights of the edges as the values of RoleSim's previous iteration values,
        # and allowing only the edges that connect neighbors of u to neighbors
        # of v, we allow networkX to perform the calculation for us.
        maximal_matching_possibilities = [max_weight_matching(subgraph)]
    else:
        # This method was found to be mildly suboptimal compared to the method from networkX above.
        maximal_matching_possibilities = all_maximal_matchings(subgraph)
    # print(maximal_matching_possibilities)
    maximum_weight_of_all = 0
    best_possibility = maximal_matching_possibilities[0]
    for possibility in maximal_matching_possibilities:
        current_weight = 0
        for edge in possibility:
            current_weight += subgraph[edge[0]][edge[1]]['weight']
        if current_weight >= maximum_weight_of_all:
            maximum_weight_of_all = current_weight
            best_possibility = possibility

    if degree_u == 0 and degree_v == 0:
        # Edge case, in case neighbors are missing from the pruned set of nodes, which is a certain possibility.
        maximum_weight_value = 0
    else:
        maximum_weight_value = maximum_weight_of_all/max(degree_u, degree_v)
    return maximum_weight_value
# ----------------------------------------------
# Actual Iterative RoleSim algorithm
# ----------------------------------------------

def iterate(iterative_rolesim_graph_initial):
    '''
    The code uses the edge weights of an undirected networkX graph to store the similarity values between nodes.
    Thus, every node is connected to every other node in this graph, and if there are n nodes, then there are
    combination(n, 2) edges (every possible edge)
    '''
    print("Initial Graph of pruned magnitude")
    print(iterative_rolesim_graph_initial.number_of_nodes())
    print(iterative_rolesim_graph_initial.nodes())

    current_iteration = nx.Graph(iterative_rolesim_graph_initial)
    previous_iteration = nx.Graph(iterative_rolesim_graph_initial)

    k = 0

    initial_iteration = True
    while initial_iteration == True or check_if_converged_graph(current_iteration, previous_iteration):
        initial_iteration = False
        k = k+1
        previous_iteration = nx.Graph(current_iteration)
        print('--------------------')
        print('Iteration ', k)
        print('--------------------')
        print('Previous_iteration')
        print(list(previous_iteration.edges(data=True))[:4])

        start_time = time.time()
        for edge in list(current_iteration.edges):
            node_u = edge[0]
            node_v = edge[1]
            current_iteration[node_u][node_v]['weight'] = (1-beta) *get_weight_of_new_iteration_graph_from_formula(previous_iteration, node_u, node_v) + beta

        elapsed_time = time.time() - start_time
        print('Current_iteration')
        print(list(current_iteration.edges(data=True))[:4])
        print('Elapsed time ', elapsed_time)
        print('--------------------')
    #     break
    print('Ended after ', k, ' iterations')

    print('Final iteration')
    print(list(current_iteration.edges(data=True))[:4])
    return current_iteration


# At each point, the first 4 edges of the graph are printed out to ensure that convergence is actually happening.

nx.write_gpickle(current_iteration, "similarity_graph.gpickle")


np.save("movie_nodes_index_id_dict.npy", movie_nodes_index_id_dict)
np.save("movie_nodes_id_index_dict.npy", movie_nodes_id_index_dict)

# Saving the dicts for accessing the IDs during the role generation step and visualization
