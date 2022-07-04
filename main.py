"""main.py: Code for performing role detection in a community graph. Works with undirected graphs."""

__author__ = "Abhishek Mahadevan Raju"
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Abhishek Mahadevan Raju"
__status__ = "Prototype"

import argparse
import networkx as nx
import logging
import sys

import rolesim

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    # Allowing graph to be imported as required.
    parser = argparse.ArgumentParser("RoleSim algorithm")
    parser.add_argument("--example", default=False, action='store_true', help="If example is true, then the example graph will be run.")
    parser.add_argument("--file", default="example.pkl", help="If example is not true, this pickle file is read by NetworkX.")
    args = parser.parse_args()

    logging.info("Parsing arguments")
    logging.info("-------------------")

    if args.example:
        logging.info("Example is true")
        logging.info("Using example graph")
        # Example Graph

        graph = nx.Graph()
        # graph.add_edge("A", "B")
        # graph.add_edge("B", "C")
        # graph.add_edge("B", "D")
        # graph.add_edge("A", "C")
        # graph.add_edge("B", "E")
        # graph.add_edge("B", "F")
        
        graph.add_edge("A", "B")
        graph.add_edge("A", "D")
        graph.add_edge("C", "D")
        graph.add_edge("A", "C")
        graph.add_edge("A", "E")
        graph.add_edge("E", "F")   
        
        for edge in graph.edges:
            logging.info("\tEdge - " + str(edge))

    else:

        # graph = nx.read_gpickle(args.file)
        logging.info("Reading file - " + args.file)
        try:
            graph = nx.read_gpickle(args.file)

            logging.info("Parsed graph with")
            logging.info("\t number of nodes = " + str(graph.number_of_nodes()))
            logging.info("\t number of edges = " + str(graph.number_of_edges()))
        except Exception as e:
            logging.error("Currently only supporting Pickled networkx graph files. Please pickle beforehand externally and supply filename.")
            logging.error(e)
            exit()

    logging.info("Starting RoleSim algorithm")
    logging.info("-------------------")

    final_graph = rolesim.full_operation(graph)

    print(list(final_graph.nodes))
    print(list(final_graph.edges))
