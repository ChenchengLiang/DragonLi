from typing import List
from src.solver.DataTypes import Variable, Terminal, Operator, Node, Edge, Equation, Term, SeparateSymbol, \
    IsomorphicTailSymbol, GlobalTerminalOccurrenceSymbol, GlobalVariableOccurrenceSymbol, EmptyNode
from src.solver.Constants import UNKNOWN

import math
import numpy as np

def concatenate_eqs(eq_list: List[Equation]):
    left_terms = []
    right_terms = []
    for eq in eq_list:
        left_terms.extend(eq.left_terms + [Term(SeparateSymbol("#"))])
        right_terms.extend(eq.right_terms + [Term(SeparateSymbol("#"))])
    return Equation(left_terms[:-1], right_terms[:-1])

def graph_to_gnn_format(nodes: List[Node], edges: List[Edge], label: int = -1, satisfiability=UNKNOWN):
    '''
    output format:
    {"nodes": [0, 1, 2, 3, 4], "node_types": [1, 1, 1, 2, 2],
    "edges": [[1, 2], [2, 3], [3, 0]], "edge_types": [1, 1, 1],
    "label": 1}
    '''
    node_type_to_int_map = {Operator: 0, Terminal: 1, Variable: 2, SeparateSymbol:3, IsomorphicTailSymbol:3,
                            GlobalVariableOccurrenceSymbol:3,GlobalTerminalOccurrenceSymbol:4,EmptyNode:5}
    edge_type_to_int_map = {None: 1}
    graph_dict = {"nodes": [], "node_types": [], "edges": [], "edge_types": [],
                  "label": label,"satisfiability":satisfiability}
    for node in nodes:
        graph_dict["nodes"].append(node.id)
        graph_dict["node_types"].append(node_type_to_int_map[node.type])
    for edge in edges:
        graph_dict["edges"].append([edge.source, edge.target])
        graph_dict["edge_types"].append(edge_type_to_int_map[edge.type])

    return graph_dict


def merge_graphs(eq_node_list_1, eq_edge_list_1, split_node_list_2, split_edge_list_2):
    # Find the maximum ID in the first graph
    max_id = max(node.id for node in eq_node_list_1) + 1

    # Shift the IDs in the second graph's nodes
    id_shift_map = {}
    for node in split_node_list_2:
        old_id = node.id
        node.id += max_id
        id_shift_map[old_id] = node.id

    # Update the edges in the second graph
    for edge in split_edge_list_2:
        edge.source = id_shift_map[edge.source]
        edge.target = id_shift_map[edge.target]

    # Merge the nodes and edges
    merged_node_list = eq_node_list_1 + split_node_list_2
    merged_edge_list = eq_edge_list_1 + split_edge_list_2

    # # Add eq node edge
    # eq_node_edge=Edge(source=0, target=max_id, type=None, content="", label=None)
    # merged_edge_list.append(eq_node_edge)

    return merged_node_list, merged_edge_list

# Function to apply sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)