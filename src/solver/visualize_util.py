





import networkx as nx
import matplotlib.pyplot as plt

def visualize_path(nodes,edges):
    '''
    inputs:
        nodes = ["1", "2", "3"]
        edges = [("1", "2", {'label': 'A'}),
                 ("3, "1", {'label': 'B'}),
                 ("3", "3", {'label': 'C'})]
    '''

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    G.add_nodes_from(nodes)

    # Add directed edges with labels
    G.add_edges_from(edges)

    # Get positions for the nodes
    pos = nx.spring_layout(G)

    # Draw the graph with node labels
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=15)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['label'] for u, v in G.edges()}, font_size=12)

    # Display the graph
    plt.show()