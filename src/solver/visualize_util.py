import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.graph_objects as go
from graphviz import Digraph



def get_node_color(status):
    if status is None:
        color = 'blue'
    elif status == "SAT":
        color = 'green'
    elif status == "UNSAT":
        color = 'red'
    elif status == "UNKNOWN":
        color = 'yellow'
    else:
        color = "black"
    return color


def visualize_path(nodes, edges, file_path):
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
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=15)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['label'] for u, v in G.edges()}, font_size=12)

    plt.savefig(file_path + ".png")

def visualize_path_png(nodes, edges, file_path):
    dot = Digraph(comment='The Graph')

    # Add nodes
    for node_id, attributes in nodes:
        fillcolor = get_node_color(attributes["status"])
        dot.node(str(node_id),style = 'filled',fillcolor=fillcolor)

    # Add edges
    for source, target, attributes in edges:
        dot.edge(str(source), str(target))

    # Save the dot file and render as a PNG
    file_name=file_path.replace(".eq","")
    dot.render(file_name, format='jpg', cleanup=True)
    print("Graph saved to", file_name + '.jpg')

def visualize_path_html(nodes, edges, file_path):
    '''
    inputs:
        nodes = [("1", {"label": "1","status":None}), ("2", {"label": "2","status":None}), ("3", {"label": "3","status":None})]
        edges = [("1", "2", {'label': 'A'}),
                 ("3", "1", {'label': 'B'}),
                 ("3", "3", {'label': 'C'})]
    '''

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    edge_centers_x = []
    edge_centers_y = []
    edge_texts = []
    arrows = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_texts.append(edge[2]['label'])
        edge_centers_x.append((x0 + x1) / 2)
        edge_centers_y.append((y0 + y1) / 2)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # Arrow annotations
        arrows.append(
            dict(
                ax=x0,
                ay=y0,
                axref='x',
                ayref='y',
                x=x1,
                y=y1,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=1.5,
                arrowcolor='black'
            )
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    annotations = [
        dict(
            x=xc, y=yc,
            xref='x', yref='y',
            text=edge_texts[i],
            showarrow=False,
            font=dict(size=10)
        )
        for i, (xc, yc) in enumerate(zip(edge_centers_x, edge_centers_y))
    ]

    node_x = []
    node_y = []
    node_ids = []
    node_hovertexts = []
    for node, attributes in G.nodes(data=True):
        x, y = pos[node]
        node_hovertexts.append(attributes['label'])
        node_ids.append(node)
        node_x.append(x)
        node_y.append(y)

    # Create a list of colors for each node based on status
    node_colors = []
    for _, attributes in G.nodes(data=True):
        status = attributes.get('status', None)
        color = get_node_color(status)
        node_colors.append(color)

    node_colors[0] = 'black'
    #print("node_colors",node_colors)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_hovertexts,
        marker=dict(
            size=20,
            color=node_colors,  # using the list of colors
            opacity=0.5,
        ),
        text=node_ids,
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph Visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        annotations=annotations+arrows
                    ))

    fig.write_html(file_path + ".html")
    print("Graph in", file_path + ".html")



def draw_graph(nodes, edges,filename="/home/cheli243/Desktop/CodeToGit/string-equation-solver/boosting-string-equation-solving-by-GNNs/Woorpje_benchmarks/examples/visualize"):
    from src.solver.DataTypes import Variable, Terminal, Operator
    equation_graph_node_color_map = {Variable: "blue", Terminal: "green", Operator: "black"}
    dot = Digraph()

    # Add nodes
    for node in nodes:
        dot.node(str(node.id), label=node.content,color=equation_graph_node_color_map[node.type])

    # Add edges
    for edge in edges:
        dot.edge(str(edge.source), str(edge.target),label=edge.content)

    # Render the graph
    # dot.view()

    # Save the DOT representation to a file
    # with open(filename + '.dot', 'w') as f:
    #     f.write(dot.source)

    dot.render(filename=filename+".png", cleanup=True)  # This will create a png file named 'filename.png'
    print("Graph in",filename+".png.pdf")
