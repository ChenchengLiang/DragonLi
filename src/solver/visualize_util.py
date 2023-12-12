import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.graph_objects as go
from graphviz import Digraph
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Removes the limit on image size

show_limited_html_nodes=True
show_html_nodes=20

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

def visualize_path_png(nodes, edges, file_path,compress=False):
    dot = Digraph(comment='The Graph')

    # Add nodes
    for node_id, attributes in nodes:
        fillcolor = get_node_color(attributes["status"])
        if attributes["output_to_file"]:
            shape = attributes["shape"]
        else:
            shape = "ellipse"
        back_track_count=attributes["back_track_count"]
        dot.node(str(node_id),label=f"{node_id}:{back_track_count}",style = 'filled',fillcolor=fillcolor,shape=shape)

    # Add edges
    for source, target, attributes in edges:
        dot.edge(str(source), str(target))

    # Save the dot file and render as a PNG
    file_name=file_path.replace(".eq","")
    dot.render(file_name, format='jpg', cleanup=True)
    if compress==True:
        compress_image(file_name + '.jpg', file_name + '_low_quality.jpg', quality=5, resize_factor=0.5)

    print("Graph saved to", file_name + '.jpg')


def compress_image(input_path, output_path, quality=85, resize_factor=1):
    """
    Compresses an image.

    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param quality: Quality for the output image (1-100). Lower means more compression.
    :param resize_factor: Factor to resize the image. 1 means no resize, less than 1 to reduce size.
    """
    # Open the image
    with Image.open(input_path) as img:
        # Optionally resize the image
        if resize_factor != 1:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size, Image.ANTIALIAS)

        # Save the image with reduced quality
        img.save(output_path, 'JPEG', quality=quality, optimize=True)


def visualize_path_html(nodes, edges, file_path):
    '''
    inputs:
        nodes = [("1", {"label": "1","status":None}), ("2", {"label": "2","status":None}), ("3", {"label": "3","status":None})]
        edges = [("1", "2", {'label': 'A'}),
                 ("3", "1", {'label': 'B'}),
                 ("3", "3", {'label': 'C'})]
    '''
    G = nx.DiGraph()
    if show_limited_html_nodes==True:
        show_nodes=nodes[:show_html_nodes]
        show_node_index = [n[0] for n in show_nodes]
        show_edges=[]
        for e in edges:
            if e[0] in show_node_index and e[1] in show_node_index:
                show_edges.append(e)

        G.add_nodes_from(show_nodes)
        G.add_edges_from(show_edges)
    else:
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
    from src.solver.DataTypes import Variable, Terminal, Operator,SeparateSymbol
    equation_graph_node_color_map = {Variable: "blue", Terminal: "green", Operator: "black",SeparateSymbol:"yellow"}
    dot = Digraph()
    # Set newrank to true for more control over ranking
    dot.attr(newrank='true')

    # Add nodes
    for node in nodes:
        dot.node(str(node.id), label=node.content,color=equation_graph_node_color_map[node.type])

    # Create a subgraph for the legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(rank='source')
        legend.attr(label='Legend', color='white')
        legend.attr(color='black', style='solid')  # Set border color and style
        # For each node type, add a legend entry
        for k,v in equation_graph_node_color_map.items():  # add other shapes as needed
            legend.node(v, color=v, label= k.__name__)

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
