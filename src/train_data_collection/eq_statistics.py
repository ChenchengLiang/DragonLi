import json
import os
from typing import List
import gc
from src.solver.Constants import bench_folder

from src.solver.Parser import EqParser, Parser
from src.solver.independent_utils import strip_file_name_suffix, create_folder, hash_graph_with_glob_info
import statistics
from tqdm import tqdm
import plotly.graph_objects as go
import networkx as nx

from src.solver.models.Dataset import get_one_dgl_graph
from src.solver.models.utils import load_model
from src.solver.Constants import project_folder
from torch import no_grad, cat, mean
from dgl import batch
from src.solver.DataTypes import Equation, Formula
from src.solver.algorithms.split_equation_utils import _get_global_info
from src.solver.utils import graph_func_map
from src.solver.algorithms import graph_to_gnn_format
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

def main():

    # benchmark_0 = "04_track_DragonLi_test_1_100"
    # model_dict_0={"graph_type":"graph_1", "experiment_id":"510045020715475220", "run_id":"f04ef1f40ef446639e7e2983369dc3db"}
    # folder = f"{bench_folder}/{benchmark_0}/ALL/ALL"
    # final_statistic_file_0 = statistics_for_one_folder(folder, model_dict_0)
    #
    # final_statistic_file_1 = f"{bench_folder}/{benchmark_0}/final_statistic.json"
    # final_statistic_file_2 = f"{bench_folder}/unsatcores_04_track_DragonLi_train_40001_80000_onecore+proof_tree/final_statistic.json"
    # compare_two_folders(final_statistic_file_1, final_statistic_file_2)



    benchmark_1 = "unsatcores_04_track_DragonLi_train_40001_80000_onecore+proof_tree"
    model_dict_1={"graph_type":"graph_3", "experiment_id":"786449904194763400", "run_id":"ec25835b29c948769d3a913783865d3d"}
    folder = f"{bench_folder}/{benchmark_1}/ALL/ALL"
    final_statistic_file_1 = statistics_for_one_folder(folder, model_dict_1)

    benchmark_2 = "04_track_DragonLi_eval_1_1000"
    model_dict_2={"graph_type":"graph_3", "experiment_id":"786449904194763400", "run_id":"ec25835b29c948769d3a913783865d3d"}
    folder = f"{bench_folder}/{benchmark_2}/ALL/ALL"
    final_statistic_file_2 = statistics_for_one_folder(folder, model_dict_2)



    compare_two_folders(final_statistic_file_1, final_statistic_file_2)


def compare_two_folders(final_statistic_file_1, final_statistic_file_2):
    comparison_folder = create_folder(
        f"{os.path.dirname(os.path.dirname(final_statistic_file_1))}/two_benchmark_comparison")
    benchmark_1_name = os.path.basename(os.path.dirname(final_statistic_file_1))
    benchmark_2_name = os.path.basename(os.path.dirname(final_statistic_file_2))

    # load json file final_statistic_file_1 to dict
    with open(final_statistic_file_1, 'r') as file:
        final_statistic_dict_1 = json.load(file)
    # load json file final_statistic_file_2 to dict
    with open(final_statistic_file_2, 'r') as file:
        final_statistic_dict_2 = json.load(file)

    differences_of_two_dict = {}
    for key in final_statistic_dict_1.keys():
        if key in final_statistic_dict_2.keys():
            if isinstance(final_statistic_dict_1[key], dict):
                compare_histograms(key, benchmark_1_name, benchmark_2_name, final_statistic_dict_1[key],
                                   final_statistic_dict_2[key], output_html=f"{comparison_folder}/{key}.html")
            elif isinstance(final_statistic_dict_1[key], list):
                pass
            else:
                differences_of_two_dict[f"abs_difference_{key}"] = abs(
                    final_statistic_dict_1[key] - final_statistic_dict_2[key])
        else:
            print(f"key {key} not match")

    # save differences_of_two_dic to file
    differences_of_two_dict_file = f"{comparison_folder}/differences_of_two_dict.json"
    print(differences_of_two_dict_file)
    with open(differences_of_two_dict_file, "w") as f:
        json.dump(differences_of_two_dict, f, indent=4)

    # get PCA for GNN embedding
    get_two_PCA(benchmark_1_name,benchmark_2_name,final_statistic_dict_1["GNN_embedding_list"], final_statistic_dict_2["GNN_embedding_list"], comparison_folder)

    return differences_of_two_dict_file


def statistics_for_one_folder(folder,model_dict):
    # load gnn model
    gnn_rank_model = load_gnn_model(model_dict)

    # get parser
    parser = Parser(EqParser())

    # get eq file list
    eq_file_list = []
    all_files = os.scandir(folder)
    for file in tqdm(all_files, desc="Processing files"):
        if file.is_file() and file.name.endswith(".eq"):
            eq_file_list.append(file.name)

    # get statistics for each eq file
    statistic_file_name_list = []
    global_dgl_hash_table = {}
    global_dgl_hash_table_hit = 0
    for eq_file in tqdm(eq_file_list, total=len(eq_file_list), desc="Processing eq files"):
        eq_file_path = os.path.join(folder, eq_file)

        # read one eq file
        parsed_content_eq = parser.parse(eq_file_path)
        variable_list: List[str] = [v.value for v in parsed_content_eq["variables"]]
        terminal_list: List[str] = [t.value for t in parsed_content_eq["terminals"]]
        terminal_list.remove("\"\"")  # remove empty string
        eq_list: List[Equation] = parsed_content_eq["equation_list"]
        
        #read one unsatcore file
        unsatcore_file=eq_file_path.replace(".eq",".unsatcore")
        if os.path.exists(unsatcore_file):
            parsed_contend_unsatcore = parser.parse(unsatcore_file)
            unsatcore_eq_list: List[Equation] = parsed_contend_unsatcore["equation_list"]
        else:
            unsatcore_eq_list = []
        

        eq_length_list = []
        variable_occurrence_list = []
        terminal_occurrence_list = []
        number_of_vairables_each_eq_list = []
        number_of_terminals_each_eq_list = []

        # todo get graph embedding for formula
        graph_func = graph_func_map[model_dict["graph_type"]]
        G_list_dgl, dgl_hash_table, dgl_hash_table_hit = _get_G_list_dgl(Formula(eq_list), graph_func,
                                                                         dgl_hash_table=global_dgl_hash_table,
                                                                         dgl_hash_table_hit=global_dgl_hash_table_hit)
        global_dgl_hash_table = dgl_hash_table
        global_dgl_hash_table_hit = dgl_hash_table_hit

        with no_grad():
            # embedding output [n,1,128]
            G_list_embeddings = gnn_rank_model.shared_gnn.embedding(batch(G_list_dgl))

            # concat target output [n,1,256]
            mean_tensor = mean(G_list_embeddings, dim=0)  # [1,128]
            G_embedding = mean_tensor.squeeze(0).tolist()

        # get statistics for each equation
        for eq in eq_list:
            # get equation length
            eq_length_list.append(eq.term_length)

            # get variable occurrence
            variable_occurrence_map = {v: 0 for v in variable_list}
            for v in variable_list:
                variable_occurrence_map[v] += (eq.eq_str).count(v)
            variable_occurrence_list.append(sum(list(variable_occurrence_map.values())))

            # get terminal occurrence
            terminal_occurrence_map = {t: 0 for t in terminal_list}
            for t in terminal_list:
                terminal_occurrence_map[t] += (eq.eq_str).count(t)
            terminal_occurrence_list.append(sum(list(terminal_occurrence_map.values())))

            # get number of variables and terminals
            number_of_vairables_each_eq_list.append(eq.variable_number)
            number_of_terminals_each_eq_list.append(eq.terminal_numbers_without_empty_terminal)

        # summary info
        line_offset = 3
        info_summary_with_id = {i + line_offset: "" for i in range(len(eq_list))}
        for i, (eq, eq_length, viariable_occurrences, terminal_occurrence) in enumerate(
                zip(eq_list, eq_length_list, variable_occurrence_list, terminal_occurrence_list)):
            info_summary_with_id[i + line_offset] = (
                f"length: {eq_length}, variables ({eq.variable_number}): {''.join([v.value for v in eq.variable_list])},"
                f" terminals ({eq.terminal_numbers_without_empty_terminal}): {''.join([t.value for t in eq.termimal_list_without_empty_terminal])},"
                f" variable_occurrence: {viariable_occurrences},"
                f" terminal_occurrence: {terminal_occurrence},"
                f" variable_occurrence_ratio: {viariable_occurrences / eq_length},"
                f" terminal_occurrence_ratio: {terminal_occurrence / eq_length},")

        # get statistics
        statistic_dict = {"number_of_equations": len(eq_list),
                          "number_of_unsatcore_equations": len(unsatcore_eq_list),
                          "number_of_variables": len(variable_list),
                          "number_of_terminals": len(terminal_list),


                          "min_eq_length": min(eq_length_list),
                          "max_eq_length": max(eq_length_list),
                          "average_eq_length": statistics.mean(eq_length_list),
                          "stdev_eq_length": custom_stdev(eq_length_list),

                          "total_variable_occurrence_ratio": sum(variable_occurrence_list) / sum(eq_length_list),
                          "total_terminal_occurrence_ratio": sum(terminal_occurrence_list) / sum(eq_length_list),

                          "min_variable_occurrence": min(variable_occurrence_list),
                          "max_variable_occurrence": max(variable_occurrence_list),
                          "average_variable_occurrence": statistics.mean(variable_occurrence_list),
                          "stdev_variable_occurrence": custom_stdev(variable_occurrence_list),

                          "min_terminal_occurrence": min(terminal_occurrence_list),
                          "max_terminal_occurrence": max(terminal_occurrence_list),
                          "average_terminal_occurrence": statistics.mean(terminal_occurrence_list),
                          "stdev_terminal_occurrence": custom_stdev(terminal_occurrence_list),

                          "info_summary_with_id": info_summary_with_id,
                          "eq_length_list": eq_length_list,
                          "variable_occurrence_list": variable_occurrence_list,
                          "terminal_occurrence_list": terminal_occurrence_list,
                          "number_of_vairables_each_eq_list": number_of_vairables_each_eq_list,
                          "number_of_terminals_each_eq_list": number_of_terminals_each_eq_list,
                          "G_embedding": G_embedding,
                          "eq_embedding_list": G_list_embeddings.tolist()}
        # save statistics to file
        statistic_file_name = f"{strip_file_name_suffix(eq_file_path)}_statistics.json"
        statistic_file_name_list.append(statistic_file_name)
        with open(statistic_file_name, 'w') as file:
            json.dump(statistic_dict, file, indent=4)

    return benchmark_level_statistics(folder, statistic_file_name_list)


def benchmark_level_statistics(folder, statistic_file_name_list):
    # get final_statistic_dict
    final_statistic_dict = {"total_variable_occurrence": 0,
                            "total_terminal_occurrence": 0,
                            "total_eq_symbol": 0,
                            "total_variable_occurrence_ratio": 0,
                            "total_terminal_occurrence_ratio": 0,

                            "min_eq_number_of_problems": 0,
                            "max_eq_number_of_problems": 0,
                            "average_eq_number_of_problems": 0,
                            "stdev_eq_number_of_problems": 0,

                            "min_unsatcore_eq_number_of_problems": 0,
                            "max_unsatcore_eq_number_of_problems": 0,
                            "average_unsatcore_eq_number_of_problems": 0,
                            "stdev_unsatcore_eq_number_of_problems": 0,

                            "min_eq_length": 0,
                            "max_eq_length": 0,
                            "average_eq_length": 0,
                            "stdev_eq_length": 0,

                            "min_variable_occurrence_of_problem": 0,
                            "max_variable_occurrence_of_problem": 0,
                            "average_variable_occurrence_of_problem": 0,
                            "stdev_variable_occurrence_of_problem": 0,

                            "min_terminal_occurrence_of_problem": 0,
                            "max_terminal_occurrence_of_problem": 0,
                            "average_terminal_occurrence_of_problem": 0,
                            "stdev_terminal_occurrence_of_problem": 0,

                            "min_variable_occurrence_of_equation": 0,
                            "max_variable_occurrence_of_equation": 0,
                            "average_variable_occurrence_of_equation": 0,
                            "stdev_variable_occurrence_of_equation": 0,

                            "min_terminal_occurrence_of_equation": 0,
                            "max_terminal_occurrence_of_equation": 0,
                            "average_terminal_occurrence_of_equation": 0,
                            "stdev_terminal_occurrence_of_equation": 0,

                            "min_variable_number_of_equation": 0,
                            "max_variable_number_of_equation": 0,
                            "average_variable_number_of_equation": 0,
                            "stdev_variable_number_of_equation": 0,

                            "min_terminal_number_of_equation": 0,
                            "max_terminal_number_of_equation": 0,
                            "average_terminal_number_of_equation": 0,
                            "stdev_terminal_number_of_equation": 0,

                            "equation_number:number_of_problems": {i: 0 for i in range(1, 101)},
                            "variable_number:number_of_problems": {i: 0 for i in range(0, 27)},
                            "terminal_number:number_of_problems": {i: 0 for i in range(0, 27)},

                            "GNN_embedding_list": []
                            }

    eq_number_list_of_problems = []
    unsatcore_eq_number_list_of_problems=[]
    eq_length_list_of_problems = []
    variable_occurrence_list_of_problems = []
    terminal_occurrence_list_of_problems = []
    variable_occurrence_list_of_all_equations = []
    terminal_occurrence_list_of_all_equations = []
    variable_number_list_of_all_equations = []
    terminal_number_list_of_all_equations = []
    GNN_embedding_list = []
    for statistic_file_name in statistic_file_name_list:
        with open(statistic_file_name, 'r') as file:
            statistic = json.load(file)
            final_statistic_dict["equation_number:number_of_problems"][statistic["number_of_equations"]] += 1
            final_statistic_dict["variable_number:number_of_problems"][statistic["number_of_variables"]] += 1
            final_statistic_dict["terminal_number:number_of_problems"][statistic["number_of_terminals"]] += 1
            final_statistic_dict["total_eq_symbol"] += sum(statistic["eq_length_list"])
            final_statistic_dict["total_variable_occurrence"] += sum(statistic["variable_occurrence_list"])
            final_statistic_dict["total_terminal_occurrence"] += sum(statistic["terminal_occurrence_list"])
            eq_length_list_of_problems.extend(statistic["eq_length_list"])
            eq_number_list_of_problems.append(statistic["number_of_equations"])
            unsatcore_eq_number_list_of_problems.append(statistic["number_of_unsatcore_equations"])

            variable_occurrence_list_of_problems.append(sum(statistic["variable_occurrence_list"]))
            terminal_occurrence_list_of_problems.append(sum(statistic["terminal_occurrence_list"]))
            variable_occurrence_list_of_all_equations.extend(statistic["variable_occurrence_list"])
            terminal_occurrence_list_of_all_equations.extend(statistic["terminal_occurrence_list"])

            variable_number_list_of_all_equations.extend(statistic["number_of_vairables_each_eq_list"])
            terminal_number_list_of_all_equations.extend(statistic["number_of_terminals_each_eq_list"])
            GNN_embedding_list.append(statistic["G_embedding"])


    html_directory=os.path.dirname(os.path.dirname(folder))
    get_one_PCA(GNN_embedding_list, html_directory)


    final_statistic_dict["total_variable_occurrence_ratio"] = final_statistic_dict["total_variable_occurrence"] / \
                                                              final_statistic_dict["total_eq_symbol"]
    final_statistic_dict["total_terminal_occurrence_ratio"] = final_statistic_dict["total_terminal_occurrence"] / \
                                                              final_statistic_dict["total_eq_symbol"]

    final_statistic_dict["min_eq_number_of_problems"] = min(eq_number_list_of_problems)
    final_statistic_dict["max_eq_number_of_problems"] = max(eq_number_list_of_problems)
    final_statistic_dict["average_eq_number_of_problems"] = statistics.mean(eq_number_list_of_problems)
    final_statistic_dict["stdev_eq_number_of_problems"] = custom_stdev(eq_number_list_of_problems)


    final_statistic_dict["min_unsatcore_eq_number_of_problems"] = min(unsatcore_eq_number_list_of_problems)
    final_statistic_dict["max_unsatcore_eq_number_of_problems"] = max(unsatcore_eq_number_list_of_problems)
    final_statistic_dict["average_unsatcore_eq_number_of_problems"] = statistics.mean(unsatcore_eq_number_list_of_problems)
    final_statistic_dict["stdev_unsatcore_eq_number_of_problems"] = custom_stdev(unsatcore_eq_number_list_of_problems)

    final_statistic_dict["min_eq_length"] = min(eq_length_list_of_problems)
    final_statistic_dict["max_eq_length"] = max(eq_length_list_of_problems)
    final_statistic_dict["average_eq_length"] = statistics.mean(eq_length_list_of_problems)
    final_statistic_dict["stdev_eq_length"] = custom_stdev(eq_length_list_of_problems)

    final_statistic_dict["min_variable_occurrence_of_problem"] = min(variable_occurrence_list_of_problems)
    final_statistic_dict["max_variable_occurrence_of_problem"] = max(variable_occurrence_list_of_problems)
    final_statistic_dict["average_variable_occurrence_of_problem"] = statistics.mean(
        variable_occurrence_list_of_problems)
    final_statistic_dict["stdev_variable_occurrence_of_problem"] = custom_stdev(variable_occurrence_list_of_problems)

    final_statistic_dict["min_terminal_occurrence_of_problem"] = min(terminal_occurrence_list_of_problems)
    final_statistic_dict["max_terminal_occurrence_of_problem"] = max(terminal_occurrence_list_of_problems)
    final_statistic_dict["average_terminal_occurrence_of_problem"] = statistics.mean(
        terminal_occurrence_list_of_problems)
    final_statistic_dict["stdev_terminal_occurrence_of_problem"] = custom_stdev(terminal_occurrence_list_of_problems)

    final_statistic_dict["min_variable_occurrence_of_equation"] = min(variable_occurrence_list_of_all_equations)
    final_statistic_dict["max_variable_occurrence_of_equation"] = max(variable_occurrence_list_of_all_equations)
    final_statistic_dict["average_variable_occurrence_of_equation"] = statistics.mean(
        variable_occurrence_list_of_all_equations)
    final_statistic_dict["stdev_variable_occurrence_of_equation"] = custom_stdev(
        variable_occurrence_list_of_all_equations)

    final_statistic_dict["min_terminal_occurrence_of_equation"] = min(terminal_occurrence_list_of_all_equations)
    final_statistic_dict["max_terminal_occurrence_of_equation"] = max(terminal_occurrence_list_of_all_equations)
    final_statistic_dict["average_terminal_occurrence_of_equation"] = statistics.mean(
        terminal_occurrence_list_of_all_equations)
    final_statistic_dict["stdev_terminal_occurrence_of_equation"] = custom_stdev(
        terminal_occurrence_list_of_all_equations)

    final_statistic_dict["min_variable_number_of_equation"] = min(variable_number_list_of_all_equations)
    final_statistic_dict["max_variable_number_of_equation"] = max(variable_number_list_of_all_equations)
    final_statistic_dict["average_variable_number_of_equation"] = statistics.mean(variable_number_list_of_all_equations)
    final_statistic_dict["stdev_variable_number_of_equation"] = custom_stdev(variable_number_list_of_all_equations)

    final_statistic_dict["min_terminal_number_of_equation"] = min(terminal_number_list_of_all_equations)
    final_statistic_dict["max_terminal_number_of_equation"] = max(terminal_number_list_of_all_equations)
    final_statistic_dict["average_terminal_number_of_equation"] = statistics.mean(terminal_number_list_of_all_equations)
    final_statistic_dict["stdev_terminal_number_of_equation"] = custom_stdev(terminal_number_list_of_all_equations)

    final_statistic_dict["GNN_embedding_list"] = GNN_embedding_list

    # save final_statistic_dict to final_statistic.json
    final_statistic_file = f"{html_directory}/final_statistic.json"
    with open(final_statistic_file, "w") as f:
        json.dump(final_statistic_dict, f, indent=4)

    return final_statistic_file


def custom_stdev(data):
    if len(data) < 2:
        # If there's only one data point, return 0 (no variability)
        return 0
    else:
        # Otherwise, use the statistics.stdev() function
        return statistics.stdev(data)


def compare_histograms(dict_name, benchmark_1, benchmark_2, dict1, dict2, output_html='comparison_histogram.html'):
    # Ensure data_dict1 and data_dict2 have keys as x-axis and values as y-axis
    x1 = list(dict1.keys())
    y1 = list(dict1.values())

    x2 = list(dict2.keys())
    y2 = list(dict2.values())

    # Create a figure with two bar traces (for both dictionaries)
    fig = go.Figure()

    # Add bar trace for the first dictionary
    fig.add_trace(go.Bar(
        x=x1, y=y1, name=f"{benchmark_1}", marker_color='blue', opacity=0.6
    ))

    # Add bar trace for the second dictionary
    fig.add_trace(go.Bar(
        x=x2, y=y2, name=f"{benchmark_2}", marker_color='red', opacity=0.6
    ))

    # Update the layout for better visualization
    fig.update_layout(
        title='Comparison of Two Dictionaries',
        xaxis_title=dict_name.split(":")[0],
        yaxis_title=dict_name.split(":")[1],
        barmode='group',  # Bars are grouped side by side
        bargap=0.2  # Space between bars
    )

    # Save the figure as an HTML file
    fig.write_html(output_html)
    print(f"Comparison histogram saved to {output_html}")


def _get_G_list_dgl(f: Formula, graph_func, dgl_hash_table, dgl_hash_table_hit):
    gc.disable()
    global_info = _get_global_info(f.eq_list)
    G_list_dgl = []

    # Local references to the hash table and counter for efficiency
    dgl_hash_table = dgl_hash_table
    dgl_hash_table_hit = dgl_hash_table_hit

    for index, eq in enumerate(f.eq_list):

        split_eq_nodes, split_eq_edges = graph_func(eq.left_terms, eq.right_terms, global_info)

        # hash eq+global info to dgl
        hashed_eq, _ = hash_graph_with_glob_info(split_eq_nodes, split_eq_edges)
        if hashed_eq in dgl_hash_table:
            dgl_graph = dgl_hash_table[hashed_eq]
            dgl_hash_table_hit += 1
        else:
            graph_dict = graph_to_gnn_format(split_eq_nodes, split_eq_edges)
            dgl_graph, _ = get_one_dgl_graph(graph_dict)
            dgl_hash_table[hashed_eq] = dgl_graph

        G_list_dgl.append(dgl_graph)

        # self.visualize_gnn_input_func(nodes=split_eq_nodes, edges=split_eq_edges,filename=self.file_name + f"_rank_call_{self.total_rank_call}_{index}")

    # Update the hit count back to the global variable
    dgl_hash_table_hit = dgl_hash_table_hit
    gc.enable()
    return G_list_dgl, dgl_hash_table, dgl_hash_table_hit


def load_gnn_model(model_dict):
    # load gnn model
    gnn_model_path = f"/home/cheli243/Desktop/CodeToGit/string-equation-solver/cluster-mlruns/mlruns/{model_dict['experiment_id']}/{model_dict['run_id']}/artifacts/model_2_{model_dict['graph_type']}_GCNSplit.pth"
    return load_model(gnn_model_path)

def get_one_PCA(E, html_directory):
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for easy visualization
    E_pca = pca.fit_transform(E)

    #visualize using plotly
    fig = px.scatter(
        x=E_pca[:, 0],  # X-axis: First principal component
        y=E_pca[:, 1],  # Y-axis: Second principal component
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
        title='PCA of Graph Embeddings',
        template='plotly',  # Using a Plotly template for a nice visual style
    )

    # Step 4: Customize Plot Appearance
    fig.update_traces(marker=dict(size=10, color='blue', opacity=0.6, line=dict(width=1, color='white')))
    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        showlegend=False,
    )

    html_file_path = f"{html_directory}/pca_graph_embedding_plot.html"
    fig.write_html(html_file_path)

    print(f"Plot saved as {html_file_path}")

def get_two_PCA(benchmark_name_1,benchmark_name_2,E1, E2, html_directory):
    # Step 1: Combine Embeddings
    # Concatenate the two sets of embeddings along rows to apply PCA together
    combined_embeddings = np.vstack((E1, E2))

    # Step 2: Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for easy visualization
    combined_pca = pca.fit_transform(combined_embeddings)

    # Step 3: Split PCA Transformed Data Back to Each Set
    n1 = len(E1)
    E1_pca = combined_pca[:n1]  # The PCA result for E1
    E2_pca = combined_pca[n1:]  # The PCA result for E2

    # Step 4: Create an Interactive Scatter Plot with Plotly
    fig = px.scatter(
        x=np.concatenate((E1_pca[:, 0], E2_pca[:, 0])),  # X-axis for both sets
        y=np.concatenate((E1_pca[:, 1], E2_pca[:, 1])),  # Y-axis for both sets
        color=[benchmark_name_1] * n1 + [benchmark_name_2] * len(E2),  # Use different colors for each set
        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
        title='PCA of Two Sets of Graph Embeddings',
        template='plotly',  # Use a Plotly template for a nice visual style
    )

    # Step 5: Customize Plot Appearance
    fig.update_traces(marker=dict(size=10, opacity=0.6, line=dict(width=1, color='white')))
    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        showlegend=True,
    )

    # Step 6: Save the Plotly Figure to an HTML File
    html_file_path = f"{html_directory}/pca_two_graph_embedding_plot.html"
    fig.write_html(html_file_path)

    print(f"Plot saved as {html_file_path}")

if __name__ == '__main__':
    main()
