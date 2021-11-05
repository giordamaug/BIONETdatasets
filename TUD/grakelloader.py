from __future__ import print_function
import os
import shutil
import zipfile
import os
import numpy as np
from collections import Counter
from sklearn.utils import Bunch
from grakel.graph import Graph
global datasets_metadata, symmetric_dataset

def read_data(
        name,
        metadata,
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False):
    """Create a dataset iterable for GraphKernel.

    Parameters
    ----------
    name : str
        The dataset name.

    metadata : dict
        the dataset meta data (used to specify which info to be used)
        
    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.


    Returns
    -------
    Gs : iterable
        An iterable of graphs consisting of a dictionary, node
        labels and edge labels for each graph.

    classes : np.array, case_of_appearance=with_classes==True
        An one dimensional array of graph classes aligned with the lines
        of the `Gs` iterable. Useful for classification.

    """
    indicator_path = "./"+str(name)+"/"+str(name)+"_graph_indicator.txt"
    edges_path = "./" + str(name) + "/" + str(name) + "_A.txt"
    node_labels_path = "./" + str(name) + "/" + str(name) + "_node_labels.txt"
    node_attributes_path = "./"+str(name)+"/"+str(name)+"_node_attributes.txt"
    edge_labels_path = "./" + str(name) + "/" + str(name) + "_edge_labels.txt"
    edge_attributes_path = \
        "./" + str(name) + "/" + str(name) + "_edge_attributes.txt"
    graph_classes_path = \
        "./" + str(name) + "/" + str(name) + "_graph_labels.txt"

    # node graph correspondence
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()

    # Associate graphs nodes with indexes
    with open(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()

    # Extract graph edges
    with open(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))

    # Extract node attributes
    if (prefer_attr_nodes and
        metadata.get(
                "na",
                os.path.exists(node_attributes_path)
                )):
        with open(node_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = \
                    [float(num) for num in
                     line[:-1].replace(' ', '').split(",")]
    # Extract node labels
    elif metadata.get(
            "nl",
            os.path.exists(node_labels_path)
            ):
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = int(line[:-1])
    elif produce_labels_nodes:
        for i in range(1, len(Graphs)+1):
            node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))

    # Extract edge attributes
    if (prefer_attr_edges and
        metadata.get(
            "ea",
            os.path.exists(edge_attributes_path)
            )):
        with open(edge_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                attrs = [float(num)
                         for num in line[:-1].replace(' ', '').split(",")]
                edge_labels[ngc[elc[i][0]]][elc[i]] = attrs

    # Extract edge labels
    elif metadata.get(
            "el",
            os.path.exists(edge_labels_path)
            ):
        with open(edge_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge_labels[ngc[elc[i][0]]][elc[i]] = int(line[:-1])

    Gs = list()
    if as_graphs:
        for i in range(1, len(Graphs)+1):
            Gs.append(Graph(Graphs[i], node_labels[i], edge_labels[i]))
    else:
        for i in range(1, len(Graphs)+1):
            Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

    if with_classes:
        classes = []
        with open(graph_classes_path, "r") as f:
            for line in f:
                classes.append(int(line[:-1]))

        classes = np.array(classes, dtype=np.int)
        return Bunch(data=Gs, target=classes)
    else:
        return Bunch(data=Gs)
