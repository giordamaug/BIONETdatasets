import igraph as ig
import numpy as np
import os

import os
from typing import Iterator
import os 
from tqdm import tqdm 
import pandas as pd

class MyTUDataset():
    r"""
    MyTUDataset contains lots of graph kernel datasets for graph classification.

    Parameters
    ----------
    name : str
        Dataset Name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``, can be the 
        datasets name on `<https://chrsmrrs.github.io/datasets/docs/datasets/>`_.
    path : str
        Dataset directory path.

    Attributes
    ----------
    max_num_node : int
        Maximum number of nodes
    num_labels : int
        Number of classes
    graph_list: list
        The list of graphs
    graph_labels:  
        The list of graph labels


    """
    def __init__(self, name, path=".", use_deg_label=False, verbose=False):
        super().__init__()
        self.name = name
        self.verbose = verbose
        self.path = path
        self.use_deg_label = use_deg_label
        self.process()

    def process(self):
        with open(os.path.join(self.path, self.name +"_graph_indicator.txt"), "r") as f:
            graph_indicator = [int(i) - 1 for i in list(f)]
        f.closed

        # Nodes.
        self.num_graphs = max(graph_indicator)
        node_indices = []
        offset = []
        c = 0

        for i in (tqdm(range(self.num_graphs + 1), desc="Getting node indices:") if self.verbose else range(self.num_graphs + 1)):
            offset.append(c)
            c_i = graph_indicator.count(i)
            node_indices.append((c, c + c_i - 1))
            c += c_i

        graph_db = []
        gc = 1
        for i in (tqdm(node_indices, desc="Loading nodes:") if self.verbose else node_indices):
            g = ig.Graph()
            g['name'] = f'{self.name}_{gc}'
            gc += 1
            for j in range(i[1] - i[0]+1):
                g.add_vertex(j)

            graph_db.append(g)

        # Edges.
        with open(os.path.join(self.path, self.name + "_A.txt"), "r") as f:
            edges = [i.split(',') for i in list(f)]
        f.closed
        edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]
        edge_list = []
        edgeb_list = []
        for e in (tqdm(edges, desc="Loading edges:")if self.verbose else edges):
            g_id = graph_indicator[e[0]]
            g = graph_db[g_id]
            off = offset[g_id]
            edgesn = [(e.source, e.target) for e in g.es]
            # Avoid multigraph (for edge_list)
            if ((e[0] - off, e[1] - off) not in edgesn) and ((e[1] - off, e[0] - off) not in edgesn):
                g.add_edge(e[0] - off, e[1] - off)
                edge_list.append((e[0] - off, e[1] - off))
                edgeb_list.append(True)
            else:
                edgeb_list.append(False)

        # Node labels.
        if os.path.exists(os.path.join(self.path, self.name + "_node_labels.txt")):
            print("Loading node labels...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_node_labels.txt"), "r") as f:
                node_labels = [str.strip(i) for i in list(f)]
            f.closed
            
            # multiple node labels (not supported!)
            #node_labels = [i.split(',') for i in node_labels]
            #int_labels = [];
            #for i in range(len(node_labels)):
            #    int_labels.append([int(j) for j in node_labels[i]])
            
            i = 0
            for g in graph_db:
                for v in g.vs:
                    v['label'] = int(node_labels[i])
                    i += 1
        else:
            if self.use_deg_label:
                print("Calculating node labels as degree...") if self.verbose else None
                i = 0
                for g in graph_db:
                    for v in g.vs:
                        v['label'] = g.degree(v)
                        i += 1

        # Node Attributes.
        if os.path.exists(os.path.join(self.path, self.name + "_node_attributes.txt")):
            print("Loading node attributes...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_node_attributes.txt"), "r") as f:
                node_attributes = [str.strip(i) for i in list(f)]
            f.closed
            
            node_attributes = [i.split(',') for i in node_attributes]
            float_attributes = [];
            for i in range(len(node_attributes)):
                float_attributes.append([float(j) for j in node_attributes[i]])
            #maxLength = max(len(x) for x in float_attributes )
            #if len(maxLength) > 0:
            i = 0
            for g in graph_db:
                for v in g.vs:
                    v['attributes'] = float_attributes[i]
                    i += 1
        # Edge Labels.
        if os.path.exists(os.path.join(self.path, self.name + "_edge_labels.txt")):
            print("Loading edge labels...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_edge_labels.txt"), "r") as f:
                edge_labels = [str.strip(i) for i in list(f)]
            f.closed

            #edge_labels = [i.split(',') for i in edge_labels]
            e_labels = []
            for i in range(len(edge_labels)):
                if(edgeb_list[i]):
                    e_labels.append([int(j) for j in edge_labels[i]])
            
            i = 0
            for g in graph_db:
                for e in g.es:
                    e['label'] = int(edge_labels[i])
                    i += 1

        # Edge Attributes.
        if os.path.exists(os.path.join(self.path, self.name + "_edge_attributes.txt")):
            print("Loading edge attributes...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_edge_attributes.txt"), "r") as f:
                edge_attributes = [str.strip(i) for i in list(f)]
            f.closed

            edge_attributes = [i.split(',') for i in edge_attributes]
            e_attributes = []
            for i in range(len(edge_attributes)):
                if(edgeb_list[i]):
                    e_attributes.append([float(j) for j in edge_attributes[i]])
            
            i = 0
            for g in graph_db:
                for e in g.es:
                    e['attributes'] = e_attributes[i]
                    i += 1

        # Classes.
        if os.path.exists(os.path.join(self.path, self.name + "_graph_labels.txt")):
            print("Loading graph labels...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_graph_labels.txt"), "r") as f:
                classes = [str.strip(i) for i in list(f)]
            f.closed
            i = 0
            for g in graph_db:
                g['class'] = int(classes[i])
                i += 1

        # Targets.
        if os.path.exists(os.path.join(self.path, self.name + "_graph_attributes.txt")):
            print("Loading graph attributes...") if self.verbose else None
            with open(os.path.join(self.path, self.name + "_graph_attributes.txt"), 'r') as f:
                targets = [str.strip(i) for i in list(f)]
            f.closed
            
            targets= [i.split(',') for i in targets]
            ts = [];
            for i in range(len(targets)):
                ts.append([float(j) for j in targets[i]])
            
            i = 0
            for g in graph_db:
                g['attributes'] = ts[i]
                i += 1
        # Generate label file
        labels = []
        for g in graph_db:
            labels.append(g['class'])
        self.graph_list = graph_db
        self.graph_labels = labels
        return self
        