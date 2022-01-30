import glob
import os
from re import VERBOSE
import shutil
from os import path as osp
from urllib.error import URLError

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from spektral.data import Dataset, Graph
from spektral.datasets.utils import download_file
from spektral.utils import io, sparse, one_hot

import igraph as ig
import tqdm as tq
import pandas as pd
from sklearn import preprocessing

class MyGMLDataset(Dataset):
    """
    Node features are computed by concatenating the following features for
    each node:

    - node attributes, if available;
    - node labels, if available, one-hot encoded.

    Some datasets might not have node features at all. In this case, attempting
    to use the dataset with a Loader will result in a crash. Unless you set the 
    `force_degree` flag, which compute degrees of nodes and use them (hot encoded)
    as node attributes.
    Otherwise You can create features using some of the transforms available in 
    `spektral.transforms` or you can define your own features by accessing the 
    individual samples in the `graph` attribute of the dataset (which is a list of 
    `Graph` objects).

    Edge features are computed by concatenating the following features for
    each node:

    - edge attributes, if available;
    - edge labels, if available, one-hot encoded.

    Graph labels are provided for each dataset.

    Specific details about each individual dataset can be found in
    `~/.spektral/datasets/TUDataset/<dataset name>/README.md`, after the dataset
    has been downloaded locally (datasets are downloaded automatically upon
    calling `TUDataset('<dataset name>')` the first time).

    **Arguments**

    - `name`: str, name of the dataset to load.
    - `path`: str, pathname of dir where the dataset is located.
    - `force_degree`: bool, flag to enable suing degree as attributes.
    """

    def __init__(self, name, path, upto=None, **kwargs):
        self.name = name
        self.datapath = path
        self.upto = upto
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(self.datapath, self.name)

    def download(self):

        # Datasets are zipped in a folder: unpack them
        return

    def read(self):


        filenames = os.listdir(os.path.join(self.datapath, self.name, 'graphml'))

        # Node features
        x_list = []
        e_list = []
        a_list = []
        graphs = []
        for f in tq.tqdm(filenames[0:self.upto]): 
            if f.endswith('graphml'):
                g = ig.load(os.path.join(self.datapath,self.name, 'graphml',f))
                graphs += [g]
                if 'name' not in g.attributes(): g['name'] = os.path.splitext(f)[0]
                node_attributes = g.vs.attributes()
                if 'id' in node_attributes: node_attributes.remove('id') 
                edge_attributes = g.es.attributes() 
                if 'id' in edge_attributes: edge_attributes.remove('id')
                a_list.append(np.array(g.get_adjacency().data))
                x_labs = np.empty((g.vcount(),0))
                e_labs = np.empty((g.ecount(),0))
                if node_attributes==[]:
                    if self.force_degree:
                        for a,n in zip(a_list, n_nodes):
                            degree = a.sum(1).astype(int)
                            degree = np.array(degree).reshape(-1)
                            max_degree = degree.max()
                            degree = one_hot(degree, max_degree + 1)
                            x_list.append(degree)
                    else:
                        print(
                            "WARNING: this dataset doesn't have node attributes."
                            "and you didn't specify degree as node feature (force_degree). "
                            "Consider creating manual features before using it with a "
                            "Loader."
                        )
                        x_list = [None] * len(n_nodes)
                else:
                    for nattr in node_attributes:
                        xm = np.array([g.vs.get_attribute_values(nattr)])
                        xm = np.concatenate(
                            [_normalize(xl_[:, None], "ohe") for xl_ in xm], -1
                        )
                        x_labs = np.concatenate((x_labs, xm), axis=1)
                    x_list.append(x_labs)
                for eattr in edge_attributes:
                    em = np.array([g.es.get_attribute_values(eattr)])
                    em = np.concatenate(
                        [_normalize(el_[:, None], "ohe") for el_ in em], -1
                    )
                    e_labs = np.concatenate((e_labs, em), axis=1)
                e_list.append(e_labs)

        dfl = pd.read_csv(os.path.join(self.datapath,self.name, f'{self.name}.txt'), sep='\t')
        last_column = dfl.iloc[:,[0] + [-1]]
        labels = [last_column[last_column['Samples'] == g["name"]].iloc[:,-1:].values[0] for g in graphs]
        le = preprocessing.LabelEncoder()
        le.fit(np.ravel(labels))
        labels = le.transform(np.ravel(labels))
        labels = _normalize(labels[:, None], "ohe")
        assert len(labels) == len(graphs)
        print(labels.shape)

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]


        # Read edge lists
        edges = io.load_txt(fname_template.format("A"), delimiter=",").astype(int) - 1
        # Remove duplicates and self-loops from edges
        _, mask = np.unique(edges, axis=0, return_index=True)
        mask = mask[edges[mask, 0] != edges[mask, 1]]
        edges = edges[mask]
        # Split edges into separate edge lists
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(edges - n_nodes_cum[edge_batch_idx, None], n_edges_cum)

        # Node features
        x_list = []
        if "node_attributes" in available:
            x_attr = io.load_txt(
                fname_template.format("node_attributes"), delimiter=","
            )
            if x_attr.ndim == 1:
                x_attr = x_attr[:, None]
            x_list.append(x_attr)
        if "node_labels" in available:
            x_labs = io.load_txt(fname_template.format("node_labels"))
            if x_labs.ndim == 1:
                x_labs = x_labs[:, None]
            x_labs = np.concatenate(
                [_normalize(xl_[:, None], "ohe") for xl_ in x_labs.T], -1
            )
            x_list.append(x_labs)
        if len(x_list) > 0:
            x_list = np.concatenate(x_list, -1)
            x_list = np.split(x_list, n_nodes_cum[1:])
        else:
            print(
                "WARNING: this dataset doesn't have node attributes."
                "Consider creating manual features before using it with a "
                "Loader."
            )
            x_list = [None] * len(n_nodes)

        # Edge features
        e_list = []
        if "edge_attributes" in available:
            e_attr = io.load_txt(fname_template.format("edge_attributes"), delimiter=",")
            if e_attr.ndim == 1:
                e_attr = e_attr[:, None]
            e_attr = e_attr[mask]
            e_list.append(e_attr)
        if "edge_labels" in available:
            e_labs = io.load_txt(fname_template.format("edge_labels"))
            if e_labs.ndim == 1:
                e_labs = e_labs[:, None]
            e_labs = e_labs[mask]
            e_labs = np.concatenate(
                [_normalize(el_[:, None], "ohe") for el_ in e_labs.T], -1
            )
            e_list.append(e_labs)
        if len(e_list) > 0:
            e_available = True
            e_list = np.concatenate(e_list, -1)
            e_list = np.split(e_list, n_edges_cum)
        else:
            e_available = False
            e_list = [None] * len(n_nodes)

        # Create sparse adjacency matrices and re-sort edge attributes in lexicographic
        # order
        a_e_list = [
            sparse.edge_index_to_matrix(
                edge_index=el,
                edge_weight=np.ones(el.shape[0]),
                edge_features=e,
                shape=(n, n),
            )
            for el, e, n in zip(el_list, e_list, n_nodes)
        ]
        if e_available:
            a_list, e_list = list(zip(*a_e_list))
        else:
            a_list = a_e_list

        # Labels
        if "graph_attributes" in available:
            labels = io.load_txt(fname_template.format("graph_attributes"))
        elif "graph_labels" in available:
            labels = io.load_txt(fname_template.format("graph_labels"))
            labels = _normalize(labels[:, None], "ohe")
        else:
            raise ValueError("No labels available for dataset {}".format(self.name))

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == "ohe":
        fnorm = OneHotEncoder(sparse=False, categories="auto")
    elif norm == "zscore":
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)

class MyTUDataset(Dataset):
    """
    Node features are computed by concatenating the following features for
    each node:

    - node attributes, if available;
    - node labels, if available, one-hot encoded.

    Some datasets might not have node features at all. In this case, attempting
    to use the dataset with a Loader will result in a crash. Unless you set the 
    `force_degree` flag, which compute degrees of nodes and use them (hot encoded)
    as node attributes.
    Otherwise You can create features using some of the transforms available in 
    `spektral.transforms` or you can define your own features by accessing the 
    individual samples in the `graph` attribute of the dataset (which is a list of 
    `Graph` objects).

    Edge features are computed by concatenating the following features for
    each node:

    - edge attributes, if available;
    - edge labels, if available, one-hot encoded.

    Graph labels are provided for each dataset.

    Specific details about each individual dataset can be found in
    `~/.spektral/datasets/TUDataset/<dataset name>/README.md`, after the dataset
    has been downloaded locally (datasets are downloaded automatically upon
    calling `TUDataset('<dataset name>')` the first time).

    **Arguments**

    - `name`: str, name of the dataset to load.
    - `path`: str, pathname of dir where the dataset is located.
    """

    def __init__(self, name, path, force_degree=False, verbose=False, **kwargs):
        self.name = name
        self.datapath = path
        self.force_degree = force_degree
        self.verbose = verbose
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(self.datapath, self.name)

    def download(self):

        # Datasets are zipped in a folder: unpack them
        return

    def read(self):
        fname_template = osp.join(self.datapath, "{}_{{}}.txt".format(self.name))
        available = [
            f.split(os.sep)[-1][len(self.name) + 1 : -4]  # Remove leading name
            for f in glob.glob(fname_template.format("*"))
        ]

        # Batch index
        node_batch_index = (
            io.load_txt(fname_template.format("graph_indicator")).astype(int) - 1
        )
        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # Read edge lists
        edges = io.load_txt(fname_template.format("A"), delimiter=",").astype(int) - 1
        # Remove duplicates and self-loops from edges
        _, mask = np.unique(edges, axis=0, return_index=True)
        mask = mask[edges[mask, 0] != edges[mask, 1]]
        edges = edges[mask]
        # Split edges into separate edge lists
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(edges - n_nodes_cum[edge_batch_idx, None], n_edges_cum)


        # Edge features
        e_list = []
        if "edge_attributes" in available:
            e_attr = io.load_txt(fname_template.format("edge_attributes"), delimiter=",")
            if e_attr.ndim == 1:
                e_attr = e_attr[:, None]
            e_attr = e_attr[mask]
            e_list.append(e_attr)
        if "edge_labels" in available:
            e_labs = io.load_txt(fname_template.format("edge_labels"))
            if e_labs.ndim == 1:
                e_labs = e_labs[:, None]
            e_labs = e_labs[mask]
            e_labs = np.concatenate(
                [_normalize(el_[:, None], "ohe") for el_ in e_labs.T], -1
            )
            e_list.append(e_labs)
        if len(e_list) > 0:
            e_available = True
            e_list = np.concatenate(e_list, -1)
            e_list = np.split(e_list, n_edges_cum)
        else:
            e_available = False
            e_list = [None] * len(n_nodes)

        # Create sparse adjacency matrices and re-sort edge attributes in lexicographic
        # order
        a_e_list = [
            sparse.edge_index_to_matrix(
                edge_index=el,
                edge_weight=np.ones(el.shape[0]),
                edge_features=e,
                shape=(n, n),
            )
            for el, e, n in zip(el_list, e_list, n_nodes)
        ]
        if e_available:
            a_list, e_list = list(zip(*a_e_list))
        else:
            a_list = a_e_list

        # Node features
        x_list = []
        if "node_attributes" in available:
            x_attr = io.load_txt(
                fname_template.format("node_attributes"), delimiter=","
            )
            if x_attr.ndim == 1:
                x_attr = x_attr[:, None]
            x_list.append(x_attr)
        if "node_labels" in available:
            x_labs = io.load_txt(fname_template.format("node_labels"))
            if x_labs.ndim == 1:
                x_labs = x_labs[:, None]
            x_labs = np.concatenate(
                [_normalize(xl_[:, None], "ohe") for xl_ in x_labs.T], -1
            )
            x_list.append(x_labs)
        if len(x_list) > 0:
            x_list = np.concatenate(x_list, -1)
            x_list = np.split(x_list, n_nodes_cum[1:])
        else:
            if self.force_degree:
                for a,n in zip(a_list, n_nodes):
                    degree = a.sum(1).astype(int)
                    degree = np.array(degree).reshape(-1)
                    max_degree = degree.max()
                    degree = one_hot(degree, max_degree + 1)
                    x_list.append(degree)
            else:
                print(
                    "WARNING: this dataset doesn't have node attributes."
                    "and you didn't specify degree as node feature (force_degree). "
                    "Consider creating manual features before using it with a "
                    "Loader."
                )
                x_list = [None] * len(n_nodes)

        # Labels
        if "graph_attributes" in available:
            labels = io.load_txt(fname_template.format("graph_attributes"))
        elif "graph_labels" in available:
            labels = io.load_txt(fname_template.format("graph_labels"))
            labels = _normalize(labels[:, None], "ohe")
        else:
            raise ValueError("No labels available for dataset {}".format(self.name))

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == "ohe":
        fnorm = OneHotEncoder(sparse=False, categories="auto")
    elif norm == "zscore":
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)
