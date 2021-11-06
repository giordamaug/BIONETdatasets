from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import loadtxt, save_graphs, load_graphs, save_info, load_info
from dgl import backend as F
from dgl.convert import graph as dgl_graph
import numpy as np

class MyTUDataset(DGLBuiltinDataset):
    r"""
    TUDataset contains lots of graph kernel datasets for graph classification.

    Parameters
    ----------
    name : str
        Dataset Name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``, can be the 
        datasets name on `<https://chrsmrrs.github.io/datasets/docs/datasets/>`_.

    Attributes
    ----------
    max_num_node : int
        Maximum number of nodes
    num_labels : int
        Number of classes

    Notes
    -----
    **IMPORTANT:** Some of the datasets have duplicate edges exist in the graphs, e.g.
    the edges in ``IMDB-BINARY`` are all duplicated.  DGL faithfully keeps the duplicates
    as per the original data.  Other frameworks such as PyTorch Geometric removes the
    duplicates by default.  You can remove the duplicate edges with :func:`dgl.to_simple`.

    Examples
    --------
    >>> data = TUDataset('DD')

    The dataset instance is an iterable

    >>> len(data)
    188
    >>> g, label = data[1024]
    >>> g
    Graph(num_nodes=88, num_edges=410,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'node_labels': Scheme(shape=(1,), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> label
    tensor([1])

    Batch the graphs and labels for mini-batch training

    >>> graphs, labels = zip(*[data[i] for i in range(16)])
    >>> batched_graphs = dgl.batch(graphs)
    >>> batched_labels = torch.tensor(labels)
    >>> batched_graphs
    Graph(num_nodes=9539, num_edges=47382,
          ndata_schemes={'node_labels': Scheme(shape=(1,), dtype=torch.int64), '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})

    Notes
    -----
    Graphs may have node labels, node attributes, edge labels, and edge attributes,
    varing from different dataset.

    Labels are mapped to :math:`\lbrace 0,\cdots,n-1 \rbrace` where :math:`n` is the
    number of labels (some datasets have raw labels :math:`\lbrace -1, 1 \rbrace` which
    will be mapped to :math:`\lbrace 0, 1 \rbrace`). In previous versions, the minimum
    label was added so that :math:`\lbrace -1, 1 \rbrace` was mapped to
    :math:`\lbrace 0, 2 \rbrace`.
    """

    def __init__(self, name, path, raw_dir=None, force_reload=False, verbose=False):
        self.path = path
        super(MyTUDataset, self).__init__(name=name, url="",
                                        raw_dir=raw_dir, force_reload=force_reload,
                                        verbose=verbose)
    
    def download(self):
        pass

    def process(self):
        DS_edge_list = self._idx_from_zero(
            loadtxt(self._file_path("A"), delimiter=",").astype(int))
        DS_indicator = self._idx_from_zero(
            loadtxt(self._file_path("graph_indicator"), delimiter=",").astype(int))
        
        if os.path.exists(self._file_path("graph_labels")):
            DS_graph_labels = self._idx_reset(
                loadtxt(self._file_path("graph_labels"), delimiter=",").astype(int))               
            self.num_labels = max(DS_graph_labels) + 1
            self.graph_labels = F.tensor(DS_graph_labels)     
        elif os.path.exists(self._file_path("graph_attributes")):
            DS_graph_labels = loadtxt(self._file_path("graph_attributes"), delimiter=",").astype(float)
            self.num_labels = None
            self.graph_labels = F.tensor(DS_graph_labels)     
        else:
            raise Exception("Unknown graph label or graph attributes")

        g = dgl_graph(([], []))
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])


        self.attr_dict = {
            'node_labels': ('ndata', 'node_labels'),
            'node_attributes': ('ndata', 'node_attr'),
            'edge_labels': ('edata', 'edge_labels'),
            'edge_attributes': ('edata', 'node_labels'),
        }

        for filename, field_name in self.attr_dict.items():
            try:
                data = loadtxt(self._file_path(filename),
                               delimiter=',')
                if 'label' in filename:
                    data = F.tensor(self._idx_from_zero(data))
                else:
                    data = F.tensor(data)
                getattr(g, field_name[0])[field_name[1]] = data
            except IOError:
                pass

        self.graph_lists = [g.subgraph(node_idx) for node_idx in node_idx_list]

    def save(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        label_dict = {'labels': self.graph_labels}
        info_dict = {'max_num_node': self.max_num_node,
                     'num_labels': self.num_labels}
        save_graphs(str(graph_path), self.graph_lists, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict['labels']
        self.max_num_node = info_dict['max_num_node']
        self.num_labels = info_dict['num_labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
            Graph with node feature stored in ``feat`` field and node label in ``node_label`` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]


    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)


    def _file_path(self, category):
        return os.path.join(self.path, self.name,
                            "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _idx_reset(idx_tensor):
        """Maps n unique labels to {0, ..., n-1} in an ordered fashion."""
        labels = np.unique(idx_tensor)
        relabel_map = {x: i for i, x in enumerate(labels)}
        new_idx_tensor = np.vectorize(relabel_map.get)(idx_tensor)
        return new_idx_tensor

    def statistics(self):
        return self.graph_lists[0].ndata['feat'].shape[1], \
            self.num_labels, \
            self.max_num_node
