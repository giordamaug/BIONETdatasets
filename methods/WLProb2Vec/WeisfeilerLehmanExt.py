import hashlib
import networkx as nx
import igraph as ig
from typing import List, Dict
from tqdm import tqdm
import netpro2vec.utils as utils

class WeisfeilerLehmanExt(object):
    """
    Weisfeiler-Lehman feature extractor class. 
    The atomic label of a node is a pair: <vertex_label>_<vertex_attribute>, where:
      - <vertex_label> is a predefinined node label (if None, the index is used as default)
      - <vertex_attribute> is an input node attribute present in the graph (if None, it is 
                           omitted and the atomic label will by only the <vertex_label>)

    Args:
        graph (iGraph graph): python igraph data structure for graphs.
        wl_iterations (int): Number of WL iterations.
        vertex_label (bool):  Presence of vertex label (prefix of atomic label).
        vertex_attribute (bool): Presence of attributes (suffix of atomic label).
        encodew (bool): flag to enable word hashing (disabled by default).
        verbose (bool): verbose print enabling flag (for debug) 

    """
    def __init__(self, 
        graph: ig.Graph, 
        wl_iterations: int, 
        vertex_attribute: bool=False, 
        vertex_label: bool=None, 
        encodew: bool=False,
        verbose: bool=False 
    ):
        """
        Initialization method which also executes feature extraction.
        """
        self.verbose = verbose
        self.attribute = vertex_attribute
        self.encodew = encodew
        self.tqdm = tqdm if self.verbose else utils.nop
        self.wl_iterations = wl_iterations
        assert self.wl_iterations >= 0, "WL recursions must be > 0"
        self.graph = graph
        self.mode = "OUT" if graph.is_directed() else 'ALL' 
        self.vertex_label = vertex_label
        self.vertex_label_list = self.__get_vertex_labels()
        self._set_features()
        self._do_recursions()

    def __get_vertex_labels(self):
        if self.vertex_label is None:     # if no node label is specified, use the index as vertex label 
           return [v.index for v in ig.VertexSeq(self.graph)]
        elif self.vertex_label in self.graph.vs.attributes():
           return [v[self.vertex_label] for v in ig.VertexSeq(self.graph)]
        else:
           raise Exception('The graph does not have the provided vertex label (option -A)')

    def _set_features(self):
        """
        Creating the features.
        """
        #self.extracted_features = [ (self.vertex_label_list[v.index], self.vertex_label_list[v.index] + '_' + str(v["feature"])) for v in ig.VertexSeq(self.graph) ]
        if self.attribute :
            if self.attribute in self.graph.vs.attributes():
                self.features = {
                        v.index: str(self.vertex_label_list[v.index]) + '_' + str(v[self.attribute]) for v in ig.VertexSeq(self.graph)
                }
            else:
                raise Exception("Vertex attribute not present in graph!")
        else:
            self.features = {
                    v.index: str(self.vertex_label_list[v.index])
                }
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}
        #self.extracted_features = []
        #for v in ig.VertexSeq(self.graph):
        #    v["feature"] = str(self.vertex_label_list[v.index]) + '_' + str(v["feature"])
        #    self.extracted_features += [v["feature"]]
        #print("INIT", self.extracted_features)

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in ig.VertexSeq(self.graph):
            nebs = self.graph.neighbors(node, mode=self.mode)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node.index])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            if self.encodew:
                hash_object = hashlib.md5(features.encode())
                hashing = hash_object.hexdigest()
                new_features[node.index] = hashing
            else:
                new_features[node.index] = features
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features
        #new_features = {}
        #for node in ig.VertexSeq(self.graph):
        #    nebs = self.graph.neighbors(node, mode=self.mode)
        #    degs = [neb["feature"] for neb in self.graph.vs[nebs]]
        #    features = [str(node["feature"])]+sorted([str(deg) for deg in degs])
        #    features = "-".join(features)
            #hash_object = hashlib.md5(features.encode())
            #hashing = hash_object.hexdigest()
            #node["feature"] = hashing
            #new_features[node.index] = hashing
            #node["feature"] = features           # NO HASHING (memory exceeding!!!)
            #new_features[node.index] = features   # NO HASHING (memory exceeding!!!)
        #self.extracted_features = [ (self.vertex_label_list[k], str(v)) for k,v in new_features.items() ]
        #self.extracted_features = [ str(v) for k,v in new_features.items() ]
        #return self.extracted_features

    def _do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.wl_iterations):
            self.features = self._do_a_recursion()
        #print(self.extracted_features)

    def get_node_features(self) -> Dict[int, List[str]]:
        """
        Return the node level features.
        """
        return self.extracted_features

    def get_graph_features(self) -> List[str]:
        """
        Return the graph level features.
        """
        return [
            feature
            for node, features in self.extracted_features.items()
            for feature in features
        ]

    def get_graph_sentence(self) -> List[str]:
        """
        Return the graph level features.
        """
        return ' '.join([feature for node,feature in self.extracted_features])

