import numpy as np
import igraph as ig
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import __version__ as gensimversion
from .WeisfeilerLehmanExt import WeisfeilerLehmanExt
from netpro2vec.DistributionGenerator import DistributionGenerator
import netpro2vec.utils as utils
import re
import random
from tqdm import tqdm
import pickle as pk
import os


class wlprob2vec:
    r"""An implementation of whiolegraph emebdding based on doc2vec and WL algorithm
    The procedure creates Weisfeiler-Lehman tree features for nodes in graphs. Using
    these features a document (graph) - feature co-occurence matrix is decomposed in order
    to generate representations for the graphs.

    The procedure assumes that nodes have no string feature present and the WL-hashing
    defaults to the degree centrality. However, if a node feature with the key "feature"
    is supported for the nodes the feature extraction happens based on the values of this key.

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        annotation (str): type of node (annotation) label used to build self.documents (default is node degree)
    """
    def __init__(self, wl_iterations: int=5, vertex_label=None, dimensions: int=128, annotation: str="degree",
                 encodew=False, workers: int=4, down_sampling: float=0.0001, epochs: int=10, 
                 learning_rate: float=0.025, min_count: int=5, seed: int=42,
                 verbose=False, save_probs=False, load_probs=False, save_vocab=False, load_vocab=False):

        self.wl_iterations = wl_iterations
        assert self.wl_iterations >= 0, "WL recursions must be > 0"
        self.verbose = verbose
        self.encodew = encodew
        self.tqdm = tqdm if self.verbose else utils.nop
        self.dimensions = dimensions
        assert self.dimensions > 0, "WL recursions must be > 0"
        self.vertex_label = vertex_label
        self.annotation = annotation
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.matcher = re.compile('^tm[0-9]+')
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.saveprobs = save_probs
        self.loadprobs = load_probs
        self.savevocab = save_vocab
        self.loadvocab = load_vocab
        self.seed = seed
        # create dir for prob mats storing
        if not os.path.exists('.np2vec'):
            os.makedirs('.np2vec')
        self.probmatfile = os.path.join('.np2vec','probmats.pkl')
        self.vocabfile = os.path.join('.np2vec','vocab.pkl')
        self.model = None

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __check_graphs(self, graphs):
        """Checking the consecutive numeric indexing."""
        for graph in graphs:
           numeric_indices = [index for index in range(graph.vcount())]
           node_indices = sorted([node.index for node in ig.VertexSeq(graph)])
           assert numeric_indices == node_indices, "The node indexing is wrong."

    def __set_features(self, graphs, annotation):
        """ compute probabilities and initilize feature of graph nodes"""
        if self.loadprobs:
            try:
                utils.vprint("Loading prob mats...", end='\n', verbose=self.verbose)
                infile = open(self.probmatfile,'rb')
                self.probmats = pk.load(infile)
                infile.close()
            except Exception as e:
                utils.vprint("Cannot load saved probs...%s"%e, end='\n', verbose=self.verbose)
                utils.vprint("...Let's generate them by scratch!", end='\n', verbose=self.verbose)
                self.probmats = DistributionGenerator(self.annotation, graphs, verbose=self.verbose).get_distributions()                
        else:
            self.probmats = DistributionGenerator(self.annotation, graphs, verbose=self.verbose).get_distributions()
        if self.saveprobs:
            try:
                utils.vprint("Saving prob mats...", end='\n', verbose=self.verbose)
                outfile = open(self.probmatfile,'wb')
                pk.dump(self.probmats, outfile)
                outfile.close()
            except Exception as e:
                raise Exception("Cannot save probs...",e, e.args)
        if self.annotation == "ndd":
            for gidx,graph in enumerate(graphs):
                for v in ig.VertexSeq(graph):
                    v["feature"] = np.argmax(self.probmats[gidx][v.index])+1
        elif self.matcher.match(self.annotation):   # match any 'tm<int>'
            for gidx,graph in enumerate(graphs):
                for v in ig.VertexSeq(graph):
                    v["feature"] = np.argmax(self.probmats[gidx][v.index],axis=0)+1
        elif self.annotation == "degree":
            for gidx,graph in enumerate(graphs):
                for v in ig.VertexSeq(graph):
                    v["feature"] = int(graph.strength(v))
        elif self.annotation == "betweenness":
            for gidx,graph in enumerate(graphs):
                for v in ig.VertexSeq(graph):
                    v["feature"] = int(graph.betweenness(vertices=v, directed=self.isDirected))
        else:
            raise Exception("Wrong distribution selection %s"%self.annotation)
        
    def fit(self, graphs: List[ig.Graph]):
        """
        Fitting a Graph2Vec model.

        Arg types:
            * **graphs** *(List of igraph graphs)* - The graphs to be embedded.
        """
        self._set_seed()
        self.__check_graphs(graphs)    # check graphs conditions
        self.isDirected = graphs[0].is_directed()
        if self.loadvocab:
            try:
                utils.vprint("Loading vocabulary...", end='\n', verbose=self.verbose)
                infile = open(self.vocabfile,'rb')
                self.documents = pk.load(infile)
                infile.close()
            except Exception as e:
                utils.vprint("Cannot load vocabulary...%s"%e, end='\n', verbose=self.verbose)
                utils.vprint("...Let's generate it by scratch!", end='\n', verbose=self.verbose)
                self.__set_features(graphs, self.annotation)
                utils.vprint("WL algorithm (depth %d)..."%self.wl_iterations, end='\n', verbose=self.verbose)
                self.documents = [WeisfeilerLehmanExt(graph, wl_iterations=self.wl_iterations, vertex_label=self.vertex_label, vertex_attribute='feature', verbose=self.verbose, encodew=self.encodew) for graph in self.tqdm(graphs)]
                utils.vprint("Building vocabulary...", end='\n', verbose=self.verbose)
                self.documents = [TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(self.tqdm(self.documents))]
        else:
            self.__set_features(graphs, self.annotation)
            utils.vprint("WL algorithm (depth %d)..."%self.wl_iterations, end='\n', verbose=self.verbose)
            self.documents = [WeisfeilerLehmanExt(graph, wl_iterations=self.wl_iterations, vertex_label=self.vertex_label, vertex_attribute='feature', verbose=self.verbose, encodew=self.encodew) for graph in self.tqdm(graphs)]
            utils.vprint("Building vocabulary...", end='\n', verbose=self.verbose)
            self.documents = [TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(self.tqdm(self.documents))]
        if self.savevocab:
            try:
                utils.vprint("Saving vocabulary...", end='\n', verbose=self.verbose)
                outfile = open(self.vocabfile,'wb')
                pk.dump(self.documents, outfile)
                outfile.close()
            except Exception as e:
                raise Exception("Cannot save vocabulary...",e, e.args)

        utils.vprint("Building model...(epochs %d, sampling %f)"%(self.epochs,self.down_sampling), end='\n', verbose=self.verbose)
        model = Doc2Vec(self.documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        epochs=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        self.model = model
        if gensimversion >= "4":
            self._embedding = model.docvecs.vectors
        else:
            self._embedding = model.docvecs.doctag_syn0
        #self._embedding = [model.docvecs[str(i)] for i, _ in enumerate(self.documents)]
        return self

    def infer_vector(self, graphs: List[ig.Graph]):
        """
        Inferring embedding from Graph2Vec model.

        Arg types:
            * **graphs** *(List of iGraph graphs)* - The graphs to be embedded.
        """
        if self.model is  None:
            raise Exception("model is empty... please fit it before inferring!")
        self._set_seed()
        self.__check_graphs(graphs)
        documents = [
            WeisfeilerLehmanExt(graph, wl_iterations=self.wl_iterations, 
               vertex_label=self.vertex_label, vertex_attribute='feature', verbose=self.verbose, encodew=self.encodew)
            for graph in graphs
        ]
        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]
        documents = [d.words for d in documents]
        if gensimversion >= "4":
            embedding_list = [self.model.infer_vector(doc,steps=0,alpha=self.learning_rate) for doc in documents]
        else:
            embedding_list = [self.model.infer_vector(doc,epochs=0,alpha=self.learning_rate) for doc in documents]
        return embedding_list


    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def get_documents(self):
        r"""Getting the embedding of graphs.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return self.documents
