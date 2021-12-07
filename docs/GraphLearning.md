

<style>
td {
  font-size: 15px
}
th {
  font-size: 15px
}
</style>

# Introduction

This presentation is a summarization of concepts from the book
[Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) by William L. Hamilton

# Basic Concepts

## What is a Graph?

Formally, a graph $\mathcal{G}=(V,E)$ is defined by a set of nodes $V$ and a set of edges $E$ between these nodes. We denote an edge going from node $u \in V$ to node $v \in V$ as $(u, v)\in E$. 

- **simple graphs**, where there is *at most one edge between each pair* of nodes, *no edges between a node and itself*, and where the *edges are all undirected*, i.e., $(u,v) \in E \leftrightarrow  (v, u)    \in E$.
- **multi-relational graphs**, 

# Node Embedding

## Graph Statistics

Extracting some statistics or features — based on heuristic functions or domain knowledge — and use these features as input to a standard machine learning classifier

### Node-Level Statistics And Features

- **Node degree**         $d_{u}=\sum_{v \in V} \mathbf{A}[u, v]$
- **Node Centrality**     $e_{u}=\frac{1}{\lambda} \sum_{v \in V} \mathbf{A}[u, v] e_{v} \forall u \in \mathcal{V}$
- **Clustering Coefficient** $c_{u}=\frac{\left|\left(v_{1}, v_{2}\right) \in \mathcal{E}: v_{1}, v_{2} \in \mathcal{N}(u)\right|}{\left(\begin{array}{c}d_{u} \\ 2\end{array}\right)}$
- **Closed Triangles**: counts the number of closed triangles within each node’s local neigh- borhood
- **Motifs**: counting arbitrary motifs or graphlets within a node’s ego graph
	- *Ego Graph*:  the subgraph containing that node, its neighbors, and all the edges between nodes in its neighborhood

### Graph-Level Features And Graph Kernels

- **Bag of Nodes**: aggregate node-level statistics 
	- 	es. histograms or other summary statistics based on the degrees, centralities, and clustering coefficients of the nodes in the graph)
- **The Weisfieler–Lehman Kernel**
	1. First, we assign an initial label $l^{(0)}(v)$ to each node. In most graphs, this label is simply the degree, i.e., $l^{(0)}(v)=d_{v} \forall v \in V$.
	2. Next, we iteratively assign a new label to each node by hashing the multi-set of the current labels within the node’s neighborhood:
    $$l^{(i)}(v)=\operatorname{HASH}\left(\left\{\left\{l^{(i-1)}(u) \forall u \in \mathcal{N}(v)\right\}\right\}\right)$$
   	where the double-braces are used to denote a multi-set and the hash function maps each unique multi-set to a unique new label.
	4. After running K iterations of relabeling (i.e.,Step 2),we now have a label $l^{(K)}(v)$  for each node that summarizes the structure of its K-hop neighborhood.

# Shallow Embeddings

## Factorization-Based Approaches

- Laplacian eigenmaps
- Inner-product methods

## Random Walk Embeddings

- DeepWalk and node2vec
- Large-scale information network embeddings
- Additional variants of the random-walk idea

## Limitations Of Shallow Embeddings

- do not share any parameters between nodes in the encoder, since the encoder directly optimizes a unique embedding vector for each node
- do not leverage node features in the encoder
- are inherently transductive

# The Graph Neural Network Model

The key idea is that we want to generate representations of nodes that actually depend on the structure of the graph, as well as any feature information we might have

To define a deep neural network over general graphs, we need to define a new kind of deep learning architecture
- **Permutation invariance and equivariance** 
	- $f\left(\mathbf{P A P}^{\top}\right)=f(\mathbf{A})$
	- $f\left(\mathbf{P A P}^{\top}\right)=\mathbf{P} f(\mathbf{A})$

## The Basic GNN

The basic GNN message passing is defined as:
$\mathbf{h}_{u}^{(k)}=\sigma\left(\mathbf{W}_{\text {self }}^{(k)} \mathbf{h}_{u}^{(k-1)}+\mathbf{W}_{\text {neigh }}^{(k)} \sum_{v \in \mathcal{N}(u)} \mathbf{h}_{v}^{(k-1)}+\mathbf{b}^{(k)}\right)$

where $\mathbf{W}_{\text {self }}^{(k)}, \mathbf{W}_{\text {neigh }}^{(k)} \in \mathbb{R}^{d^{(k)} \times d^{(k-1)}}$ are trainable parameter matrices and 􏰇 denotes an elementwise nonlinearity (e.g., a tanh or ReLU)