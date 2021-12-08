# Introduction

This presentation is a summarization of concepts from the book
[Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) by William L. Hamilton

# Basic Concepts

## What is a Graph?

Formally, a graph $\mathcal{G}=(V,E)$ is defined by a set of nodes $V$ and a set of edges $E$ between these nodes. We denote an edge going from node $u \in V$ to node $v \in V$ as $(u, v) \in E$. 

+ **simple graphs**, where there is *at most one edge between each pair* of nodes, *no edges between a node and itself*, and where the *edges are all undirected*, i.e., $(u,v) \in E \leftrightarrow  (v, u)    \in E$.
+ **multi-relational graphs**, 

# Node Embedding

## Graph Statistics

Extracting some statistics or features — based on heuristic functions or domain knowledge — and use these features as input to a standard machine learning classifier

### Node-Level Statistics And Features

- **Node degree**         $d_{u}=\sum_{v \in V} \mathbf{A}[u, v]$
- **Node Centrality**     $e_{u}=\frac{1}{\lambda} \sum_{v \in V} \mathbf{A}[u, v] e_{v} \forall u \in \mathcal{V}$
- **Clustering Coefficient** 
$
c_{u}=\frac{\left|\left(v_{1}, v_{2}\right) \in \mathcal{E}: v_{1}, v_{2} \in \mathcal{N}(u)\right|}{\binom{d_{u}}{2}}
$
- **Closed Triangles**: counts the number of closed triangles within each node’s local neigh- borhood
- **Motifs**: counting arbitrary motifs or graphlets within a node’s ego graph
	- *Ego Graph*:  the subgraph containing that node, its neighbors, and all the edges between nodes in its neighborhood

### Graph-Level Features And Graph Kernels

- **Bag of Nodes**: aggregate node-level statistics 
	- 	es. histograms or other summary statistics based on the degrees, centralities, and clustering coefficients of the nodes in the graph)
- **The Weisfieler–Lehman Kernel**
	- First, we assign an initial label $l^{(0)}(v)$ to each node. In most graphs, this label is simply the degree, i.e., $l^{(0)}(v)=d_{v} \forall v \in V$.
	- Next, we iteratively assign a new label to each node by hashing the multi-set of the current labels within the node’s neighborhood:

$$l^{(i)}(v)=\operatorname{HASH}\left(\left\{\left\{l^{(i-1)}(u) \forall u \in \mathcal{N}(v)\right\}\right\}\right)$$

- where the double-braces are used to denote a multi-set and the hash function maps each unique multi-set to a unique new label.
	- After running K iterations of relabeling (i.e.,Step 2),we now have a label $l^{(K)}(v)$  for each node that summarizes the structure of its K-hop neighborhood.

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

## Neural Message Passing

<img src="https://snap-stanford.github.io/cs224w-notes/assets/img/aggregate_neighbors.png" width="500" height="200" />

During each message-passing iteration in a GNN, a *hidden embedding* $\mathbf{h}_{u}^{(k)}$ corresponding to
each node $u \in V$ is updated according to information aggregated from $u$’s graph neighborhood $\mathcal{N}(v)$ (Figure 5.1). This message-passing update can be expressed as follows:

$$
\begin{aligned}
\mathbf{h}_{u}^{(k+1)} &=\operatorname{UPDATE}^{(k)}\left(\mathbf{h}_{u}^{(k)}, \text { AGGREGATE }^{(k)}\left(\left\{\mathbf{h}_{v}^{(k)}, \forall v \in \mathcal{N}(u)\right\}\right)\right) \\
&=\operatorname{UPDATE}^{(k)}\left(\mathbf{h}_{u}^{(k)}, \mathbf{m}_{\mathcal{N}(u)}^{(k)}\right)
\end{aligned}
$$

where update and aggregate are arbitrary differentiable functions (i.e., neural networks) and $\mathbf{m}_{\mathcal{N}(u)}$ is the “message” that is aggregated from $u$’s graph neighborhood $\mathcal{N}(u)$.

## The Basic GNN

The basic GNN message passing is defined as:
$\mathbf{h}_{u}^{(k)}=\sigma\left(\mathbf{W}_{\text {self }}^{(k)} \mathbf{h}_{u}^{(k-1)}+\mathbf{W}_{\text {neigh }}^{(k)} \sum_{v \in \mathcal{N}(u)} \mathbf{h}_{v}^{(k-1)}+\mathbf{b}^{(k)}\right)$

where $\mathbf{W}_{\text {self }}^{(k)}, \mathbf{W}_{\text {neigh }}^{(k)} \in \mathbb{R}^{d^{(k)} \times d^{(k-1)}}$ are trainable parameter matrices and 􏰇 denotes an elementwise nonlinearity (e.g., a tanh or ReLU)

We can equivalently define the basic GNN through the $\text{ UPDATE }$ and $\text{ AGGREGATE }$ functions:

$$
\begin{array}{l}
\mathbf{m}_{\mathcal{N}(u)}=\sum_{v \in \mathcal{N}(u)} \mathbf{h}_{v}, \\
\text { UPDATE }\left(\mathbf{h}_{u}, \mathbf{m}_{\mathcal{N}(u)}\right)=\sigma\left(\mathbf{W}_{\text {self }} \mathbf{h}_{u}+\mathbf{W}_{\text {neigh }} \mathbf{m}_{\mathcal{N}(u)}\right),
\end{array}
$$

where we recall that we use:

$$
\mathbf{m}_{\mathcal{N}(u)}=\operatorname{AGGREGATE}\left(\left\{\mathbf{h}_{v}, \forall v \in \mathcal{N}(u)\right\}\right)
$$

### Message Passing With Self-Loops

As a simplification of the neural message passing approach, it is common to add self-loops to the input graph and omit the explicit update step. In this approach we define the message passing simply as

$$
\mathbf{h}_{u}^{(k)}=\operatorname{AGGREGATE}\left(\left\{\mathbf{h}_{v}^{(k-1)}, \forall v \in \mathcal{N}(u) \cup\{u\}\right\}\right)
$$

where now the aggregation is taken over the set $\mathcal{N}(u) \cup\{u\}$, i.e., the node’s neighbors as well as the node itself.

### Neighborhood Normalization

The most basic neighborhood aggregation operation (Equation (5.8)) simply takes the sum of the neighbor embeddings. It can be **unstable** and highly **sensitive to node degrees**.

One solution to this problem is to simply normalize the aggregation operation

- **average normalizattion**: $
\mathbf{m}_{\mathcal{N}(u)}=\frac{\sum_{v \in \mathcal{N}(u)} \mathbf{h}_{v}}{|\mathcal{N}(u)|}
$

- **symmetric normalization** $
\mathbf{m}_{\mathcal{N}(u)}=\sum_{v \in \mathcal{N}(u)} \frac{\mathbf{h}_{v}}{\sqrt{|\mathcal{N}(u)||\mathcal{N}v)|}}
$

## Graph Convolutional Networks (GCNs)

One of the most popular baseline graph neural network models—the graph convolutional net- work (GCN)—employs the **symmetric-normalized aggregation** as well as the **self-loop update** approach. The GCN model thus defines the message passing function as:

$$
\mathbf{h}_{u}^{(k)}=\sigma\left(\mathbf{W}^{(k)} \sum_{v \in \mathcal{N}(u) \cup\{u\}} \frac{\mathbf{h}_{v}}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}\right)
$$

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
```
 
  see turorial [How to use edge features in Graph Neural Networks (and PyTorch Geometric)
](https://www.youtube.com/watch?v=mdWQYYapvR8)