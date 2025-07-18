# Graph Neural Networks – Manager‑Friendly Cheat‑Sheet

### 1  Why use a *graph* (and a neural network)?
| Talking point | Simple explanation | 30‑second example |
|---------------|-------------------|-------------------|
| **Graphs capture relationships** | Nodes = things, edges = connections. Most business data is naturally a network. | Customers sharing devices, proteins interacting, products bought together. |
| **Classic ML flattens the graph away** | Tabular → loses structure; GNN keeps the “who‑connects‑to‑whom.” | Fraud rules vs. ring detection on the whole payment network. |
| **Neural nets do automatic feature learning** | No need to hand‑engineer graph‑theory metrics; the model learns them. | Triangle counts, centrality scores, etc. emerge inside the network. |

---

### 2  How does a Graph Neural Network work?
1. **Input encoding**  
   • *Node features* – anything numeric or one‑hot: category, text embedding, timestamp bucket, etc.  
   • *Edge features* (optional) – weight, type, timestamp.  
   • *Adjacency* – who is linked to whom (sparse matrix or edge list).

2. **Message‑passing layer (— the “secret sauce”)**  
   Each node gathers its neighbours’ features, transforms them with shared weights, then updates its own embedding.  
   Stack 2–3 layers → information propagates 2–3 hops.

3. **Readout / task head**  
   • *Node classification* (fraud/not)  
   • *Link prediction* (will these users connect?)  
   • *Graph classification* (toxic molecule?)

---

### 3  Four GNN flavours you’re using
| Model | Core idea | Strengths | Watch‑outs |
|-------|-----------|-----------|------------|
| **GCN** (Kipf & Welling ’17) | *Neighborhood averaging*: each node gets the mean of its 1‑hop neighbours (via normalized Laplacian). | Simple, fast on small/medium graphs; good baseline. | Needs full graph in GPU memory; fixed receptive field. |
| **GraphSAGE** (Hamilton ’17) | *Sample & aggregate*: randomly sample neighbours, aggregate (mean/LSTM/pooling). | **Inductive** – handles unseen nodes/graphs; linear scale via sampling. | Stochastic batches add variance; sampling hyper‑params matter. |
| **GAT** (Veličković ’18) | *Self‑attention on edges*: learn weights (αᵢⱼ) for each neighbour so important nodes talk louder. | Captures heterogeneity; no need for node degrees in advance. | Attention heads ↑ memory; needs careful regularisation. |
| **SIGN** (Frasca ’20) | *Pre‑compute* K powers of adjacency (A, A², …) into features, then run a standard MLP. | **Training‑time speed** – no message passing during training; fits giant static graphs. | Extra offline preprocessing; less flexible for dynamic graphs. |

*Rule of thumb*  
• **Scale issue?** Try **SIGN** or **GraphSAGE**.  
• **Edge importance?** Try **GAT**.  
• **Quick baseline?** **GCN**.

---

### 4  Key questions & crisp answers
| Possible question | One‑liner answer |
|-------------------|------------------|
| “Why not just use a regular neural net?” | Regular nets ignore who’s connected; GNN learns from both node attributes **and** relationships. |
| “What if we add a new customer tomorrow?” | GraphSAGE (or SIGN with re‑compute) handles unseen nodes – no full retraining needed. |
| “Is this explainable?” | Attention weights (GAT) and feature probes show which neighbours/features drove a prediction. |
| “Is it production‑ready?” | Industry uses GNNs in recommendation (Pinterest), fraud (PayPal), drug discovery (DeepMind). Scalable frameworks (PyG, DGL) run on clusters. |

---

### 5  Pitfalls & tips
* **Garbage‑in, garbage‑out** – inaccurate edges hurt more in GNNs; invest in graph quality.  
* **Over‑smoothing** – deep GCN layers blur all nodes; 2–3 layers usually enough.  
* **Class imbalance** – rare‑fraud nodes need weighting or sampling.  
* **Monitoring** – track graph drift (new edge types, degree distribution) as well as conventional metrics.

---

### 6  Slide order suggestion
1. Business problem & why it’s a graph.  
2. Visual: simple 5‑node graph → one GNN layer animation.  
3. The four models table (above).  
4. Small performance/latency chart (baseline vs. GNN).  
5. Roadmap & next steps.
