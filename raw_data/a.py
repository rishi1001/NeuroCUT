import os
import torch
import networkx as nx
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.datasets import HeterophilousGraphDataset, Actor


# Step 1: Load Citeseer dataset
# dataset = Planetoid(root='data/Citeseer', name='Citeseer', transform=T.NormalizeFeatures())

dataset = Actor(root='data/actor')
data = dataset[0]

breakpoint()
# Step 2: Convert to NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Remove self-loop edges
graph.remove_edges_from(nx.selfloop_edges(graph))

# Step 3: Find the largest connected component
largest_cc = max(nx.connected_components(graph), key=len)

# Step 4: Extract the subgraph corresponding to the largest connected component
largest_cc_graph = graph.subgraph(largest_cc)

print(largest_cc_graph)

nodes = list(largest_cc_graph.nodes())

# Step 5: Get node features for nodes in the largest connected component
node_features = data.x[nodes]

# Step 6: Save graph and node features
os.makedirs('actor', exist_ok=True)
torch.save(node_features, 'actor/node_embeddings.pt')

# Create a mapping between original node indices and indices in largest connected component
node_mapping = {node: idx for idx, node in enumerate(largest_cc_graph.nodes())}

# Save graph in .txt file
with open('actor/graph.txt', 'w') as f:
    for edge in largest_cc_graph.edges():
        adjusted_edge = (node_mapping[edge[0]], node_mapping[edge[1]])
        f.write(f"{adjusted_edge[0]} {adjusted_edge[1]}\n")

print("actor data saved successfully.")