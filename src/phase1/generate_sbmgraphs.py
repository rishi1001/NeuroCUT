import networkx as nx
import torch
from networkx.generators.community import stochastic_block_model
import matplotlib.pyplot as plt
import os

# Define the number of nodes in each block
# n = [100, 100, 100, 100, 100]
n = [10,10,10]
# n = [5,5]
# n = [10,10,10,10,10]
tot_nodes = sum(n)
tot_blocks = len(n)


# Define the probability of edges between nodes in each block
# p = [[0.8,0.2,0.2,0.2,0.2],[0.2,0.8,0.2,0.2,0.2],[0.2,0.2,0.8,0.2,0.2],[0.2,0.2,0.2,0.8,0.2],[0.2,0.2,0.2,0.2,0.8]]
# p = [[0.6,0.01,0.01,0.01,0.01],[0.01,0.6,0.01,0.01,0.01],[0.01,0.01,0.6,0.01,0.01],[0.01,0.01,0.01,0.6,0.01],[0.01,0.01,0.01,0.01,0.6]]
# p = [[0.8, 0.2], [0.2, 0.8]]
# p = [[0.8, 0.2, 0.2, 0.2, 0.2], [0.2, 0.8, 0.2, 0.2, 0.2], [0.2, 0.2, 0.8, 0.2, 0.2], [0.2, 0.2, 0.2, 0.8, 0.2], [0.2, 0.2, 0.2, 0.2, 0.8]]
# p=[[0.2, 0.002, 0.002, 0.002, 0.002], [0.002, 0.2, 0.002, 0.002, 0.002], [0.002, 0.002, 0.2, 0.002, 0.002], [0.002, 0.002, 0.002, 0.2, 0.002], [0.002, 0.002, 0.002, 0.002, 0.2]]
p=[[0.8,0.05,0.05],[0.05,0.8,0.05],[0.05,0.05,0.8]]
# Generate a stochastic block model
folder_name = '../SBM_small/' +str(tot_nodes) + '_' + str(tot_blocks) + '/'

for i in range(1,6):
    folder_name = '../SBM_small/' +str(tot_nodes) + '_' + str(tot_blocks) + f'_{i}/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    G = stochastic_block_model(n, p)

    # Print the number of nodes and edges in the generated graph
    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))

    # plot the graph
    nx.draw(G, with_labels=False)
    # save the graph image
    plt.savefig(folder_name+"graph.png")
    nx.write_edgelist(G, folder_name+"graph.txt", data=False)
    plt.clf()
    # save the stats about graph in .txt file
    with open(folder_name+'stats.txt','w') as f:
        # total number of nodes
        f.write(f'total_nodes: {len(G.nodes)}')
        f.write('\n')
        # total number of edges
        f.write(f'total_edges: {len(G.edges)}')
        f.write('\n')
        # save n, p from above
        f.write(f'n: {n}')
        f.write('\n')
        f.write(f'p: {p}')
        f.write('\n')
        # total connected components
        total_connected_components = nx.number_connected_components(G)
        f.write(f'total_connected_components: {total_connected_components}')
        f.write('\n')
        # avg degree
        avg_degree = 2*len(G.edges)/len(G.nodes)
        f.write(f'avg_degree: {avg_degree}')
        f.write('\n')
        # highest degree, lowest degree
        highest_degree = max(dict(G.degree).values())
        lowest_degree = min(dict(G.degree).values())
        f.write(f'highest_degree: {highest_degree}')
        f.write('\n')
        f.write(f'lowest_degree: {lowest_degree}')
        f.write('\n')
        # avg clustering coefficient
        avg_clustering_coeff = nx.average_clustering(G)
        f.write(f'avg_clustering_coeff: {avg_clustering_coeff}')
        f.write('\n')
        # avg degree centrality
        avg_degree_centrality = sum(nx.degree_centrality(G).values())/len(nx.degree_centrality(G))
        f.write(f'avg_degree_centrality: {avg_degree_centrality}')
        f.write('\n')
        # avg closeness centrality
        avg_closeness_centrality = sum(nx.closeness_centrality(G).values())/len(nx.closeness_centrality(G))
        f.write(f'avg_closeness_centrality: {avg_closeness_centrality}')
        f.write('\n')
        # avg betweenness centrality
        avg_betweenness_centrality = sum(nx.betweenness_centrality(G).values())/len(nx.betweenness_centrality(G))
        f.write(f'avg_betweenness_centrality: {avg_betweenness_centrality}')
        f.write('\n')

