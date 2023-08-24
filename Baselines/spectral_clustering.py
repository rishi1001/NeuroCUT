## DO spectral clustering on networkx graph

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import sys
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import time
# Load graph
# graph_file = 'data/adjacency_matrix.txt'
def normalised_cuts(cuts, A,num_cuts=5):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    # Y is n*k matrix where n is number of nodes and k is number of partitions
    # make Y from cuts by one hot encoding
    Y = torch.zeros((cuts.shape[0],num_cuts))          # TODO : change num_cuts to g
    Y[torch.arange(cuts.shape[0]), cuts] = 1
    n = Y.shape[0]
    g = Y.shape[1]
    D = torch.sum(A, dim=1)
    # print(A.shape)
    # print(cuts.shape)
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense())
    return loss

def plot_norm_cut(A,num_cuts):
    '''
    function to plot norm_cuts vs number of components
    '''
    norm_cuts=[]
    n_components=[i for i in range(num_cuts-10,num_cuts+100,2)]

    for i in n_components:
        print(num_cuts)
        sc = SpectralClustering(n_clusters=num_cuts,n_components=i,eigen_solver='amg', affinity='precomputed')
        sc.fit(A)

        labels = sc.labels_
        norm_cut=normalised_cuts(labels,torch.from_numpy(A.toarray()),num_cuts=num_cuts)
        norm_cuts.append(norm_cut)
        print(i,norm_cut)
    
    plt.clf()
    plt.plot(n_components,norm_cuts)
    plt.xlabel('n_components')
    plt.ylabel('norm_cut')
    plt.title('norm_cut vs n_components')
    plt.savefig('norm_cut_vs_n_components.png')

def plot_norm_cut_vs_num_cuts(A,num_components):
    '''
    function to plot norm_cuts vs number of components
    '''
    norm_cuts=[]
    num_cuts=[i for i in range(1,num_components+45,2)]

    for i in num_cuts:
        print(i)
        sc = SpectralClustering(n_clusters=i,n_components=num_components,eigen_solver='amg', affinity='precomputed')
        sc.fit(A)

        labels = sc.labels_
        norm_cut=normalised_cuts(labels,torch.from_numpy(A.toarray()),num_cuts=i)
        norm_cuts.append(norm_cut)
        print(i,norm_cut)
    
    plt.clf()
    plt.plot(num_cuts,norm_cuts)
    plt.xlabel('num_cuts')
    plt.ylabel('norm_cut')
    plt.title('norm_cut vs num_cuts')
    plt.savefig('norm_cut_vs_num_cuts.png')


if len(sys.argv) != 5:
    print('Usage: python spectral_clustering.py graph_file num_nodes num_cuts output_file')
    sys.exit(1)
graph_file = sys.argv[1]
num_nodes = int(sys.argv[2])
num_cuts= int(sys.argv[3])
output_file = sys.argv[4]
# graph_file = './SBM_500_same_5_10_8/test_set/1/graph.txt'
# num_nodes = 500
# num_cuts= 8
# output_file = './SBM_500_same_5_10_8/test_set/1/cut_spectral.txt'

## reading graph
tgraph=nx.read_edgelist(graph_file, nodetype=int)
G=nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(tgraph.edges())
print("number of connected components: ", nx.number_connected_components(G))
A=nx.adjacency_matrix(G)
A=A.toarray()
# plot_norm_cut(A,100)
# exit(0)

# plot_norm_cut_vs_num_cuts(A,35)
# exit(0)
# L=nx.laplacian_matrix(G)
# print(L)
# L=L.toarray()
# L=np.absolute(L)
# print(L)
# print(np.isnan(L).any())
st=time.time()
# sc = SpectralClustering(n_clusters=num_cuts,n_components=70, affinity='precomputed')
sc = SpectralClustering(n_clusters=num_cuts,n_components=35,eigen_solver='amg', affinity='precomputed')
# sc = SpectralClustering(n_clusters=num_cuts,eigen_solver='amg', affinity='precomputed')
# print(sc.get_params())
A=np.asarray(A)
sc.fit(A)
ed=time.time()

# Extract the cluster assignments
labels = sc.labels_
np.savetxt(output_file, labels, fmt='%d')
print("time taken spectral clustering:", ed-st, "seconds")


# A = nx.adjacency_matrix(G)
# # A = torch.from_numpy(A.toarray())
# norm_cut=normalised_cuts(labels,A,num_cuts=num_cuts)
# print(norm_cut)



# Print the cluster assignments
# sc.fit_predict(L)
# print("here4")
# ## Save numpy array to file
