import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from models import *
import sys
import pickle as pkl
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import multiprocessing as mp
from matplotlib import pyplot as plt


def input_matrix():
    '''
    Returns a test sparse SciPy adjecency matrix
    '''
    # N = 8
    # data = np.ones(2 * 11)
    # row = np.array([0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,6,6,6,7,7])
    # col = np.array([1,2,0,2,3,0,1,3,1,2,4,3,5,6,7,4,6,4,5,7,4,6])

    N = 7
    data = np.ones(2 * 9)
    row = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6])
    col = np.array([2, 3, 4, 6, 0, 4, 5, 6, 0, 4, 5, 1, 2, 3, 2, 3, 1, 2])

    # N = 3
    # data = np.array([1/2,1/2,1/3,1/3])
    # row = np.array([0,1,1,2])
    # col = np.array([1,0,2,1])

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    return A

def custom_loss_sparse(Y, A,device):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    n = Y.shape[0]
    g = Y.shape[1]
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.tensor([0.], requires_grad=True).to(device)
    # for i in range(idx.shape[1]):
    #     loss = loss.clone() +  torch.dot(YbyGamma[idx[0,i],:], Y_t[:,idx[1,i]])*data[i]
    ## LOSS 1 is cutloss term
    loss1 = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
    # oneT = torch.ones((1, Y.shape[0])).to(device)
    # # Loss2 is normalizing term
    # # TODO remove loss 2(not in ICLR paper)
    # loss2 = torch.sum(torch.mm(torch.mm(oneT,Y) - n/g, (torch.mm(oneT,Y) - n/g).t())).to(device)
    # loss2 = torch.sum((torch.mm(oneT,Y) - n/g)**2).to(device)
    # print("Loss1: {}, Loss2: {}".format(100*loss1,0.01*loss2))
    loss=loss1
    return loss

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def test_partition(Y):
    _, idx = torch.max(Y, 1)
    return idx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def edge_cut(Y,A):
    node_idx = test_partition(Y)
    partitions = torch.zeros(Y.shape).to(device)
    partitions[torch.arange(Y.shape[0]), node_idx] = 1
    #print(partitions)
    # sum of all edges such that both vertices belong to different partitions
    cut_sum = torch.sum(torch.mm(partitions,(1 - partitions).t()) * A.to_dense()).to(device)
    tot_edges = torch.sum(A.to_dense())
    return cut_sum/tot_edges

def balanceness(Y):
    n = Y.shape[0]
    g = Y.shape[1]
    node_idx = test_partition(Y)
    bin_count = torch.bincount(node_idx)
    # RMSE of bin_count and n/g
    # n(number of nodes) is used as normalizing term
    return 1-torch.sqrt(torch.mean((bin_count - n/g)**2))/n

def load_brightkite():
    graph=nx.read_edgelist("../../data/Brightkite_edges.txt",nodetype=int)
    # print(graph)
    # print(list(graph))
    # nx.write_edgelist(graph,"tttt.txt",data=False,delimiter='   ')
    A = nx.adjacency_matrix(graph)
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
    As = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor
    return adj , adj, As,  graph

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    tgraph=nx.read_edgelist(f'../../data/{dataset_str}.txt', nodetype=int)
    graph=nx.Graph()
    num_nodes = tgraph.number_of_nodes()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(tgraph.edges())
    print(graph.number_of_nodes())
    print(graph.number_of_edges())
    A = nx.adjacency_matrix(graph)
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
    As = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor
    return adj , adj, As,  graph

    # if (dataset_str=='brightkite'):
    #     return load_brightkite()
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("../../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally, graph = tuple(objects)        # x is sparse matrix 
    # ## graph IS A dictionary 
    # test_idx_reorder = parse_index_file("../../data/ind.{}.test.index".format(dataset_str))
    # # print(len(test_idx_reorder))
    # test_idx_range = np.sort(test_idx_reorder)

    # if dataset_str == 'citeseer':               # TODO
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # features = sparse_mx_to_torch_sparse_tensor(features).to(device)
    
    # A = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    # norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    # adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
    # As = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor

    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return features, adj, As,  nx.from_dict_of_lists(graph)

def pca_embedding(adj2,comp):
    adj=adj2.to_dense()
    # print(adj)
    normalized_features= StandardScaler().fit_transform(adj.cpu())
    # print("Normalizing features")
    # print(normalized_features)
    # print(normalized_features.shape)

    pca = PCA(n_components=comp)
    principalComponents = pca.fit_transform(normalized_features)
    # exp_var_pca = pca.explained_variance_ratio_
    # #
    # # Cumulative sum of eigenvalues; This will be used to create step plot
    # # for visualizing the variance explained by each principal component.
    # #
    # cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    # #
    # # Create the visualization plot
    # #
    # plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    # #plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal component index')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig("variance_citeseer_100.png")
    # plt.clf()
    # #plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    # plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    # plt.ylabel('Explained variance ratio')
    # plt.xlabel('Principal component index')
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig("cumulative_variance_citeseer_100.png")

    # print("PCA")
    # print(principalComponents)
    # print(principalComponents.shape)
    pcas=torch.from_numpy(principalComponents).to(device)
    #print(pcas.to_sparse())
    return pcas.to_sparse().float()

def single_source_shortest_path_length_range(graph, node_range, cutoff):
	dists_dict = {}
	for node in node_range:
		dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight='haversine')
	return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

##  k number OF anchors 
##  G networkx graphs
##    
def lipschitz_node_embeddings(nodes_forward, G, k):
    nodes = list(nodes_forward.keys())
    G_temp=G.copy()
    anchor_nodes = random.sample(nodes, k)
    print('Starting Dijkstra')
    num_workers = 32
    cutoff = None
    pool = mp.Pool(processes = num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range, \
        args=(G_temp, anchor_nodes[int(k/num_workers*i):int(k/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    print('Dijkstra done')
    embeddings = torch.zeros((len(nodes),k)).to(device)
    for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(nodes):
            dist = shortest_dist.get(node_j, -1)
            if dist!=-1:
                embeddings[nodes_forward[node_j], i] = 1 / (dist + 1)
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    return embeddings
''' 
T is transition matrix
node_range=nodes to do random walks
r is restanrt probablity
max_iters is probablity to start
'''
def RWR(T,node_range,num_nodes,r=0.15,max_iters=100):
    dists_dict = {}
    for node in node_range:
        # print(node)
        R=torch.zeros((num_nodes,1))
        R[node][0]=1.0
        E=R
        for i in range (max_iters):
            R=(1-r)*(torch.mm(T,R)) + r*(E)

        dists_dict[node] = R
    return dists_dict

def lipschitz_RW_node_embeddings(nodes_forward, G,k):
    nodes = list(nodes_forward.keys())
    anchor_nodes = random.sample(nodes, k)
    A = nx.adjacency_matrix(G)
    A=sparse_mx_to_torch_sparse_tensor(A).to_dense()
    d = A.sum(axis = 0)
    T=A/d
    print('Random walks')
    # num_workers = 32
    # pool = mp.Pool(processes = num_workers)
    # results = [pool.apply_async(RWR, \
    #     args=(T, anchor_nodes[int(k/num_workers*i):int(k/num_workers*(i+1))], len(nodes))) for i in range(num_workers)]
    # output = [p.get() for p in results]
    # rw_dicts = merge_dicts(output)
    # pool.close()
    # pool.join()
    rw_dicts=RWR(T,anchor_nodes,len(nodes))
    # print(dists_dict)
    embeddings = torch.zeros((len(nodes),k)).to(device)
    for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        rw_dist = rw_dicts[node_i]
        for j, node_j in enumerate(nodes):
            # dist = shortest_dist.get(node_j, -1)
            dist = rw_dist[node_j]
            ## TODO check next line
            embeddings[nodes_forward[node_j], i] = dist
    embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    return embeddings
## G is networkx graph
## k is  number of anchor nodes
## num is number of nodes
def lipschitz_rw_embedding(G, k,num):
    # print("here")
    nodes_forward = {i:i for i in range(num)}
    embedding=lipschitz_RW_node_embeddings(nodes_forward,G,k)
    embedding=embedding.to_sparse().float()
    return embedding


def lipschitz_embedding(G, k,num):
    nodes_forward = {i:i for i in range(num)}
    embedding=lipschitz_node_embeddings(nodes_forward,G,k)
    embedding=embedding.to_sparse().float()
    return embedding

def curr_score(seq,A):
    # list of partitions of seq
    # print(seq)
    ans = 0
    for i in range(len(seq)):
        for j in range(i+1,len(seq)):
            if seq[i]!=seq[j]:
                ans+=A[seq[i],seq[j]]
    return -1*ans


def beam_search(prob,A,k,b):            # TODO make it faster(also check for correctness)

    sequences = [[[list() for i in range(k)], 0.0]]

    for ind in range(len(prob)):
        row = prob[ind]
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # print(seq)
            modified_row = []
            for j in range(k):               # k = len(row)
                modified_row.append([row[j],j])
            modified_row.sort(reverse=True)
            # print(modified_row)
            for j in range(b):
                seq[modified_row[j][1]].append(ind)
                # print(seq)
                candidate = [seq, curr_score(seq[0],A)]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:b]
        print(ind)
    return sequences[-1]

