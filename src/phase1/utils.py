import networkx as nx
import torch,sklearn
import random
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn.functional as F
def cluster_centrality(g):
    return nx.clustering(g)

def degree_centrality(g):
    return nx.degree_centrality(g)

def betweenness_centrality(g):
    return nx.betweenness_centrality(g)

def closeness_centrality(g):
    return nx.closeness_centrality(g)

def core_number(g):
    g1=g.copy()
    g1.remove_edges_from(nx.selfloop_edges(g1))
    return nx.core_number(g1)

def edge_betweenness_centrality(g):
    return nx.edge_betweenness_centrality(g)

# def edge_current_flow_betweenness_centrality(g):
#     return nx.edge_current_flow_betweenness_centrality(g)

# def edge_load_centrality(g):
#     return nx.edge_load_centrality(g)

def single_source_shortest_path_length_range(graph, node_range, cutoff):
	dists_dict = {}
	for node in node_range:
		dists_dict[node] = nx.single_source_dijkstra_path_length(graph, node, cutoff=cutoff, weight='haversine')
	return dists_dict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

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
    random.seed(1)
    print("Seed for generating anchor node: ", 1)
    anchor_nodes = random.sample(nodes, k)
    print(anchor_nodes)
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
    embeddings = torch.zeros((len(nodes),k))
    for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(nodes):
            dist = shortest_dist.get(node_j, -1)
            if dist!=-1:
                embeddings[nodes_forward[node_j], i] = 1 / (dist + 1)
    # embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
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
    random.seed(1)
    print("Seed for generating anchor node: ", 1)
    anchor_nodes = random.sample(nodes, k)
    print(anchor_nodes)
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
    embeddings = torch.zeros((len(nodes),k))
    for i, node_i in tqdm(enumerate(anchor_nodes), dynamic_ncols=True):
        rw_dist = rw_dicts[node_i]
        for j, node_j in enumerate(nodes):
            # dist = shortest_dist.get(node_j, -1)
            dist = rw_dist[node_j]
            ## TODO check next line
            embeddings[nodes_forward[node_j], i] = dist
    # embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)
    return embeddings
## G is networkx graph
## k is  number of anchor nodes
## num is number of nodes
def lipschitz_rw_embedding(G, k,num):
    # print("here")
    nodes_forward = {i:i for i in range(num)}
    embedding=lipschitz_RW_node_embeddings(nodes_forward,G,k)
    return embedding


def lipschitz_embedding(G, k,num):
    nodes_forward = {i:i for i in range(num)}
    embedding=lipschitz_node_embeddings(nodes_forward,G,k)
    # embedding=embedding.to_sparse().float()
    return embedding



def generate(y):
    ind_0 = torch.nonzero(y == 0).reshape(-1)
    ind_1 = torch.nonzero(y == 1).reshape(-1)
    # print(ind_0,ind_1)
    index0 = list(torch.utils.data.RandomSampler(ind_0, replacement=True, num_samples=2))
    ind_0_ind = ind_0[index0]
    index1 = list(torch.utils.data.RandomSampler(ind_1, replacement=True, num_samples=2))
    ind_1_ind = ind_1[index1]
    test_set, dummy = torch.cat((ind_0_ind, ind_1_ind)).sort()

    a = []
    # remove ind_0_ind from ind_0
    for i in range(len(ind_0)):
        if ind_0[i] not in ind_0_ind:
            a.append(ind_0[i])

    ind_0 = torch.tensor(a)

    a = []
    # remove ind_0_ind from ind_0
    for i in range(len(ind_1)):
        if ind_1[i] not in ind_1_ind:
            a.append(ind_1[i])

    ind_1 = torch.tensor(a)
    # print(ind_0,ind_1)
    index0 = list(torch.utils.data.RandomSampler(ind_0, replacement=True, num_samples=2))
    ind_0_ind = ind_0[index0]
    index1 = list(torch.utils.data.RandomSampler(ind_1, replacement=True, num_samples=2))
    ind_1_ind = ind_1[index1]
    val_set, dummy = torch.cat((ind_0_ind, ind_1_ind)).sort()

    a = []
    # remove ind_0_ind from ind_0
    for i in range(len(ind_0)):
        if ind_0[i] not in ind_0_ind:
            a.append(ind_0[i])

    ind_0 = torch.tensor(a)

    a = []
    # remove ind_0_ind from ind_0
    for i in range(len(ind_1)):
        if ind_1[i] not in ind_1_ind:
            a.append(ind_1[i])

    ind_1 = torch.tensor(a)
    train_set, dummy = torch.cat((ind_0, ind_1)).sort()
    # print(train_set,val_set,test_set)

    return train_set, val_set, test_set

def getSamplesSplit(y,index,num=50):
    # ind_0 = torch.tensor([2,1,2])
    # print(data[ind_0])
    # print(y)
    ind_0 = torch.nonzero(y == 0).reshape(-1)
    ## removing indexed which are not present in index
    ind_0 = ind_0[[(i in index) for i in ind_0]]
    # print(ind_0)
    ind_1 = torch.nonzero(y == 1).reshape(-1)
    ind_1 = ind_1[[(i in index) for i in ind_1]]
    
    # print(ind_1)
    ind_0_ind = list(torch.utils.data.RandomSampler(ind_0, replacement=True, num_samples=num))
    ind_0 = ind_0[ind_0_ind]
    # print(ind_0_ind,ind_0)
    ind_1_ind = list(torch.utils.data.RandomSampler(ind_1, replacement=True, num_samples=num))
    ind_1 = ind_1[ind_1_ind]
    # print(ind_1_ind,ind_1)
    # print(list(indexes))
    indexes, dummy = torch.cat((ind_0, ind_1)).sort()
    # print(indexes)
    # print(x[indexes])

    return indexes


def getSamples(y,num=50):
    # ind_0 = torch.tensor([2,1,2])
    # print(data[ind_0])
    # print(y)
    ind_0 = torch.nonzero(y == 0).reshape(-1)
    ## removing indexed which are not present in index
    # print(ind_0)
    ind_1 = torch.nonzero(y == 1).reshape(-1)
    
    # print(ind_1)
    ind_0_ind = list(torch.utils.data.RandomSampler(ind_0, replacement=True, num_samples=num))
    ind_0 = ind_0[ind_0_ind]
    # print(ind_0_ind,ind_0)
    ind_1_ind = list(torch.utils.data.RandomSampler(ind_1, replacement=True, num_samples=num))
    ind_1 = ind_1[ind_1_ind]
    # print(ind_1_ind,ind_1)
    # print(list(indexes))
    indexes, dummy = torch.cat((ind_0, ind_1)).sort()
    # print(indexes)
    # print(x[indexes])

    return indexes

def plot(pred,actual,path):
    df=pd.DataFrame()
    df["edge_type"] = actual
    df["Edge_Index"] = [i for i in range(len(actual))]
    df["Non_Cut_Edge_Probability"] = pred
    x=[i for i in range(len(pred))]
    df['type'] = np.where(df['edge_type']< 0.5, "Cut_edge","Non_Cut_edge" )
    plt.scatter(x,actual, color ='tab:orange',label='Actual')
    sns.set_palette(['cyan','red'])
    sns.scatterplot(x="Edge_Index", y="Non_Cut_Edge_Probability", hue=df.type.tolist(),palette=sns.color_palette(),data=df,s=20).set(title="Edge Projection")
    plt.draw()
    plt.savefig(f'{path}/Visualize.png')
    plt.clf()
    
def calculate(pred,actual,threshold):
    out = (pred>threshold).float()
    print(out)
    out=out.type(torch.int64)
    uniq,count=torch.unique(out,return_counts=True) 
    print(uniq,count)
    actual=actual.type(torch.int64)
    ## if diff==1 predict non-cut actual cut  very bad
    ## if diff=-1 predict cut actual non-cut
    diff=out-actual
    uniq,count=torch.unique(diff,return_counts=True)   
    print(uniq,count)
    pass

def confusion_matrix(pred,actual,threshold):
    out = (pred>threshold).float()
    tp = torch.sum(out*actual)
    tn = torch.sum((1-out)*(1-actual))
    fp = torch.sum(out*(1-actual))
    fn = torch.sum((1-out)*actual)
    return tp,tn,fp,fn

def accuracy(out,actual,threshold):
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred == actual  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(len(out))  # Derive ratio of correct predictions.

def precision(pred,actual,threshold):
    tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
    return tp/(tp+fp)
    
def recall(pred,actual,threshold):
    tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
    return tp/(tp+fn)

def f1_score(pred,actual,threshold):
    tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
    return 2*tp/(2*tp+fp+fn)

def plot_confusion_matrix(pred,actual,threshold,path):
    tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
    plt.figure(figsize=(10,10))
    plt.imshow([[tp,fp],[fn,tn]],cmap=plt.cm.Blues)
    plt.xticks([0,1],['Positive','Negative'])
    plt.yticks([0,1],['Positive','Negative'])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.draw()
    plt.savefig(f'{path}/confusion_matrix.png')
    plt.clf()

def plot_roc_curve(pred,actual,path):
    for threshold in range(0,100):
        threshold = threshold/100
        tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
        plt.scatter(fp/(fp+tn),tp/(tp+fn),color='tab:blue')
    plt.xlabel('TP rate')
    plt.ylabel('FP rate')
    plt.title("Roc-curve")
    plt.draw()
    plt.savefig(f'{path}/roc_curve.png')
    plt.clf()


def plot_precision_recall(pred,actual,path):
    for threshold in range(0,100):
        threshold = threshold/100
        tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
        plt.scatter(tp/(tp+fp),tp/(tp+fn),color='tab:blue')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title("PR-curve")
    plt.draw()
    plt.savefig(f'{path}/precision_recall.png')
    plt.clf()
    
def get_auc(pred,actual):
    auc = 0
    for threshold in range(0,100):
        threshold = threshold/100
        tp,tn,fp,fn = confusion_matrix(pred,actual,threshold)
        auc += (fp/(fp+tn))*(tp/(tp+fn))
    return auc

def plot_TSNE(actual,embeddings,path):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    # plot using tsne
    plt.figure(figsize=(10,10))
    plt.scatter(x = tsne_results[:,0],y = tsne_results[:,1], hue = actual, cmap=plt.cm.Blues).set(title='TSNE')
    plt.draw()
    plt.savefig(f'{path}/tsne.png')
    plt.clf()
    return tsne_results

def distribute(g, cuts):
    # 0 -> core nodes
    # 1 -> near surface nodes
    # 2 -> surface nodes
    node_type = {i:0 for i in g.nodes()}
    for edge in g.edges():
        if (edge[0] in list(g.nodes)) and (edge[1] in list(g.nodes)) and cuts[edge[0]]!=cuts[edge[1]]:
            # print(edge)
            node_type[edge[0]] = 2
            node_type[edge[1]] = 2
    
    for i in g.nodes():
        if node_type[i]==2:
            for j in g.neighbors(i):
                if node_type[j]==0:
                    node_type[j]=1
    return node_type

def plot_node_TSNE(embeddings, path, original = False):

    if not original:
        # load from pkl
        embeddings = pkl.load(open('node_embeddings.pkl','rb'))

    graph=nx.read_edgelist(path+'/graph.txt')
    if type(list(graph.nodes())[0])==str:
        mapping= {i:int(i) for i in graph.nodes()}
        graph=nx.relabel_nodes(graph,mapping)
    cuts = torch.load(path+'/cuts.pt',map_location=torch.device('cpu'))
    node_type = distribute(graph, cuts)
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    df = pd.DataFrame()
    df["node_type"] = node_type
    df["x1"] = tsne_results[:,0]
    df["x2"] = tsne_results[:,1]
    sns.scatterplot(x="x1", y="x2", hue=df.node_type.tolist(),palette=sns.color_palette("hls", 3),data=df).set(title="Node T-SNE projection")
    plt.draw()
    plt.savefig(f'{path}/node_tsne_{original}.png')
    plt.clf()
    return tsne_results

def plot_edge_TSNE(data, path, original=False):
    node_embedding = data.x
    edge_index = data.edge_index
    edge_type = data.y

    edge_embedding = torch.cat([node_embedding[edge_index[0]],node_embedding[edge_index[1]]],dim=1)
    print(edge_embedding.shape)

    if not original:
        # load from pkl
        edge_embedding = pkl.load(open('edge_embeddings.pkl','rb'))
        print("not original", edge_embedding.shape)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(edge_embedding)
    df = pd.DataFrame()
    df["edge_type"] = edge_type
    df["x1"] = tsne_results[:,0]
    df["x2"] = tsne_results[:,1]
    sns.scatterplot(x="x1", y="x2", hue=df.edge_type.tolist(),palette=sns.color_palette("hls", 2),data=df).set(title="Edge T-SNE projection")
    plt.draw()
    plt.savefig(f'{path}/edge_tsne_{original}.png')
    plt.clf()
    return tsne_results

def gap_loss_sparse(Y, A,device):
    '''
    loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''
    D = torch.sparse.sum(A, dim=1).to_dense()
    Gamma = torch.mm(Y.t(), D.unsqueeze(1).float())
    YbyGamma = torch.div(Y, Gamma.t())
    Y_t = (1 - Y).t()
    loss = torch.sum(torch.mm(YbyGamma, Y_t) * A.to_dense()).to(device)
    return loss

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

def pca_embedding(graph,comp):
    # print(adj)
    A = nx.adjacency_matrix(graph)
    
    A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    adj = sparse_mx_to_torch_sparse_tensor(norm_adj) # SciPy to Torch sparse
    adj=adj.to_dense()
    normalized_features= StandardScaler().fit_transform(adj)
    pca = PCA(n_components=comp)
    principalComponents = pca.fit_transform(normalized_features)
    pcas=torch.from_numpy(principalComponents)
    return pcas.float()

def edge_types(folder,trained_on,pred,actual,threshold):   
    dic_fp = {}
    dic_fn = {}
    dic_tp = {}
    dic_tn = {}
    tot_edge = {}
    print(folder)
    g=nx.read_edgelist(folder+f'/1/graph.txt')
    if type(list(g.nodes())[0])==str:
        mapping= {i:int(i) for i in g.nodes()}
        g=nx.relabel_nodes(g,mapping)
    if (trained_on=='gap'):
        cuts = torch.load(folder+f'/1/cuts.pt',map_location=torch.device('cpu'))        # index = 1 
    elif (trained_on=='hmetis'):
        cuts=torch.from_numpy(np.loadtxt(folder+f'/1/cut_hmetis.txt'))
        cuts=cuts.long()
    else:
        print("Trained on not present")
        exit(0)
    node_type = distribute(g,cuts)
    for i,edge in enumerate(g.edges()):
        out = (pred[i]>threshold).int()
        l = [node_type[edge[0]],node_type[edge[1]]]
        l.sort()
        l = tuple(l)
        if out==1 and actual[i]==0:         # fp
            if l not in dic_fp:
                dic_fp[l]=1
            else:
                dic_fp[l]+=1
        elif out==0 and actual[i]==1:          # fn
            if l not in dic_fn:
                dic_fn[l]=1
            else:
                dic_fn[l]+=1
        elif out==1 and actual[i]==1:          # tp
            if l not in dic_tp:
                dic_tp[l]=1
            else:
                dic_tp[l]+=1
        elif out==0 and actual[i]==0:          # tn
            if l not in dic_tn:
                dic_tn[l]=1
            else:
                dic_tn[l]+=1
        else:
            print("Should not be here(some bug)")
            exit(0)
        
        if l not in tot_edge:
            tot_edge[l]=1
        else:
            tot_edge[l]+=1
        
    # for i in tot_edge.keys():
    #     if i in dic_fp:
    #         dic_fp[i]=round(dic_fp[i]/tot_edge[i],3)
    #     if i in dic_fn:
    #         dic_fn[i]=round(dic_fn[i]/tot_edge[i],3)
    #     if i in dic_tp:
    #         dic_tp[i]=round(dic_tp[i]/tot_edge[i],3)
    #     if i in dic_tn:
    #         dic_tn[i]=round(dic_tn[i]/tot_edge[i],3)
    print("False Positives: ")
    print(dic_fp)
    print("False Negatives: ")
    print(dic_fn)
    print("True Positives: ")
    print(dic_tp)
    print("True Negatives: ")
    print(dic_tn)


def plot_node_fea_analysis(dict, node_type, filename):
    # draw scatter plot with values from dict
    # node_type is a list of 0,1,2
    # print(node_type)
    # print(dict.values())
    color=['red','green','blue']
    label=['Core','Near Surface','Surface']
    marker=['o','o','o']
    data=[]
    for j in range(3):
        index=[]
        for i in node_type:
            if node_type[i]==j:
                index.append(i)
        y_axis = [dict[index[i]] for i in range(len(index))] 
        data.append(y_axis)
        plt.scatter(index, y_axis, color=color[j],label=label[j],marker=marker[j],s=10)
    plt.legend()
    plt.xlabel("Node id")
    plt.ylabel("Value")
    # plt.show()
    plt.savefig(filename)
    plt.clf()
    # plt.boxplot(data)
    # plt.xticks([1,2,3],['Core','Near Surface','Surface'])
    # plt.savefig(filename+"box.png")
    # plt.clf()

def plot_edge_fea_analysis(folder,trained_on):

    print(folder)
    filename = 'edge_fea_analysis.png'
    g=nx.read_edgelist(folder+f'/1/graph.txt')
    if type(list(g.nodes())[0])==str:
        mapping= {i:int(i) for i in g.nodes()}
        g=nx.relabel_nodes(g,mapping)
    if (trained_on=='gap'):
        cuts = torch.load(folder+f'/1/cuts.pt',map_location=torch.device('cpu'))        # index = 1 
    elif (trained_on=='hmetis'):
        cuts=torch.from_numpy(np.loadtxt(folder+f'/1/cut_hmetis.txt'))
        cuts=cuts.long()
    else:
        print("Trained on not present")
        exit(0)

    dict_feature = edge_betweenness_centrality(g)

    edges = []
    for i in g.edges():
        edges.append(i)
    
    node_type = distribute(g,cuts)
    edge_type = ['Core-Core' for i in range(len(g.edges()))]
    for i,edge in enumerate(g.edges()):
        l = [node_type[edge[0]],node_type[edge[1]]]
        l.sort()
        l = tuple(l)
        if l==(0,0):
            edge_type[i]='Core-Core'
        elif l==(0,1):
            edge_type[i]='Core-Near Surface'
        elif l==(0,2):
            edge_type[i]='Core-Surface'
        elif l==(1,1):
            edge_type[i]='Near Surface-Near Surface'
        elif l==(1,2):
            edge_type[i]='Near Surface-Surface'
        elif l==(2,2):
            edge_type[i]='Surface-Surface'
    # draw scatter plot with values from dict
    # edge_type is a list of 0,1,2
    # print(edge_type)
    # print(dict.values())
    color=['red','green','blue','c','m','y']
    label=['Core-Core','Core-Surface','Surface-Surface','Core-Near Surface','Near Surface-Near Surface','Near Surface-Surface']
    marker=['o','o','o','o','o','o']
    data=[]
    for j in range(len(label)):
        index=[]
        for i in range(len(edge_type)):
            if edge_type[i]==label[j]:
                index.append(i)
        y_axis = [dict_feature[edges[index[i]]] for i in range(len(index))]
        print(label[j],len(y_axis)) 
        data.append(y_axis)
        plt.scatter(index, y_axis, color=color[j],label=label[j],marker=marker[j],s=10)
    plt.legend()
    plt.xlabel("Edge id")
    plt.ylabel("Value")
    # plt.show()
    plt.savefig(filename)
    plt.clf()
    

    # only plot for surface-surface edges to distinguish between intra and inter cluster
    filename = 'edge_fea_analysis_surface_surface.png'
    edge_type = ['Intra-Intra' for i in range(len(g.edges()))]
    for i,edge in enumerate(g.edges()):
        if node_type[edge[0]]==2 and node_type[edge[1]]==2:
            if cuts[edge[0]]==cuts[edge[1]]:
                edge_type[i]='Intra Cluster'
            else:
                edge_type[i]='Inter Cluster'
        else:
            edge_type[i]='Others'

    color=['red','green']
    label=['Intra Cluster','Inter Cluster']
    marker=['o','o']
    data=[]
    for j in range(len(label)):
        index=[]
        for i in range(len(edge_type)):
            if edge_type[i]==label[j]:
                index.append(i)
        y_axis = [dict_feature[edges[index[i]]] for i in range(len(index))] 
        print(label[j],len(y_axis))
        data.append(y_axis)
        plt.scatter(index, y_axis, color=color[j],label=label[j],marker=marker[j],s=10)
    plt.legend()
    plt.xlabel("Edge id")
    plt.ylabel("Value")
    # plt.show()
    plt.savefig(filename)
    plt.clf()



def plot_edge_fea_missclassified(folder,trained_on, pred, actual, threshold):

    print(folder)
    
    g=nx.read_edgelist(folder+f'/1/graph.txt')
    if type(list(g.nodes())[0])==str:
        mapping= {i:int(i) for i in g.nodes()}
        g=nx.relabel_nodes(g,mapping)
    if (trained_on=='gap'):
        cuts = torch.load(folder+f'/1/cuts.pt',map_location=torch.device('cpu'))        # index = 1 
    elif (trained_on=='hmetis'):
        cuts=torch.from_numpy(np.loadtxt(folder+f'/1/cut_hmetis.txt'))
        cuts=cuts.long()
    else:
        print("Trained on not present")
        exit(0)

    feature_names = ['clustering_coefficients','degree_centrality','closeness_centrality','betweenness_centrality']

    node_type = distribute(g,cuts)
    edges = []
    for i in g.edges():
        edges.append(i)

    for fea in feature_names:
        if fea=='clustering_coefficients':
            dict_feature = nx.clustering(g)
        elif fea=='degree_centrality':
            dict_feature = nx.degree_centrality(g)
        elif fea=='closeness_centrality':
            dict_feature = nx.closeness_centrality(g)
        elif fea=='betweenness_centrality':
            dict_feature = nx.betweenness_centrality(g)

        filename = 'edge_'+fea+'_missclass.png'

        # surface-surface edge with 4 labels - tp, tn, fp, fn
        edge_type = ['TP' for i in range(len(g.edges()))]
        edge_feature = [0 for i in range(len(g.edges()))]
        for i,edge in enumerate(g.edges()):
            # edge feature is avg of node features
            edge_feature[i] = (dict_feature[edge[0]]+dict_feature[edge[1]])/2
            if node_type[edge[0]]==2 and node_type[edge[1]]==2:         # surface-surface edge
                out = (pred[i]>threshold).int()
                if out==1 and actual[i]==1:
                    edge_type[i]='TP'
                elif out==0 and actual[i]==0:
                    edge_type[i]='TN'
                elif out==1 and actual[i]==0:
                    edge_type[i]='FP'
                else:
                    edge_type[i]='FN'
            else:
                edge_type[i]='Others'
        color = ['r','g','b','y']
        label = ['TP','TN','FP','FN']
        marker = ['o','o','o','o']

        data=[]
        for j in range(len(label)):
            index=[]
            for i in range(len(edge_type)):
                if edge_type[i]==label[j]:
                    index.append(i)
            y_axis = [edge_feature[index[i]] for i in range(len(index))]
            print(label[j],len(y_axis)) 
            data.append(y_axis)
            plt.scatter(index, y_axis, color=color[j],label=label[j],marker=marker[j],s=10)
        plt.legend()
        plt.xlabel("Edge id")
        plt.ylabel("Value")
        # plt.show()
        plt.savefig(filename)
        plt.clf()
    
def multi_class_confusion_matrix(probability,actual,path):
    # TODO should we apply softmax here 
    _,pred=torch.max(probability,1)  ## pred is of size [num_nodes]
    cm = sklearn.metrics.confusion_matrix(actual, pred)  ## 3*3
    # cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
    cm_df = pd.DataFrame(cm,
                     index = ['0','1'], 
                     columns = ['0','1'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True,fmt='0.2f')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(f'{path}/cm.png')
    plt.clf()

def plot_core_prob(folder,prob,g):
    # Apply softmax on prob
    core_values=F.softmax(prob,dim=1)[:,1]         # convert into probability: 0 is non-core prob, 1 is core-prob
    tgraph=nx.read_edgelist(g)
    if type(list(tgraph.nodes())[0])==str:
        mapping= {i:int(i) for i in tgraph.nodes()}
        tgraph=nx.relabel_nodes(tgraph,mapping)
    l=sorted(tgraph.nodes())
    graph=nx.Graph()
    graph.add_nodes_from(l)
    graph.add_edges_from(tgraph.edges())
    colors=['#ff0000','#e50000','#cc0000','#b20000','#990000','#7f0000','#660000','#4c0000','#330000','#190000','#090000','#000000']
        
    # assign node color of increasing intensity based on core prob
    node_color=[]
    for i in graph.nodes():
        node_color.append(colors[int(core_values[i]*10)])
    # print("core_valuesss:   ")
    # for i,j in enumerate(core_values):
    #     print(i,int(100*j),end=": ")
    # print()

    if len(list(graph.nodes())) < 101:
        nx.draw(graph,node_color=node_color,with_labels=True,node_size=20)
        
    else:
        nx.draw(graph,node_color=node_color,with_labels=False,node_size=20)        
    plt.savefig(f'{folder}/core_prob.png')
    plt.clf()
    
