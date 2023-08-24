import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from utils import *
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from scipy import stats
import re
from torch_geometric.utils import from_networkx
import json
class dataset(Dataset):
    def __init__(self,folder,embeding,trained_on,norm,device='cpu',anchors=100):
        super().__init__()
        self.folder=folder
        self.embeding=embeding
        self.anchors=anchors
        self.trained_on=trained_on
        self.device=device
        self.norm=norm

        # TODO make generic
        self.num_no_features=5           # change when we add more node features
        self.num_ed_features=1            # change when we add more edge features

        self.generate_features(1)       
        print("Total Node Features : ", self.num_no_features)


    def read_num_nodes(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        num_nodes=js['num_nodes']
        return num_nodes
    
    def generate_edge_index(self,index):
        num_nodes=self.read_num_nodes(index)
        tgraph=nx.read_edgelist(self.folder+f'/{index}/graph.txt', nodetype=int)
        graph=nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(tgraph.edges())
        
        d = from_networkx(graph)
        return d.edge_index
        # A = nx.adjacency_matrix(self.graph)
        # A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        # norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        # adj = sparse_mx_to_torch_sparse_tensor(norm_adj) # SciPy to Torch sparse
        # As = sparse_mx_to_torch_sparse_tensor(A)

    def generate_features(self,index):
        if (self.embeding=="coefficents" ):
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.norm}.pkl'
        elif (self.embeding=='spectral'):
            filename=self.folder+f'/{index}/features_spectral_5.pkl'
        else:
            filename=self.folder+f'/{index}/features_{self.embeding}_{self.anchors}_{self.norm}.pkl'
        # print(filename)
        # exit(0)
        if os.path.exists(filename):
            # print("Loading from pickle file")
            features = torch.load(open(filename,"rb"))
        else:
            num_nodes=self.read_num_nodes(index)
            tgraph=nx.read_edgelist(self.folder+f'/{index}/graph.txt', nodetype=int)
            graph=nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(tgraph.edges())
            # features=[list(cluster_centrality(graph).values()),list(degree_centrality(graph).values()),list(betweenness_centrality(graph).values()),list(closeness_centrality(graph).values()),list(core_number(graph).values())]
            # print(list(nx.selfloop_edges(graph)))
            features_list=["cluster_centrality","degree_centrality","betweenness_centrality","closeness_centrality","kcore"]  # trying with 3

            features = []
            for feature_name in features_list:
                if feature_name=='cluster_centrality':
                    feat=cluster_centrality(graph)
                elif feature_name=='closeness_centrality':
                    feat=closeness_centrality(graph)
                elif feature_name=='degree_centrality':
                    feat=degree_centrality(graph)
                elif feature_name=='betweenness_centrality':
                    feat=betweenness_centrality(graph)
                elif feature_name=='kcore':
                    feat=core_number(graph)
                feat = [feat[i] for i in range(len(feat))]
                features.append(feat)

            features=torch.tensor(features).t()     ### features (num_nodes,4)
            # print("shape after transpose", features.shape)
            if self.embeding=='Lipschitz_rw_only':
                lip=lipschitz_rw_embedding(graph,self.anchors,len(graph.nodes()))
                features=lip
            if self.embeding=="coefficents_percentile":
                features=torch.tensor(features).t()
                temp=[(stats.rankdata(f, 'average')-1)/len(f) for f in features]
                features=torch.tensor(temp).t()
                # features=features.t()
                # print("sssss",features.shape)

            if (self.embeding=='Lipschitz_rw'):
                lip=lipschitz_rw_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.cat((features,lip),1)
                # sum=torch.sum(features,dim=0)
                # features=features/sum

            if (self.embeding=='Lipschitz_sp'):
                ## TODO add code for sp
                lip=lipschitz_embedding(graph,self.anchors,len(graph.nodes()))
                features=torch.cat((features,lip),1)
                # sum=torch.sum(features,dim=0)
                # features=features/sum

            if (self.embeding=='pca'):
                pca=pca_embedding(graph,self.anchors)
                features=torch.cat((features,pca),1)

                # minn=torch.min(features,dim=0)
                # sum=torch.sum(features,dim=0)
                # features=(features+minn.values)/sum
            # print("feature generated")
            features=features.type(torch.DoubleTensor)

            if self.norm=="MinMax":
                scaler=MinMaxScaler()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "Standard":
                scaler=StandardScaler()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "Normalizer":
                scaler=Normalizer()
                features=torch.tensor(scaler.fit_transform(features))
            elif self.norm == "percentile":
                features=torch.tensor(features).t()
                temp=[(stats.rankdata(f, 'average')-1)/len(f) for f in features]
                features=torch.tensor(temp).t()
            elif self.norm!="None":         # use this in case of percentile
                print("Norm not present")
                exit(0)

            with open(filename, "wb") as f:
                torch.save(features, f)
            # print(type(features))
        self.num_no_features=features.shape[1]  # set the total num of node features
        return features

    def generate_edge_features(self,index):
        filename=self.folder+f'/{index}/edge_features_{self.embeding}_{self.norm}.pkl'
        if os.path.exists(filename):
            # print("Loading from pickle file")
            edge_features = torch.load(open(filename,"rb"))
        else:
            num_nodes=self.read_num_nodes(index)
            tgraph=nx.read_edgelist(self.folder+f'/{index}/graph.txt', nodetype=int)
            graph=nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(tgraph.edges())

            edge_features=[list(edge_betweenness_centrality(graph).values())]
            edge_features=torch.tensor(edge_features).t()     ### edge_features (num_nodes,1)

            if self.embeding=="coefficents_percentile":
                edge_features=torch.tensor(edge_features).t()
                temp=[(stats.rankdata(f, 'average')-1)/len(f) for f in edge_features]
                edge_features=torch.tensor(temp).t()

            edge_features=edge_features.type(torch.DoubleTensor)

            if self.norm=="MinMax":
                scaler=MinMaxScaler()
                edge_features=torch.tensor(scaler.fit_transform(edge_features))
            elif self.norm == "Standard":
                scaler=StandardScaler()
                edge_features=torch.tensor(scaler.fit_transform(edge_features))
            elif self.norm == "Normalizer":
                scaler=Normalizer()
                edge_features=torch.tensor(scaler.fit_transform(edge_features))
            elif self.norm!="None":         # use this in case of percentile
                print("Norm not present")
                exit(0)

            with open(filename, "wb") as f:
                torch.save(edge_features, f)

        return edge_features


    def generate_y(self,index,edge_index):      # IMP: NOTE THAT WE NEED CORE/NON-CORE as training data
        if (self.trained_on=='gap'):
            cuts = torch.tensor(torch.load(self.folder+f'/{index}/core_surface.pt',map_location=torch.device('cpu')))
            cuts=cuts.long()

        elif (self.trained_on=='hmetis'):
            cuts = torch.tensor(torch.load(self.folder+f'/{index}/core_surface.pt',map_location=torch.device('cpu')))
            cuts=cuts.long()
        else:
            print("Trained on not present")
            exit(0)
        #TODO to remove combined
        # for i,j in enumerate(cuts):
        #     print(i,j.item(),end=" : ")
        # print()
        # exit(0)
        return cuts
        # g=nx.read_edgelist(self.folder+f'/{index}/graph.txt')
        # if type(list(g.nodes())[0])==str:
        #     mapping= {i:int(i) for i in g.nodes()}
        #     g=nx.relabel_nodes(g,mapping)
        # node_type = [0 for i in g.nodes()]
        # for edge in g.edges():
        #     if (edge[0] in list(g.nodes)) and (edge[1] in list(g.nodes)) and cuts[edge[0]]!=cuts[edge[1]]:
        #         # print(edge)
        #         node_type[edge[0]] = 2
        #         node_type[edge[1]] = 2

        # for i in g.nodes():
        #     if node_type[i]==2:
        #         for j in g.neighbors(i):
        #             if node_type[j]==0:
        #                 node_type[j]=1
        # return torch.tensor(node_type)

    def read_num_cuts(self,index):
        with open(self.folder+f'/{index}/graph_stats.txt') as f:
            data = f.read()
        js = json.loads(data)
        num_cuts=js['num_cuts']
        return num_cuts
    
    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        # print(idx)
        # return {'x':torch.tensor(self.dataset[idx][0]).type(torch.DoubleTensor),'y': torch.tensor(self.dataset[idx][1]).type(torch.DoubleTensor),'edge_weight':self.edge_weight,'edge_index':self.edge_index}
        x=[self.generate_features(idx+1)]
        # print(x[0].shape[0])
        # exit(0)
        ## TODO: why list
        # print(x[0].shape)
        edge_index=self.generate_edge_index(idx+1)
        #print(edge_index)
        y=self.generate_y(idx+1,edge_index)
        num_cuts=self.read_num_cuts(idx+1)
        return Data(x=x,edge_index=edge_index,y=y,num_nodes=x[0].shape[0],num_cuts=num_cuts).to(self.device)
