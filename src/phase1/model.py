import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from utils import *
import pickle as pkl
import matplotlib.pyplot as plt



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features,num_layers,device):
        super().__init__()
        torch.manual_seed(1234567)
        self.num_layers=num_layers

        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)

        self.conv_layers=[GCNConv(num_features, hidden_channels).to(device).double()]
        self.batchnorms=[torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)]
        
        # self.conv1 = GCNConv(num_features, hidden_channels)
        # self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        for i in range(num_layers-1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels).to(device).double())
            self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)) 
        # batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)) 
        # self.conv_layers=torch.tensor(self.conv_layers)
        # mlp
        self.lin1_1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm4 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        self.lin1_2 = Linear(hidden_channels, 2)

        # # feature edge mlp
        # self.edge_mlp = Linear(num_edge_features,1)

    def forward(self, x, edge_index, final=False):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue 
        # # print(x.sha
        # pe)
        # print(edge_index.shape)
        # print(x, edge_index)
        for i in range(self.num_layers):
            # print(i,self.conv_layers[i].to(device))
            # print(x.device)
            x=self.conv_layers[i](x,edge_index)
            x=self.batchnorms[i](x)
            x=x.relu()

        # x = self.conv1(x, edge_index)
        # x = self.batch_norm1(x)
        # x = x.relu()
        # # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)     
        # x = self.batch_norm2(x)      
        # x = x.relu()

        # x = self.conv3(x, edge_index)     
        # x = self.batch_norm3(x)      
        # x = x.relu()              # node embeddings
        if final:
            pkl.dump(x, open('node_embeddings.pkl', 'wb'))
        
        # feature_edge = self.edge_mlp(feature_edge)

        ## TODO - batch norm here, also check affine=True/False

        # if final:
        #     pkl.dump(edge_embedding_1, open('edge_embeddings.pkl', 'wb'))

        #mlp
        x = self.lin1_1(x)
        x = self.batch_norm4(x)      
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin1_2(x)

        return x



class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features,num_edge_features ,heads,num_layers,device):
        super().__init__()
        torch.manual_seed(1234567)
        
        self.num_layers=num_layers
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)

        self.conv_layers=[GATConv(num_features, hidden_channels, edge_dim = 1, heads=heads).to(device).double()]
        self.batchnorms=[torch.nn.BatchNorm1d(hidden_channels*heads,track_running_stats=False,affine=False).to(device)]
        self.heads=heads
        for i in range(num_layers-1):
            self.conv_layers.append(GATConv(self.heads*hidden_channels, hidden_channels , head=1).to(device).double())
            self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)) 
            self.heads=1


        # self.conv1 = GATConv(num_features, hidden_channels, edge_dim = 1, heads=heads)
        # self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels*heads, track_running_stats=False,affine=False)
        # self.conv2 = GATConv(heads*hidden_channels, hidden_channels , head=1)
        # self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False,affine=False)
        # self.conv3 = GATConv(hidden_channels, hidden_channels , head=1)
        # self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels, track_running_stats=False,affine=False)

        # mlp
        self.lin1_1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm4 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        self.lin1_2 = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, final=False):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue 
        # # print(x.sha
        # pe)
        # print(edge_index.shape)
        # print(x, edge_index)
        # x = self.conv1(x, edge_index, edge_attr=feature_edge)
        # x = self.batch_norm1(x)
        # x = x.relu()
        # # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)     
        # x = self.batch_norm2(x)      
        # x = x.relu()

        # x = self.conv3(x, edge_index)     
        # x = self.batch_norm3(x)      
        # x = x.relu()              # node embeddings
        # if final:
        #     pkl.dump(x, open('node_embeddings.pkl', 'wb'))

        for i in range(self.num_layers):
            x=self.conv_layers[i](x,edge_index)
            x=self.batchnorms[i](x)
            x=x.relu()


        # feature_edge = self.edge_mlp(feature_edge)
        
        # generate edge embedding by concatenating node embeddings
        # edge_embedding_1 = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        # edge_embedding_1 = torch.cat([edge_embedding_1, feature_edge], dim=1)  # add edge features
        # edge_embedding_2 = torch.cat([x[edge_index[1]], x[edge_index[0]]], dim=1)
        # edge_embedding_2 = torch.cat([edge_embedding_2, feature_edge], dim=1)  # add edge features

        # ## TODO - batch norm here, also check affine=True/False

        # if final:
        #     pkl.dump(edge_embedding_1, open('edge_embeddings.pkl', 'wb'))

        # #mlp
        # x_1 = self.lin1_1(edge_embedding_1)
        # x_1 = x_1.relu()
        # # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        # x_1 = self.lin2_1(x_1)
        # # x_1 = x_1.sigmoid()     # edge prediction - make the output between 0 and 1

        # x_2 = self.lin1_2(edge_embedding_2)
        # x_2 = x_2.relu()
        # # x_2 = F.dropout(x_2, p=0.5, training=self.training)
        # x_2 = self.lin2_2(x_2)

        # # Using BCEWithLogitsLoss, so sigmoid is not needed
        # # x_2 = x_2.sigmoid()     # edge prediction - make the output between 0 and 1

        # # x = avg of x1,x2
        # x = (x_1 + x_2) / 2
        # return x
        x = self.lin1_1(x)
        x = self.batch_norm4(x)      
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin1_2(x)

        return x


class Graphsage(torch.nn.Module):
    def __init__(self, hidden_channels, num_features,num_layers,device):
        super().__init__()
        # print("graphsage used")
        torch.manual_seed(1234567)
        
        self.num_layers=num_layers
        self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)
        
        self.conv_layers=[SAGEConv(num_features, hidden_channels).to(device).double()]
        self.batchnorms=[torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)]
        
        
        # self.conv1 = SAGEConv(num_features, hidden_channels)
        # self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        # self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        
        for i in range(num_layers-1):
            self.conv_layers.append(SAGEConv(hidden_channels, hidden_channels).to(device).double())
            self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)) 
        
        # mlp
        self.lin1_1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm4 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        self.lin1_2 = Linear(hidden_channels, 2)

        # # feature edge mlp
        # self.edge_mlp = Linear(num_edge_features,1)

    def forward(self, x, edge_index, final=False):
        # initial batch norm
        # x = self.batch_norm0(x) ## TODO : eval issue 
        # # print(x.sha
        # pe)
        # print(edge_index.shape)
        # print(x, edge_index)
        # x = self.conv1(x, edge_index)
        # x = self.batch_norm1(x)
        # x = x.relu()
        # # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)     
        # x = self.batch_norm2(x)      
        # x = x.relu()

        # x = self.conv3(x, edge_index)     
        # x = self.batch_norm3(x)      
        # x = x.relu()              # node embeddings
        if final:
            pkl.dump(x, open('node_embeddings.pkl', 'wb'))
        
        # feature_edge = self.edge_mlp(feature_edge)

        ## TODO - batch norm here, also check affine=True/False

        # if final:
        #     pkl.dump(edge_embedding_1, open('edge_embeddings.pkl', 'wb'))

        for i in range(self.num_layers):
            x=self.conv_layers[i](x,edge_index)
            x=self.batchnorms[i](x)
            x=x.relu()


        #mlp
        x = self.lin1_1(x)
        x = self.batch_norm4(x)      
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin1_2(x)

        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_features,device):
        
        super().__init__()
        
        torch.manual_seed(1234567)
        # self.num_layers=num_layers

        # self.batch_norm0 = torch.nn.BatchNorm1d(num_features,track_running_stats=False,affine=False)

        # self.conv_layers=[GCNConv(num_features, hidden_channels).to(device).double()]
        # self.batchnorms=[torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)]
        
        # self.conv1 = GCNConv(num_features, hidden_channels)
        # self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        # for i in range(num_layers-1):
        #     self.conv_layers.append(GCNConv(hidden_channels, hidden_channels).to(device).double())
        #     self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False).to(device)) 
        # # batchnorms.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)) 
        # self.conv_layers=torch.tensor(self.conv_layers)
        # mlp
        self.lin1_0 = Linear(num_features, hidden_channels)
        self.batch_norm0 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        self.lin1_1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)
        self.lin1_2 = Linear(hidden_channels, 2)

        # # feature edge mlp
        # self.edge_mlp = Linear(num_edge_features,1)

    def forward(self, x, edge_index, final=False):
        x = self.lin1_0(x)
        x = self.batch_norm0(x)      
        x = x.relu()

        x = self.lin1_1(x)
        x = self.batch_norm1(x)      
        x = x.relu()
        # x_1 = F.dropout(x_1, p=0.5, training=self.training)
        x = self.lin1_2(x)

        return x


