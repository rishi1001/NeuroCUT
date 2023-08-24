import math
import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time

gpu=3
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        W = self.weight
        b = self.bias

        HW = torch.mm(H, W)
        # AHW = SparseMM.apply(A, HW)
        AHW = torch.spmm(A, HW)
        if self.bias is not None:
            return AHW + b
        else:
            return AHW

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):

    def __init__(self, gl, ll, dropout):
        super(GCN, self).__init__()
        if ll[0] != gl[-1]:
            assert 'Graph Conv Last layer and Linear first layer sizes dont match'
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.graphlayers = nn.ModuleList([GraphConvolution(gl[i], gl[i+1], bias=True) for i in range(len(gl)-1)])
        self.linlayers = nn.ModuleList([nn.Linear(ll[i], ll[i+1]) for i in range(len(ll)-1)])

    def forward(self, H, A):
        #print("In Model forward")
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        for idx, hidden in enumerate(self.graphlayers):
            H = F.relu(hidden(H,A))
            if idx < len(self.graphlayers) - 2:
                H = F.dropout(H, self.dropout, training=self.training)

        H_emb = H

        for idx, hidden in enumerate(self.linlayers):
            H = F.relu(hidden(H))

        # print(H)
        #print("Out Model forward")
        return F.softmax(H, dim=1)

    def __repr__(self):
        return str([self.graphlayers[i] for i in range(len(self.graphlayers))] + [self.linlayers[i] for i in range(len(self.linlayers))])


class CutLoss(torch.autograd.Function):         # TODO understand the arugments
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    @staticmethod
    def forward(ctx, Y, A):                 # TODO understand the ctx
        print("In Forward Cutloss")
        ctx.save_for_backward(Y,A)
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        YbyGamma = torch.div(Y, Gamma.t())
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss = torch.tensor([0.], requires_grad=True).to(device)
        idx = A._indices()
        data = A._values()
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i]
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
        #print("Out Forward cutloss")
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        print("In Backward Cutloss")
        Y, A, = ctx.saved_tensors
        idx = A._indices()
        data = A._values()
        D = torch.sparse.sum(A, dim=1).to_dense()
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        # print(Gamma.shape)
        gradient = torch.zeros_like(Y)
        print(gradient.shape)

        for i in tqdm(range(gradient.shape[0]), position=0, leave=True):
        #for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                alpha_ind = (idx[0, :] == i).nonzero()
                alpha = idx[1, alpha_ind]
                A_i_alpha = data[alpha_ind]
                st=time.time()
                temp = A_i_alpha / torch.pow(Gamma[j], 2) * (Gamma[j] * (1 - 2 * Y[alpha, j]) - D[i] * (
                            Y[i, j] * (1 - Y[alpha, j]) + (1 - Y[i, j]) * (Y[alpha, j])))
                ed=time.time()
                print("LINE 1 ","{:.2f}".format(ed-st))
                gradient[i, j] = torch.sum(temp)

                l_idx = list(idx.t())
                l2 = []
                l2_val = []
                # [l2.append(mem) for mem in l_idx if((mem[0] != i).item() and (mem[1] != i).item())]
                st=time.time()
                
                for ptr, mem in enumerate(l_idx):
                    if ((mem[0] != i).item() and (mem[1] != i).item()):
                        l2.append(mem)
                        l2_val.append(data[ptr])
                ed=time.time()
                extra_gradient = 0
                print("LINE 2 ","{:.2f}".format(ed-st))
                st=time.time()
                if (l2 != []):
                    for val, mem in zip(l2_val, l2):
                        extra_gradient += (-D[i] * torch.sum(
                            Y[mem[0], j] * (1 - Y[mem[1], j]) / torch.pow(Gamma[j], 2))) * val

                gradient[i, j] += extra_gradient
                ed=time.time()
                print("LINE 3 ","{:.2f}".format(ed-st))

        # print(gradient)
        #print("Out Backward Cutloss")
        return gradient, None