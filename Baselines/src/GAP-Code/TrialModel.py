from __future__ import division
from __future__ import print_function
from errno import ECHILD
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from scipy import sparse
from models import *
import os
import matplotlib.pyplot as plt
import sys

gpu=3
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
# print(device)
#print(device)
parser = argparse.ArgumentParser(
    description="Command Line Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
def parse():
    parser.add_argument('--embedding', type=str, default='given',
                    help='What embedding to use (Given, pca or Lipschitz)')
    parser.add_argument('--train', type=str, default='cora',
                    help='Dataset to train')
    parser.add_argument('--test', type=str, default='cora',
                    help='Dataset to test')
    parser.add_argument('--anchors', type=int, default=30, help='Number of Anchors for Lipschitz')
    parser.add_argument('--cuts', type=int, default=2, help='Number of desired cuts')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs train')
    parser.add_argument('--components', type=int, default=1000, help='Number of Components for PCA')
    # parser.add_argument('--training', type=bool,default=False)
    parser.add_argument("--training", type=str,default='false', help="Need training")
    args = parser.parse_args()
    return args


loss_train = []
loss_test = []
balance_train = []
balance_test = []
edge_cut_test = []
edge_cut_train = []
epoch_test = []
epoch_train = []

def Test(model, x, adj, A,filename,plot=False):
    '''
    Test Final Results
    '''
    global loss_test,balance_test,edge_cut_test
    Y = model(x, adj)
    # print("IN test")
    # print(Y)
    node_idx = test_partition(Y)
    torch.save(Y,filename+'/matrix.pt')
    torch.save(node_idx,filename+'/cuts.pt')
    np.savetxt(filename+'/cut_gap.txt',np.array(node_idx.cpu()), fmt='%d')     
    bin_count = torch.bincount(node_idx)
    # print("Clusters Info: ", bin_count)
    # balanceness_t = balanceness(Y)
    # edge_cut_t = edge_cut(Y,A)
    # # k = Y.shape[1]
    # # beam_search_results = beam_search(Y,A,k,1)            # TODO : beam search
    # print("balanceness: ",balanceness_t)
    # print("Edge Cut: ",edge_cut_t)
    # # print("Beam Search Results: ",beam_search_results)
    # # print(node_idx)
    # loss = custom_loss_sparse(Y,A,device)
    # print('Normalized Cut obtained using the above partition is : {0:.3f}'.format(loss.item()))
    # if plot:
    #     loss_test.append(loss.item())
    #     balance_test.append(balanceness_t.item())
    #     edge_cut_test.append(edge_cut_t.item())


max_epochs=1000
def Train(model, x, adj, A, optimizer,filename):
    '''
    Training Specifications
    '''
    global best_model
    min_loss = 10**17
    for epoch in (range(max_epochs)):
        Y = model(x, adj)
        #loss = CutLoss.apply(Y,A)           # forward pass - look at cutloss foward function
        print("Epoch ",epoch,end=" ")
        loss = custom_loss_sparse(Y,A,device)
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./{}/trial_weight.pt".format(filename))
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()   
        print('Loss = {}'.format(loss.item()))
        epoch_train.append(epoch)
        loss_train.append(loss.item())
        balance_train.append(balanceness(Y).item())
        edge_cut_tr = edge_cut(Y,A)
        # print(edge_cut_tr)
        edge_cut_train.append(edge_cut_tr.item())
        # if epoch%100==0:
        #     global epoch_test
        #     #Test(model,xt,adjt,At,plot=True)
        #     epoch_test.append(epoch)

def read_graph(args,train_set):
    st=time.time()
    features, adj, As, graph = load_data(train_set)
    #print(features)
    # featuresc, adjc, Asc = load_data("cora")
    # featuresci, adjci, Asci = load_data("citeseer")
    # featuresp, adjp, Asp = load_data("pubmed")
    
    ## For PCA
    if args.embedding=='pca':
        #print("PCA embeddings")
        if os.path.exists("./features/features_pca_{}_{}.pkl".format(train_set,args.components)):
            print("Loading from pickle file")
            features = torch.load(open("./features/features_pca_{}_{}.pkl".format(train_set,args.components), "rb")).to(device)
        else:
            features = pca_embedding(adj,args.components)
            with open("./features/features_pca_{}_{}.pkl".format(train_set,args.components), "wb") as f:
                torch.save(features, f)
    elif args.embedding=='lipschitz':
        ## For Lipschitz
        # if pickle file is present then dont call the function
        print("hreee")
        if os.path.exists("./features/features_lipschitz_{}_{}.pkl".format(train_set,args.anchors)):
            print("Loading from pickle file")
            features = torch.load(open("./features/features_lipschitz_{}_{}.pkl".format(train_set,args.anchors), "rb")).to(device)
        else:
            features=lipschitz_embedding(graph,args.anchors,features.shape[0])
            with open("./features/features_lipschitz_{}_{}.pkl".format(train_set,args.anchors), "wb") as f:
                torch.save(features, f)
    elif args.embedding=='lipschitz_rw':
        ## For Lipschitz
        # if pickle file is present then dont call the function
        if os.path.exists("./features/features_lipschitz_rw_{}_{}.pkl".format(train_set,args.anchors)):
            print("Loading from pickle file")
            features = torch.load(open("./features/features_lipschitz_rw_{}_{}.pkl".format(train_set,args.anchors), "rb")).to(device)
        else:
            print("here")
            features=lipschitz_rw_embedding(graph,args.anchors,features.shape[0])
            with open("./features/features_lipschitz_rw_{}_{}.pkl".format(train_set,args.anchors), "wb") as f:
                torch.save(features, f)
    
    elif args.embedding !='given':
            print("Wrong embedding input")
            exit(0)
    ed=time.time()
    print("Embeddings time for {} is {}".format(train_set,ed-st))
    return features,adj,As
    #print(features)

def plot_loss(filename):
    # plot loss vs epoch
    plt.plot(epoch_train,loss_train,label='Train')
    # plt.plot(epoch_test,loss_test,label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig("./{}/loss_epoch.png".format(filename))
    plt.clf()
    
    print("here")
    # print(type(epoch_train))
    # print(type(edge_cut_train[0]))
    plt.plot(np.array(epoch_train),np.array(edge_cut_train),label='Train')
    # print("here1")
    # plt.plot(epoch_test,edge_cut_test,label='Test')
    # print("here2")
    plt.xlabel('Epoch')
    plt.ylabel('Edge Cut')
    plt.title('Edge Cut vs Epoch')
    plt.legend()
    plt.savefig("./{}/edge_cut_epoch.png".format(filename))
    plt.clf()
    print("here")
    plt.plot(epoch_train,balance_train,label='Train')
    # plt.plot(epoch_test,balance_test,label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Balenceness')
    plt.title('Balenceness vs Epoch')
    plt.legend()
    plt.savefig("./{}/balance_epoch.png".format(filename))
    plt.clf()
    return

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Random seed set as {}".format(seed))


def main():
    
    set_seed()
    args=parse()
    '''
    Adjecency matrix and modifications
    '''
    # A = input_matrix()
    # # print(A)
    # # exit()
    # # Modifications
    # A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    # norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    # adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(device) # SciPy to Torch sparse
    # As = sparse_mx_to_torch_sparse_tensor(A).to(device)  # SciPy to sparse Tensor
    # A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to(device)   # SciPy to Torch Tensor
    # #print(A)

    

    # '''
    # Declare Input Size and Tensor
    # '''
    # N = A.shape[0]
    # d = 512

    # torch.manual_seed(100)
    # features = torch.randn(N, d)
    # np.savetxt("fextures.txt",features,delimiter= " ")
    # features = features.to(device)
    #print(features)
    
    
    ## LOSS From a file

    global max_epochs
    train_set=args.train
    test_set=args.test
    max_epochs=args.epochs
    #print(args.epochs)
    if args.embedding == 'lipschitz' or args.embedding == 'lipschitz_rw' :
        filename="../../Results/Gap/{}_{}_{}_{}_{}_{}".format(train_set,test_set,args.embedding,args.anchors, max_epochs,args.cuts)
        train_filename = "../../Results/Gap/train/{}_{}_{}_{}_{}".format(train_set,args.embedding,args.anchors, max_epochs,args.cuts)
    elif args.embedding == 'pca':
        train_filename = "../../Results/Gap/train/{}_{}_{}_{}_{}".format(train_set,args.embedding,args.components, max_epochs,args.cuts)
        filename="../../Results/Gap/{}_{}_{}_{}_{}_{}".format(train_set,test_set,args.embedding,args.components, max_epochs,args.cuts)
    else:
        train_filename = "../../Results/Gap/train/{}_{}_{}_{}".format(train_set,args.embedding, max_epochs,args.cuts)
        filename="../../Results/Gap/{}_{}_{}_{}_{}".format(train_set,test_set,args.embedding, max_epochs,args.cuts)
    

    need_training=args.training.lower()
    # print(need_training)
    # exit(0)

    if(need_training=="true" or (not os.path.exists("{}/trial_weight.pt".format(train_filename)))):  ## training is required
        print("Training on",train_set)
        os.makedirs(train_filename,exist_ok=True)
        sys.stdout = open(train_filename+'/stats.txt','w')
        features, adj, As = read_graph(args,train_set)
        N,d = features.shape            # N = 2708, d=1433 for cora N=3327,d=3703 for citeseer
        gl = [d, 64, 16]            # TODO might need to change this?
        ll = [16, args.cuts]                # TODO change this also ?
        model = GCN(gl, ll, dropout=0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
        
        print("Training on ",train_set)
        st=time.time()
        Train(model, features, adj, As, optimizer,train_filename)
        ed=time.time()
        print("Train time", ed-st)

        print("Testing on ",train_set)
        st=time.time()
        Test(model, features, adj, As,train_filename)
        ed=time.time()
        print("Test time for {} is {}".format(train_set,ed-st))
        
        plot_loss(train_filename)
        sys.stdout.close()

    if need_training=="false" :  
        os.makedirs(filename,exist_ok=True)
        sys.stdout = open(filename+'/stats.txt','w')
        print("Directly testing on",test_set)
        featurest, adjt, Ast = read_graph(args,test_set)
        N,d = featurest.shape            #N = 2708, d=1433 for cora N=3327,d=3703 for citeseer
        gl = [d, 64, 16]            #TODO might need to change this?
        ll = [16, args.cuts]                # TODO change this also ?
        model = GCN(gl, ll, dropout=0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
        model.load_state_dict(torch.load("./{}/trial_weight.pt".format(train_filename)))     # loading the best model
        print("Testing on ",test_set)
        st=time.time()
        Test(model, featurest, adjt, Ast,filename)
        ed=time.time()
        print("Test time for {} is {}".format(test_set,ed-st))
        sys.stdout.close()




    # print("Number of Nodes",N,d)    # N=19717 500 for pubmed
    # exit(0)
    '''
    
    Model Definition
    '''
    # print("Number of cuts",args.cuts)
    # print("Max_epochs",args.epochs)
    
    
    # plot the graphs
    #plot_loss(filename)
    # Test the best partition
    
    # print("Testing on ",train_set)
    # ed=time.time()
    # model.load_state_dict(torch.load("./{}/trial_weight.pt".format(filename)))     # loading the best model
    # Test(model, features, adj, As,"train",filename)
    # st=time.time()
    # print("Test time for {} is {}".format(train_set,st-ed))
    
    
    
    
    
if __name__ == '__main__':
    main()
