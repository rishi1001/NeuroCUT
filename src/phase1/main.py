import torch
import os
from model import GCN,GAT,Graphsage,MLP
from sklearn.utils.class_weight import compute_class_weight
import numpy
from utils import *
import networkx as nx
from datasets import dataset
from torch_geometric.loader import DataLoader
import argparse
import sys
from torchmetrics.classification import BinaryHingeLoss
from tqdm import tqdm
import torch.nn.functional as F
# from plot_features import plot_false_histo_features
import wandb
def set_seed(seed: int = 1) -> None:
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

set_seed()


parser = argparse.ArgumentParser(
    description="Command Line Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--embedding', type=str, default='coefficients',
                    help='What embedding to use (Lipschitz_rw or Lipschitz_sp or coefficients )')
parser.add_argument('--train', type=str, default='cora',
                help='Dataset to train')
parser.add_argument('--anchors', type=int, default=30, help='Number of Anchors for Lipschitz')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs train')
parser.add_argument('--model_name',type=str,default='None', help='Use whether gcnconv,gatconv or SageConv')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers used')
parser.add_argument('--trained_on',type=str,default='gap', help='Use whether gcnconv or gatconv')
parser.add_argument('--need_training',type=str,default='false', help='training or testing')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for inference')
parser.add_argument('--pos_weight', type=float, default=1, help='pos_weight for inference')
parser.add_argument('--num_times_samples', type=int, default=50, help='Number of time sample in train')
parser.add_argument('--num_edges_samples', type=int, default=50, help='Number of edges to sample in one time in train')
parser.add_argument('--norm',type=str,default='none', help='Use whether none,MinMax,Standard or Normalizer')
parser.add_argument('--gpu', type=int, default=4, help='gpu to train')

# parser.add_argument('--training', type=bool,default=False)
args = parser.parse_args()

# Variables of Arg Parser
num_epochs=args.epochs
dataset_path='../../data/'+args.train
embedding=args.embedding
anchors=args.anchors
model_name=args.model_name
num_layers=args.num_layers
temp=args.need_training.lower()
need_training= temp=='true'
threshold=args.threshold
num_times_samples=args.num_times_samples
num_edges_samples=args.num_edges_samples
trained_on=args.trained_on.lower()
norm=args.norm
pos_weight_val=args.pos_weight


gpu=args.gpu
if (gpu > 3 or gpu < 0): 
    print("Not a Valid gpu ")
    exit(0)
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)



# result_folder=f'../results/split_{args.train}_{args.embedding}_{args.norm}'
result_folder=f'../../results_phase1/{args.train}_{args.embedding}_{args.norm}_{model_name}_{num_layers}'

wandb_flag=False
if wandb_flag:
    wandb.init(project="phase1", name=f'{args.train}_{args.embedding}_{args.norm}_{model_name}_{num_layers}', config=args,entity='mincuts')

model_folder=f'../../models_phase1/{args.train}_{args.embedding}_{args.norm}_{model_name}_{num_layers}'
if (not os.path.exists(model_folder) ):
    need_training=True

os.makedirs(result_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(result_folder+"/train_set", exist_ok=True)
os.makedirs(result_folder+"/val_set", exist_ok=True)
os.makedirs(result_folder+"/test_set", exist_ok=True)

train_dataset=dataset(dataset_path+'/train_set',embedding,trained_on,norm,device,anchors)
val_dataset=dataset(dataset_path+'/val_set',embedding,trained_on,norm,device,anchors)
test_dataset=dataset(dataset_path+'/test_set',embedding,trained_on,norm,device,anchors)
# print("train",len(train_dataset))
# for i in range(len(train_dataset)):
#     print(train_dataset[i])
# print("val",len(val_dataset))
# for i in range(len(val_dataset)):
#     print(val_dataset[i])
# print("test",len(test_dataset))
# for i in range(len(test_dataset)):
#     print(test_dataset[i])
# exit(0)
# breakpoint()

dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)

if (model_name=='gat'):
    model = GAT(num_features=train_dataset.num_no_features, num_edge_features=train_dataset.num_ed_features , hidden_channels=16,heads=2,num_layers=num_layers,device=device).to(device)  # Note change num_features when needed
elif model_name=='gcn':
    model = GCN(num_features=train_dataset.num_no_features, hidden_channels=32,num_layers=num_layers,device=device).to(device)   # Note change num_features when needed
elif (model_name=='graphsage'):
    model = Graphsage(num_features=train_dataset.num_no_features, hidden_channels=32,num_layers=num_layers,device=device).to(device)   # Note change num_features when needed
elif (model_name=='mlp'):
    model = MLP(num_features=train_dataset.num_no_features, hidden_channels=32,device=device).to(device)   # Note change num_features when needed
else:
    print("No model found")
    exit(0)
if wandb_flag:
    wandb.watch(model)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=5e-6)
criterion = torch.nn.CrossEntropyLoss()          # TODO check this
# criterion = BinaryHingeLoss()
best_loss = -1
bestmodel = None
model=model.double()

def train(epoch):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()  # Clear gradients.
    # TODO batch wise training
    pbar=tqdm(dataloader)
    till_now=0
    for data in pbar:
        # print(data.x.shape)
        # exit(0)
        out = model(data.x[0], data.edge_index)
        # indexes=getSamplesSplit(data.y,train_ind,num_edges_samples) for split
        # loss = criterion(out[train_ind], data.y[train_ind])
        loss = criterion(out, data.y)         # TODO discuss this
        loss.backward()
        optimizer.step()        # TODO try doing this after 10 samples
        optimizer.zero_grad()  # Clear gradients.
        running_loss += loss.item()
        till_now+=1
        pbar.set_postfix({'Loss': running_loss/(till_now), 'Epoch': epoch+1})
    # running_loss/=num_times_samples
    print('epoch %d training loss: %.3f' % (epoch + 1, running_loss / (len(train_dataset))))

    return running_loss/len(train_dataset)

tp_list = []
fp_list = []
tn_list = []
fn_list = []
prec_list = []
recall_list = []

def test(test_set,test=False):         # (test false)validation or test set
    model.eval()    # TODO Due to removal of batch norm, results are different
    global best_loss
    global bestmodel
    running_loss = 0.0
    prec=0.0
    with torch.no_grad():
        for i in range(len(test_set)):
            data=test_set[i]
            # print(data.x.shape)
            out = model(data.x[0], data.edge_index)
            loss=criterion(out,data.y)
            running_loss += loss.item()
        if test:       
            print('Average Test loss: {:.3f}'.format(running_loss / (len(test_set))))
        else:
            print('Average Val loss: {:.3f}'.format(running_loss / (len(test_set))))


    ## TODO: Should precision be the metric
    if not test:
        if best_loss==-1 or running_loss < best_loss:
            best_loss=running_loss
            # Saving our trained model
            torch.save(model.state_dict(), model_folder+'/bestval.pth')
            bestmodel = model

    return running_loss/len(test_set)

# Calculate stats for dataset given
# Dataset : train_dataset,val_dataset,test_dataset
# type: train_set,val_set,test_set
def calculate_stats(dataset,type):   
    for i in range(len(dataset)):
        data=dataset[i]
        out = model(data.x[0], data.edge_index, final=False)
        # out = torch.sigmoid(out.reshape(-1))
        ## TODO : ADD function to evaluate out 
        result_path=f"{result_folder}/{type}/{i+1}"
        os.makedirs(result_path, exist_ok=True)

        # save the output in .pt file
        # print(out.shape)
        
        torch.save(out, f"{result_path}/train_node_core.pt")
        plot_core_prob(result_path,out,dataset_path+f'/{type}/{i+1}/graph.txt')
        orgininal_std=sys.stdout
        with open(f"{result_path}/stats.txt",'w') as f:
            sys.stdout=f
            # print("confusion: ",confusion_matrix(out,data.y,threshold))
            print("accuracy: ",accuracy(out,data.y,threshold))
            # print("precision: ",precision(out,data.y,threshold))
            # print("recall: ",recall(out,data.y,threshold))
            # print("f1_score: ",f1_score(out,data.y,threshold))
            # print("Auc: ",get_auc(out,data.y))
        sys.stdout=orgininal_std
        out=out.cpu()
        y=data.y.cpu()

        multi_class_confusion_matrix(out,y,result_path)
        plt.close()


if __name__ == '__main__':
    os.makedirs('../models', exist_ok=True)
    loss=[]
    val_loss=[]
    
    if (need_training):
        print("Started Training")

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # print('epoch ', epoch + 1)
            loss.append(train(epoch))
            val_loss.append(test(val_dataset))      # on validation set
            if wandb_flag:
                wandb.log({"Train_Loss":loss[-1],"Val_Loss":val_loss[-1]},step=epoch)
            torch.save(model.state_dict(), model_folder+'/lastmodel.pth')

        print('Performing test')
        test(test_dataset,test=True)      # on test set
        print('Finished Training')
    print("Calculating Stats")
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_folder+'/lastmodel.pth'))
        calculate_stats(train_dataset,"train_set")
        
        model.load_state_dict(torch.load(model_folder+'/bestval.pth'))
        # TODO (use bestval or last model?)
        # model.load_state_dict(torch.load(model_folder+'/lastmodel.pth'))
        calculate_stats(val_dataset,"val_set")
        calculate_stats(test_dataset,"test_set")

        if (need_training):
            ## plot loss
            plt.clf()
            plt.plot([i for i in range(len(loss))],loss,label="Train_Loss")
            plt.plot([i for i in range(len(val_loss))],val_loss,label="Val_Loss")
            plt.legend()
            plt.title("Loss vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.draw()
            plt.savefig(f"{result_folder}/loss.png")
            plt.close()

            # plot tp,fp,tn,fn,prec,recall
            # plt.clf()
            # plt.plot([i for i in range(len(tp_list))],tp_list,label="True Positives",color="b")
            # plt.plot([i for i in range(len(fp_list))],fp_list,label="False Positives",color="g")
            # plt.plot([i for i in range(len(tn_list))],tn_list,label="True Negatives",color="r")
            # plt.plot([i for i in range(len(fn_list))],fn_list,label="False Negatives",color="c")
            # plt.legend()
            # plt.title("Metrics vs Epoch")
            # plt.xlabel("Epoch")
            # plt.ylabel("Different Metrics")
            # plt.draw()
            # plt.savefig(f"{result_folder}/metrics_epochs.png")
            # plt.close()

            # plt.clf()
            # plt.plot([i for i in range(len(prec_list))],prec_list,label="Precision",color="b")
            # plt.plot([i for i in range(len(recall_list))],recall_list,label="Recall",color="g")
            # plt.legend()
            # plt.title("Precision,Recall vs Epoch")
            # plt.xlabel("Epoch")
            # plt.ylabel("Prec,Recal")
            # plt.draw()
            # plt.savefig(f"{result_folder}/pr_epochs.png")
            # plt.close()
        # plot_TSNE(data.y, data.x, result_path)
    # Saving our trained model
    #

# TODO train on 10 graph & overfit

#### Important at final inference time
# class_weight = compute_class_weight("balanced", classes=[0,1], y=dataset[0].y.numpy())
# out, loss = model(dataset[0].features, dataset[0].edge_index, dataset[0].y, class_weight, test=True)
# out = torch.sigmoid(out.reshape(-1))
# # out = model(dataset[0].features, dataset[0].edge_index).reshape(-1)
# print(out,loss)
## F1-score
## PR curve
## ROCE curve
