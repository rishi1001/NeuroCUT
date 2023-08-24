import networkx as nx
import scipy as sp
import pickle as pkl
import sys
import os
import re
import subprocess
from tqdm import tqdm
import json
import time
import numpy as np
def convert(filename, num_nodes,outfile,node_weights=None):
    # graph_file = pkl.load(open(file_name, 'rb'))
    # A = nx.adjacency_matrix(nx.from_dict_of_lists(graph_file))
    tgraph=nx.read_edgelist(filename, nodetype=int)

    G=nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(tgraph.edges())
    edges = []
    tot_edges = 0
    tot_nodes = G.number_of_nodes()

    for edge in G.edges():
        if edge[0]>edge[1]:
            edges.append((edge[1], edge[0]))
        else:
            edges.append((edge[0], edge[1]))
        tot_edges += 1
    edges.sort()
    f = open(outfile, 'w')
    if node_weights:
        f.write(str(tot_edges) + ' ' + str(tot_nodes)+' 10')
    else:
        f.write(str(tot_edges) + ' ' + str(tot_nodes))
    f.write('\n')
    for edge in edges:
        f.write(str(edge[0]+1) + ' ' + str(edge[1]+1))
        f.write('\n')
    if node_weights:
        for i in node_weights:
            f.write(str(i))
            f.write('\n')
    f.close()

def run(file,numcuts):
    (subprocess.check_call(f"./shmetis {file} {numcuts} 1",shell=True))

def run_gomory(file,numcuts,folder):
    s=f"{file} {numcuts} {folder}cut_gomory.txt "
    subprocess.call(f'python3 ../src/Gomory-cuts.py {s}',shell=True)

def run_karger(file,numcuts,folder):
    s=f"{file} {numcuts} {folder}cut_karger.txt "
    subprocess.call(f'python3 ../src/karger_stein.py {s}',shell=True)

def run_spectral_clustering(file,num_nodes,numcuts,folder):
    s=f"{folder}cut_spectral.txt "
    subprocess.call(f'python3 spectral_clustering.py {file} {num_nodes} {numcuts} {s}',shell=True)

def change(folder,num_cuts):
    subprocess.check_call(f"mv {folder}graph_converted_weighted.txt.part.{num_cuts} {folder}cut_hMetis_weights.txt ",shell=True)

def change_no_weight(folder,num_cuts):
    subprocess.check_call(f"mv {folder}graph_converted.txt.part.{num_cuts} {folder}cut_hMetis.txt ",shell=True)

def run_visualise(folder):
    print(folder)
    subprocess.check_call(f"python3 visualise_cuts.py {folder}",shell=True)

def remove(file):
    subprocess.check_call(f"rm {folder}num_cuts.txt ",shell=True)
    

def read_num_nodes(folder):
    with open(f'{folder}graph_stats.txt') as f:
        data = f.read()
    js = json.loads(data)
    num_nodes=js['num_nodes']
    num_cuts=js['num_cuts']
    return num_nodes,num_cuts
    # js['num_cuts']=5
    # with open(f'{folder}graph_stats.txt', 'w') as f:
    #     json.dump(js, f)
    # return num_nodes,5
def read_num_cuts(folder):
    f=open(f"{folder}/num_cuts.txt")
    content=f.read()
    num=re.findall(r'\d+',content)
    return int(num[0])

def get_num_nodes(folder):
    tgraph=nx.read_edgelist(f"{folder}graph.txt")
    return tgraph.number_of_nodes()

def create_graph_stats(folder,num_nodes,num_cuts):
    js={}
    js["num_nodes"]=num_nodes
    js["num_cuts"]=num_cuts 
    with open(f'{folder}graph_stats.txt', 'w') as f:
        json.dump(js, f)

if __name__ == '__main__':
    #check number of arguments
    
    
    if (len(sys.argv)!=2 ):
        print("Usage python convert.py <folder>")
        sys.exit(0)
    next_folder=sys.argv[1]
    num_nodes,num_cuts=read_num_nodes(next_folder)
    convert(next_folder+"/graph.txt",num_nodes,next_folder+"/graph_converted.txt")
    run(next_folder+'/graph_converted.txt',num_cuts)
    change_no_weight(next_folder,num_cuts)
    run_spectral_clustering(next_folder+'/graph.txt',num_nodes,num_cuts,next_folder+'/')




    # read_brightkite(file_name,outfile)
    # folder=f"./{input_folder}/{j}/{i}/"
    #         print(folder)
    #         next_folder=folder
    #         # num_nodes=get_num_nodes(next_folder)
    #         # num_cuts=read_num_cuts(next_folder)
    #         num_nodes,num_cuts=read_num_nodes(next_folder)
    #         # create_graph_stats(folder,num_nodes,num_cuts)
    #         # remove(next_folder)
    #         # convert(next_folder+"graph.txt",num_nodes,next_folder+"graph_converted.txt")
    #         # run(next_folder+'graph_converted.txt',num_cuts)
    #         # change(next_folder,num_cuts)
    #         run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)
    #         run_visualise(folder)
    # folder="./facebook_lc_5/val_set"
    # for i in range(1,2):
    #     next_folder=folder+f"/{i}/" 
    # # next_folder="./SBM_500_mixed3/test_set/2/"
    #     num_nodes,num_cuts=read_num_nodes(next_folder)
    #     print(num_cuts)
    #     convert(next_folder+"graph.txt",num_nodes,next_folder+"graph_converted.txt")
    #     st=time.time()
    #     run(next_folder+'graph_converted.txt',num_cuts)
    #     ed=time.time()
    #     print(ed-st)
    #     change(next_folder,num_cuts)
    #     run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)
    #     run_visualise(next_folder)

    # exit(0)       

    # for input_folder in ['citeseer_lc','dblp_lc','facebook_lc']:
    # for input_folder in ['test_data']:
    #     for j in ['train_set','val_set','test_set']:
    #         l=len(os.listdir(f"{input_folder}/{j}/"))
    #         for i in range(1,2):
    #             folder=f"./{input_folder}/{j}/{i}/"
    #             print(folder)
    #             next_folder=folder
    #             # num_nodes=get_num_nodes(next_folder)
    #             # num_cuts=read_num_cuts(next_folder)
    #             num_nodes,num_cuts=read_num_nodes(next_folder)
    #             # create_graph_stats(folder,num_nodes,num_cuts)
    #             # remove(next_folder)
    #             # convert(next_folder+"graph.txt",num_nodes,next_folder+"graph_converted.txt")
    #             # run(next_folder+'graph_converted.txt',num_cuts)
    #             # change(next_folder,num_cuts)
    #             run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)
    #             run_visualise(folder)
    #         #     break
    #     # break
    # # run_visualise(next_folder)
    
    # folder="./SBM_500_diffcuts/val_set/1/"
    # for f in ['facebook','facebook_lc','cora','cora_lc','citeseer','citeseer_lc','dblp_lc','german','lastfm']:
    #     folder=f"./{f}/train_set/1/"
    #     next_folder=folder
    #     num_nodes,num_cuts=read_num_nodes(next_folder)
    #     run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)
    #     run_visualise(next_folder)
    # # run(next_folder+'graph_converted.txt',num_cuts)
    # change(next_folder,num_cuts)
    # run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)



    # folder='./processed_data'
    # for i in tqdm(os.listdir(folder)):
    #     # next_folder=folder+"/"+i+"/"
    #     next_folder=folder+"/lastfm_lc/"
    #     num_nodes,num_cuts=read_num_nodes(next_folder)
    #     print(next_folder,num_nodes,num_cuts)
    #     convert(next_folder+"graph.txt",num_nodes,next_folder+"graph_converted.txt")

    #     try :
    #         run(next_folder+'graph_converted.txt',num_cuts)
    #         change(next_folder,num_cuts)
    #         # run_spectral_clustering(next_folder+'graph.txt',num_nodes,num_cuts,next_folder)
    #         run_visualise(next_folder)
    #     #     run_gomory(next_folder+'graph.txt',num_cuts,next_folder)
    #     #     run_karger(next_folder+'graph.txt',num_cuts,next_folder) 
    #     #     run_visualise(next_folder)
    #     except:
    #         print("errorr ##################33",i)
    #     exit(0)