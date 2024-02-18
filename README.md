1. https://arxiv.org/abs/1906.01227
2. https://github.com/chaitjo/graph-convnet-tsp


## Commands to run
1. Activate Environment or create from [requirements.txt](./requirements.txt)
```
source {environment name}/bin/activate
```
2. Various parameters are listed in [makefile](/src/makefile), run commmand
```
make parameter_1={value_1} parameter_2={value_2} .. 

Reults will be formed in the result_phase2_rl folder
```

## Sample Dataset
A sample Cora graph is given in the data folder. The structure of input graph is as follows:
```
<Graph_name> //input to the model
    <test_set> 
        <1>...<n>
    <val_set> 
        <1>...<n>
    <train_set> 
        <1>...<n>
// Each folder in train/val/test set should have a graph.txt and graph_stats.txt. The number of required cuts can be modifited in the graph_stats file.
```
Also, to run on any new graph, you need to add the graph.txt and node_embedding.pt file in raw_data folder. 

## Running Baselines
Code in Baselines folder
### HMETIS/Spectral Clustering
1. Run a convert.py which takes a graph folder which contains `graph.txt` ans `graph_stats.txt`
```
python3 convert.py data/sample_graph/test_set/1/
```
2. Now,you can use visualise_cuts scripts to get values of various metrics using the cut formed

### GAP
1. In Baseline run make gap 

