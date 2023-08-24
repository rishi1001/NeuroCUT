import sys
import networkx as nx
import random
import numpy as np
## Implementing Union-Find data structure
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]

    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        xroot = self.find(x)
        yroot = self.find(y)
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1


def karger_stein(G,k):
    n=G.number_of_nodes()
    m=G.number_of_edges()
    
    # print("n=",n,"m=",m)

    ## UNION-FIND
    uf = UnionFind(n)
    # print(uf.parent)

    vertices_remaining = n
    while vertices_remaining > k:
        # pick random edge
        e = random.choice(list(G.edges()))
        # print("e=",e)

        # contracting edge
        v1 = e[0]
        v2 = e[1]
        # print("v1=",v1,"v2=",v2)

        rootv1=uf.find(v1)
        rootv2=uf.find(v2)
        # print("rootv1=",rootv1,"rootv2=",rootv2)

        if rootv1 != rootv2:
            uf.union(rootv1,rootv2)
            # print(uf.parent)
            vertices_remaining -= 1
    


    ## Print all components
    ## Use Union-Find to find the components
    components = {}
    for i in range(n):
        root = uf.find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    ## get cut value
    cut_value = 0
    for e in G.edges():
        v1 = e[0]
        v2 = e[1]
        rootv1=uf.find(v1)
        rootv2=uf.find(v2)
        if rootv1 != rootv2:
            cut_value += G[v1][v2]['weight']
    # print("cut_value=",cut_value)

    return components,cut_value


if __name__ == "__main__":
    
    ## Check if 2 args is given
    if len(sys.argv) != 4:
        print("Usage: python3 karger_stein.py <graph> <k> <cuts_file>")
        sys.exit(1)

    g= sys.argv[1]
    k= int(sys.argv[2])
    cuts_file = sys.argv[3]
    # // G must be connected 
    G = nx.read_edgelist(g, nodetype=int)



    # G must be connected
    if not nx.is_connected(G):
        print("Graph is not connected")
        sys.exit(1)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1
    # print(G.edges(data=True))
    # print(G.nodes())

    number_iterations = 25
    cuts=None
    ## intialize cut_value to infinity
    cut_value=float('inf')
    for i in range(number_iterations):
        components,cut_value_tmp = karger_stein(G,k)
        if cut_value_tmp < cut_value:
            cut_value = cut_value_tmp
            cuts = components
    # print(cut_value)
    # print(cuts)

    # print("cuts=",cuts)
    ## Change key to 0-based
    cuts_array = [-1 for i in range(G.number_of_nodes())]
    ind=0
    for key,value in cuts.items():
        for v in value:
            cuts_array[v] = ind
        ind += 1
    
    # print("cut_value=",cut_value)
    # print("cuts=",cuts_array)
    np.savetxt(cuts_file,cuts_array,fmt="%d")
