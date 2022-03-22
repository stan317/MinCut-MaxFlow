import numpy as np
import networkx as nx
from math import *

def grow(Graph, Active):
    while len(Active)>0:
        node_name = Active.pop()
        node = Graph.nodes[node_name]
        terminal = node['terminal']
        if terminal != None:
            for neighbor_name in Graph[node_name]:
                neighbor = Graph.nodes[neighbor_name]
                if neighbor['terminal'] == None:
                    Active.insert(0,neighbor_name)
                    neighbor['terminal'] = terminal
                    neighbor['parent'] = node_name
                    Graph[node_name][neighbor_name]['color'] = (terminal=='S' and 'r') or 'b'
                if neighbor['terminal'] != terminal:
                    Active.append(node_name)
                    return(node_name,neighbor_name)
    return None

def get_path_flow(Graph, initial_node):
    max_flow = inf
    node_name = initial_node
    path = [node_name]
    node = Graph.nodes[node_name]
    while not node_name in ['S','T']:
        parent_name = node['parent']
        parent = Graph.nodes[parent_name]
        max_flow = min(Graph[node_name][parent_name]['capacity'],max_flow)
        path += [parent_name]
        node_name=parent_name
        node = parent
    return path,max_flow

def augment(Graph,path,max_flow):
    Orphans = []
    for i in range(len(path)-1):
        node = path[i]
        next_node = path[i+1]
        Graph[node][next_node]['capacity'] -= max_flow
        if Graph[node][next_node]['capacity'] == 0:
            Graph.remove_edge(node,next_node)
            terminal = Graph.nodes[node]['terminal']
            if terminal == Graph.nodes[next_node]['terminal']:
                if terminal == 'S':
                    Graph.nodes[next_node]['parent'] = None
                    Orphans += [next_node]
                else:
                    Graph.nodes[node]['parent'] = None
                    Orphans += [node]

    return Orphans

def root(Graph,node):
    current = node
    parent = Graph.nodes[node]['parent']
    while not parent in ['S','T',None]:
        current = parent
        parent =  Graph.nodes[current]['parent']
    return current, parent

def adopt(Graph,Orphans,Active):
    while len(Orphans) != 0:
        node_name = Orphans.pop()
        node = Graph.nodes[node_name]
        children = []
        for neighbor_name in Graph[node_name]:
            neighbor = Graph.nodes[neighbor_name]
            assert node['terminal']
            if node['terminal'] == neighbor['terminal']:
                if neighbor['parent'] == node_name:
                    children.append(neighbor_name)
                elif root(Graph, neighbor_name)[0] != node_name:
                    node['parent'] = neighbor_name
                    Graph[node_name][neighbor_name]['color'] = (node['terminal']=='S' and 'r') or 'b'
                    break
                Active.insert(0,neighbor_name)
        Active = list(filter(lambda n: n!=node_name, Active))
        if (node['parent'] == None):
            for child in children:
                Graph.nodes[child]['parent'] = None
                Graph[node_name][child]['color'] = 'black'
            Orphans += children
            node['terminal'] = None
    return

def min_cut_max_flow(Graph, S, T):
    Graph.nodes[S]['terminal'] = 'S'
    Graph.nodes[T]['terminal'] = 'T'
    active = [S,T]
    touching_nodes = grow(Graph,active)
    epoch = 0
    while touching_nodes:
        n1, n2 = touching_nodes
        path1, maxflow1 = get_path_flow(Graph,n1)
        path2, maxflow2 = get_path_flow(Graph,n2)
        if Graph.nodes[n1]['terminal'] == 'S':
            path = path1[::-1]+path2
        else:
            path = path2[::-1]+path1
        orphans = augment(Graph,path,min(maxflow1, maxflow2, Graph[n1][n2]['capacity']))
        adopt(Graph,orphans,active)
        touching_nodes = grow(Graph,active)
        epoch+=1
        if epoch%10 == 0: print("iteration ",epoch)

    return(Graph)


def dinic_path_search(Graph, S):
    active = S
    while len(active)>0:
        node_name = active.pop()
        node = Graph.nodes[node_name]
        for neighbor_name in Graph[node_name]:
            neighbor = Graph.nodes[neighbor_name]
            if (neighbor['terminal'] == None) and (neighbor['parent'] == None):
                neighbor['parent'] = node_name
                active += [neighbor_name]
            elif neighbor['terminal'] == 'T':
                tree_path, flow = get_path_flow(Graph,node_name)
                min_flow = min(Graph[node_name][neighbor_name]['capacity'],flow)
                path = [neighbor_name]+tree_path
                dinic_augment(Graph, path, min_flow)
                return False
    return True

def dinic_augment(Graph, path, max_flow):
    for i in range(len(path)-1):
        Graph[path[i]][path[i+1]]['capacity'] -= max_flow
        if Graph[path[i]][path[i+1]] == 0:
            Graph.remove_edge(path[i],path[i+1])

def dinic(Graph,S,T):
    Graph.nodes[S]['terminal'] = 'S'
    Graph.nodes[T]['terminal'] = 'T'
    is_cut = False
    epoch = 0
    while not is_cut:
        epoch += 1
        nx.set_node_attributes(Graph, None, "parent")
        is_cut = dinic_path_search(Graph, [S])
        if epoch%10 == 0: print("iteration ",epoch)
    return Graph

def get_cut(Graph,width,height):
    segmentation = np.zeros((width,height))
    for x in range(width):
        for y in range(height):
            if Graph.nodes[str(x)+','+str(y)]['terminal'] == 'S':
                segmentation[x,y] = 1
    return segmentation


def get_nx_cut(nodes,width,height):
    s_nodes, t_nodes = nodes
    segmentation = np.zeros((width,height))
    for x in range(width):
        for y in range(height):
            if str(x)+','+str(y) in s_nodes:
                segmentation[x,y] = 1
    return segmentation