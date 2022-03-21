import sys
import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from tqdm import tqdm

# For graph related operations
import networkx as nx

# P : original image
# L : labeling of the original image
# V(.,.) : interaction potential
# set_N : set of all pairs of neighboring pixels


def get_neighbors(i, j, n, m):
    """ 
    Returns the neighbors of the pixel (i, j) according to its location in the image
    
    :param (i, j): coordinates of the pixel
    :param (n, m): size of the matrix
    :rtype: list
    """
    
    neighbors = []

    if i!=0:
        neighbors.append(str(i-1) + "," + str(j))
    if i!=n-1:
        neighbors.append(str(i+1) + "," + str(j))
    if j!=0:
        neighbors.append(str(i) + "," + str(j-1))
    if j!=m-1:
        neighbors.append(str(i) + "," + str(j+1))

    return neighbors


def get_neighbors_undirected(i, j, n, m):
    """
    Returns only the needed neigbors to construct an undirected graph

    :param (i, j): coordinates of the pixel
    :param (n, m): size of the matrix
    :rtype: list  
    """
    
    neighbors = []

    if i!=n-1:
        neighbors.append(str(i+1) + "," + str(j))
    if j!=m-1:
        neighbors.append(str(i) + "," + str(j+1))
    
    return neighbors


def gaussian(img: np.array, mu: float, sigma: float) -> np.array:
    """
    Computes the Gaussian PDF at all points in the image

    :param img: studied image
    :param mu: mean of the Gaussian PDF
    :param sigma: standard deviation of the Gaussian PDF
    :rtype: np.array
    """

    num = np.exp(-0.5 * (((img - mu) / sigma)**2))
    den = np.sqrt(2 * np.pi)*sigma

    return num / den


def regional_term(img: np.array, mean: np.array, std: np.array) -> np.array:
    """
    Returns the log-likelihood ratio of all pixels (This is independant of X, but depends on mu and sigma)
    Corresponds to the regional term of the energy and is motivated by the MAP-MRF formulation

    :param img: studied image
    :param mu: means of the image pixels contained in the object and the background
    :param sigma: standard deviations of the image pixels contained in the object and the background
    :rtype: np.array
    """

    gauss_obj = gaussian(img, mean[1], std[1])
    gauss_bkg = gaussian(img, mean[0], std[0])

    return np.log(gauss_obj), np.log(gauss_bkg)


def dist(x, y):
    """
    Compute the L1-norm between two pixels of the image

    :param x: coordinates of the first pixel
    :param y: coordinates of the second pixel
    :rtype:   
    """

    return np.abs(x[0]-y[0]) + np.abs(x[1]-y[1])


def boundary_term(img: np.array, p: tuple, q: tuple, Sigma: np.array) -> float:
    """
    Compute the interaction potential between two image pixels

    :param img: studied image
    :param (p, q): the two nodes that are linked by the edge we want to compute the weight
    :param Sigma: hyperparameter of the problem
    :rtype: float
    """

    return np.exp(-(img[p[0],p[1]]-img[q[0],q[1]])**2 / (2*Sigma**2)) / dist(p, q)


def create_graph(img: np.array, Sigma: np.array) -> nx.Graph:
    """
    Create the directed and weighted graph associated to the given image

    :param img: studied image
    :param mu: means of the image pixels contained in the object and the background
    :param sigma: standard deviations of the image pixels contained in the object and the background
    :rtype: nx.Graph
    """

    n, m = img.shape

    graph = nx.Graph()

    nodes = [str(i)+str(j) for i in range(n) for j in range(m)]
    graph.add_nodes_from(nodes)

    for i in tqdm(range(n)):
        for j in range(m):
            
            neighbors = get_neighbors_undirected(i, j, n, m)
            
            for node in neighbors:
                v = node.split(",")
                w = boundary_term(img, (i,j), (int(v[0]),int(v[1])), Sigma)
                graph.add_edge(str(i)+","+str(j), node, weight=w)

    return graph


def add_S_T(graph: nx.Graph, img: np.array, hard_cstr:np.array, mean: np.array, std: np.array, lambda_: float =1.) -> nx.Graph:
    """
    Add the sink and source points to the graph with corresponding weights

    :param graph : pre-constructed graph with all the nodes of the pixels linked
    :pram
    :rtype:
    """

    #Compute the edge value of the edges between sink or source and pixels corresponding to hard constraints
    K = 0
    for node in graph.nodes:
        K = np.maximum(K, np.sum([graph.get_edge_data(node, neighbor)["weight"] for neighbor in graph.neighbors(node)]))
    K += 1


    #Add nodes source and sink
    graph.add_nodes_from(["S", "T"])


    #Add edges between sink and pixels of background hard constraint
    bkg_cstr = np.vstack(np.where(hard_cstr[:,:,0]==219)) #red lines
    n = bkg_cstr.shape[1]
    bkg_cstr = bkg_cstr.reshape((n,2))

    for x in tqdm(bkg_cstr):
        graph.add_edge(str(x[0])+","+str(x[1]), "T", weight=K)


    #Add edges between source and pixels of object hard constraint 
    obj_cstr = np.vstack(np.where(hard_cstr[:,:,0]==255)) #white lines
    n = obj_cstr.shape[1]
    obj_cstr = obj_cstr.reshape((n,2))

    for x in tqdm(obj_cstr):
        graph.add_edge("S", str(x[0])+","+str(x[1]), weight=K)


    #Add edges between the source and sink nodes and the pixels that have no hard constraint
    no_cstr = np.vstack(np.where(hard_cstr[:,:,0]==0))
    n = no_cstr.shape[1]
    no_cstr = no_cstr.reshape((n,2))

    RT, RS = regional_term(img, mean, std)

    for x in tqdm(no_cstr):
        graph.add_edge(str(x[0])+","+str(x[1]), "T", weight=lambda_*RT)
        graph.add_edge("S", str(x[0])+","+str(x[1]), weight=lambda_*RS)


    return graph


def main(name: str, Sigma: float, lambda_: float =1.):
    """
    Run a set of functions to create the graph associated to the given image for minimum cut

    :param name: name of the image
    :param lambda_, Sigma : hyperparameters
    """

    img = imread("./data/images/" + name + ".jpg")
    segmented = imread("./data/images-gt/" + name + ".png")
    hard_cstr = imread("./data/images-labels/" + name + "-anno.png")

    gray = rgb2gray(img)
    mean = np.zeros(2)
    std = np.zeros(2)

    bkg = (segmented == 0)
    img_bkg = gray[bkg]
    mean[0], std[0] = img_bkg.mean(), img_bkg.std()

    obj = (segmented == 255)
    contour = (segmented == 128)
    obj_cont = obj + contour
    img_obj = gray[obj_cont]
    mean[1], std[1] = img_obj.mean(), img_obj.std()

    graph = create_graph(gray, Sigma)
    graph = add_S_T(graph, gray, hard_cstr, mean, std, lambda_)

    return graph



#############################################################################



if __name__=='main':

    print('How to use the .py file from a terminal : in the folder of the file write "python ToGraph.py name lambda sigma",where name is the name of the image file, lambda and sigma are both hyperparameters')

    name = str(sys.argv[1])
    lambda_ = float(sys.argv[2])
    Sigma = float(sys.argv[3])

    img = imread("./data/images/" + name + ".jpg")
    segmented = imread("./data/images-gt/" + name + ".png")
    hard_cstr = imread("./data/images-labels/" + name + "-anno.png")

    gray = rgb2gray(img)
    mean = np.zeros(2)
    std = np.zeros(2)

    bkg = (segmented == 0)
    img_bkg = gray[bkg]
    mean[0], std[0] = img_bkg.mean(), img_bkg.std()

    obj = (segmented == 255)
    contour = (segmented == 128)
    obj_cont = obj + contour
    img_obj = gray[obj_cont]
    mean[1], std[1] = img_obj.mean(), img_obj.std()

    graph = create_graph(gray, Sigma)
    graph = add_S_T(graph, gray, hard_cstr, mean, std, lambda_)