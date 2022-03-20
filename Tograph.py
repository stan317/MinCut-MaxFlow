import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# For graph related operations
import networkx as nx
from skimage.color import rgb2gray

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
    if i!=n:
        neighbors.append(str(i+1) + "," + str(j))
    if j!=0:
        neighbors.append(str(i) + "," + str(j-1))
    if j!=m:
        neighbors.append(str(i) + "," + str(j+1))

    return neighbors


def get_neighbors_undirect(i, j, n, m):
    """
    Returns only the needed neigbors to construct an undirected graph

    :param (i, j): coordinates of the pixel
    :param (n, m): size of the matrix
    :rtype: list  
    """
    
    neighbors = []

    if i!=n:
        neighbors.append(str(i+1) + "," + str(j))
    if j!=m:
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


def regional_term(img: np.array, mu: np.array, sigma: np.array) -> np.array:
    """
    Returns the log-likelihood ratio of all pixels (This is independant of X, but depends on mu and sigma)
    Corresponds to the regional term of the energy and is motivated by the MAP-MRF formulation

    :param img: studied image
    :param mu: means of the image pixels contained in the object and the background
    :param sigma: standard deviations of the image pixels contained in the object and the background
    :rtype: np.array
    """

    num = gaussian(img, mu[1], sigma[1])
    den = gaussian(img, mu[0], sigma[0])

    return np.log(num / den)


def dist(x, y):
    """
    Compute the L1-norm between two pixels of the image

    :param x:
    :param y:
    :rtype:   
    """

    return np.abs(x[0]-y[0]) + np.abs(x[1]-y[1])


def boundary_term(img: np.array, p: list, q: list, sigma: np.array) -> float:
    """
    Compute the interaction potential between two image pixels

    :param img:
    :param p:
    :param q:
    :param sigma:
    :rtype: float
    """

    return np.exp(-(img[p]-img[q])**2 / (2*sigma**2)) / dist(p, q)


def capacity(img: np.array, p: list, q:list, mu: np.array, sigma: np.array, lambda_:float =1.) -> float:
    """
    Compute the capacity of an edge

    :param img:
    :param p:
    :param q:
    :param mu: means of the image pixels contained in the object and the background
    :param sigma: standard deviations of the image pixels contained in the object and the background
    :param lambda_:
    :rtype: float
    """

    Rpq = regional_term(img, mu, sigma)
    Bpq = boundary_term(img, p, q, sigma)

    return Bpq + lambda_*Rpq


def add_S_T(graph: nx.Graph):
    """
    Add the sink and source points to the graph with corresponding weights

    :param
    :rtype:
    """

    graph.add_nodes_from(["S", "T"])

    K = 0
    for x in :
        K = np.maximum(K, np.sum(graph.get_edge_data()))
    
    graph.add_edge()


def create_graph(img: np.array, mu: np.array, sigma: np.array) -> nx.Graph:
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

    for i in range(n):
        for j in range(m):
            
            neighbors = get_neighbors_undirect(i, j, n, m)
            
            for node in neighbors:

                w = capacity(img, [i,j], [node[0],node[2]], mu, sigma)
                graph.add_edge(str(i)+","+str(j), node, weight=w)
    
    return graph


if __name__=='main':

    name = str(sys.argv[1])
    lambda_ = float(sys.argv[2])

    img = cv2.open("./data/images/" + name + ".jpg")
    segmented = cv2.open("./data/images-gt/" + name + ".png")
    hard_cstr = cv2.open("./data/images-labels/" + name + "-anno.png")

    mu = np.zeros((1,2))
    sigma = np.zeros((1,2))

    B = (segmented == 0.)
    y0 = img[B]
    mu[0], sigma[0] = y0.mean(), y0.std()

    O = (segmented == 1.)
    y1 = img[O]
    mu[1], sigma[1] = y1.mean(), y1.std()

    graph = create_graph(img)