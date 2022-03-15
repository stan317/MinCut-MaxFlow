import cv2
import numpy as np
from matplotlib import pyplot as plt

# For graph related operations
import networkx as nx
from networkx.algorithms.flow import minimum_cut
from sklearn import neighbors

# P : original image
# L : labeling of the original image
# V(.,.) : interaction potential
# set_N : set of all pairs of neighboring pixels

def getNeighbors(i, j, n, m):
    """ 
    Returns the neighbors of the pixel (i, j) according to ist location in the image
    
    :param (i, j): coordinates of the pixel
    :param (n, m): size of the matrix
    :rtype: list
    """
    
    neighbors = []

    if i!=0:
        neighbors.append(str(i-1) + str(j))
    if i!=n:
        neighbors.append(str(i+1) + str(j))
    if j!=0:
        neighbors.append(str(i) + str(j-1))
    if j!=n:
        neighbors.append(str(i) + str(j+1))

    return neighbors


def gaussian(img: np.array, mu: np.array, sigma: np.array) -> np.array:
    """
    Computes the Gaussian PDF at all points in Y

    :param img: studied image
    :param mu: mean of the Gaussian PDF
    :param sigma: standard deviation of the Gaussian PDF
    :rtype: np.array
    """

    num = np.exp(-0.5 * (((img - mu) / sigma)**2))
    den = np.sqrt(2 * np.pi)*sigma

    return num / den


def LLR(img: np.array, mu: np.array, sigma: np.array) -> np.array:
    """
    Returns the log-likelihood ratio of all pixels (This is independant of X, but depends on mu and sigma)
    Corresponds to the regional term of the energy and is motivated by the MAP-MRF formulation

    :param img: studied image
    :param mu: mean of the Gaussian PDF
    :param sigma: standard deviation of the Gaussian PDF
    :rtype: np.array
    """

    num = gaussian(img, mu[1], sigma[1])
    den = gaussian(img, mu[0], sigma[0])

    return np.log(num / den)


def boundary_term(img: np.array, sigma: np.array) -> float:
    """
    Compute the interaction potential between two labels of image pixels

    :param img:
    :param sigma:
    :rtype: float
    """

    pass


def create_graph(img: np.array) -> nx.Graph:
    """
    Create the directed and weighted graph associated to the given image

    :param img: studied image
    :rtype: nx.Graph
    """

    n, m = img.shape

    graph = nx.Graph()

    nodes = [str(i)+str(j) for i in range(n) for j in range(m)]
    graph.add_nodes_from(nodes)

    for i in range(n):
        for j in range(m):
            
            neighbors = getNeighbors(i, j, n, m)
            
            for node in neighbors:
                w = ...
                graph.add_edge(str(i)+str(j), node, weight=w)
    
    return graph