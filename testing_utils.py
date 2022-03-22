import numpy as np


def dice_score(seg_1, seg_2):
    n1 = len(seg_1.flatten())
    n2 = len(seg_2.flatten())
    assert n1 == n2
    unique, count = np.unique(seg_1[seg_1==seg_2], return_counts=True)
    intersection_nb = count[list(unique).index(1)]
    return 2*intersection_nb/(n1+n2)

def IoU(seg_1, seg_2):
    n1 = len(seg_1.flatten())
    n2 = len(seg_2.flatten())
    assert n1 == n2
    unique_inter, count_inter = np.unique(seg_1[seg_1==seg_2], return_counts=True)
    unique_1, count_1 = np.unique(seg_1, return_counts=True)
    unique_2, count_2 = np.unique(seg_2, return_counts=True)
    intersection_nb = count_inter[list(unique_inter).index(1)]
    union_nb = count_1[list(unique_1).index(1)] + count_2[list(unique_2).index(1)] - intersection_nb
    return intersection_nb/union_nb
