import math
from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx


def GetRecvWeights(topo: nx.DiGraph, rank: int) -> Tuple[float, Dict[int, float]]:
    """Return a Tuple of self_weight and neighbor_weights for receiving dictionary."""
    weight_matrix = nx.to_numpy_array(topo)
    self_weight = 0.0
    neighbor_weights = {}
    for src_rank in topo.predecessors(rank):
        if src_rank == rank:
            self_weight = weight_matrix[src_rank, rank]
        else:
            neighbor_weights[src_rank] = weight_matrix[src_rank, rank]
    return self_weight, neighbor_weights


def GetSendWeights(topo: nx.DiGraph, rank: int) -> Tuple[float, Dict[int, float]]:
    """Return a Tuple of self_weight and neighbor_weights for sending dictionary."""
    weight_matrix = nx.to_numpy_array(topo)
    self_weight = 0.0
    neighbor_weights = {}
    for recv_rank in topo.successors(rank):
        if recv_rank == rank:
            self_weight = weight_matrix[rank, recv_rank]
        else:
            neighbor_weights[recv_rank] = weight_matrix[rank, recv_rank]
    return self_weight, neighbor_weights


def isPowerOf(x, base):
    assert isinstance(base, int), "Base has to be a integer."
    assert base > 1, "Base has to a interger larger than 1."
    assert x > 0
    if (base ** int(math.log(x, base))) == x:
        return True
    return False


def ExponentialTwoGraph(size: int) -> nx.DiGraph:
    """Generate graph topology such that each points only connected to a
    point such that the index difference is the power of 2."""
    assert size > 0
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def ExponentialGraph(size: int, base: int = 2) -> nx.DiGraph:
    """Generate graph topology such that each points only connected to a
    point such that the index difference is power of base."""
    x = [1.0]
    for i in range(1, size):
        if isPowerOf(i, base):
            x.append(1.0)
        else:
            x.append(0.0)
    x_a = np.array(x)
    x_a /= x_a.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x_a, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def MeshGrid2DGraph(size: int, shape: Optional[Tuple[int, int]] = None) -> nx.DiGraph:
    """Generate 2D MeshGrid structure of graph.

    Assume shape = (nrow, ncol), when shape is provided, a meshgrid of nrow*ncol will be generated.
    when shape is not provided, nrow and ncol will be the two closest factors of size.

    For example: size = 24, nrow and ncol will be 4 and 6, respectively.
    We assume  nrow will be equal to or smaller than ncol.
    If size is a prime number, nrow will be 1, and ncol will be size, which degrades the topology
    into a linear one.
    """

    assert size > 0
    if shape is None:
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
    nrow, ncol = shape
    assert size == nrow * ncol, "The shape doesn't match the size provided."
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i][i] = 1.0
        if (i + 1) % ncol != 0:
            topo[i][i + 1] = 1.0
            topo[i + 1][i] = 1.0
        if i + ncol < size:
            topo[i][i + ncol] = 1.0
            topo[i + ncol][i] = 1.0

    # According to Hasting rule (Policy 1) in https://arxiv.org/pdf/1702.05122.pdf
    # The neighbor definition in the paper is different from our implementation,
    # which includes the self node.
    topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
    for i in range(size):
        for j in topo_neighbor_with_self[i]:
            if i != j:
                topo[i][j] = 1.0 / max(
                    len(topo_neighbor_with_self[i]), len(topo_neighbor_with_self[j])
                )
        topo[i][i] = 2.0 - topo[i].sum()
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def StarGraph(size: int, center_rank: int = 0) -> nx.DiGraph:
    """Generate star structure of graph.

    All other ranks are connected to the center_rank. The connection is
    bidirection, i.e. if the weight from node i to node j is non-zero, so
    is the weight from node j to node i.
    """
    assert size > 0
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i, i] = 1 - 1 / size
        topo[center_rank, i] = 1 / size
        topo[i, center_rank] = 1 / size
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G


def RingGraph(size: int, connect_style: int = 0) -> nx.DiGraph:
    """Generate ring structure of graph (uniliteral).

    Argument connect_style should be an integer between 0 and 2, where
    0 represents the bi-connection, 1 represents the left-connection,
    and 2 represents the right-connection.
    """
    assert size > 0
    assert 0 <= connect_style <= 2, (
        "connect_style has to be int between 0 and 2, where 1 "
        "for bi-connection, 1 for left connection, 2 for right connection."
    )
    if size == 1:
        return nx.from_numpy_array(np.array([[1.0]]), create_using=nx.DiGraph)
    if size == 2:
        return nx.from_numpy_array(
            np.array([[0.5, 0.5], [0.5, 0.5]]), create_using=nx.DiGraph
        )

    x = np.zeros(size)
    x[0] = 0.5
    if connect_style == 0:  # bi-connection
        x[0] = 1 / 3.0
        x[-1] = 1 / 3.0
        x[1] = 1 / 3.0
    elif connect_style == 1:  # left-connection
        x[-1] = 0.5
    elif connect_style == 2:  # right-connection
        x[1] = 0.5
    else:
        raise ValueError("Connect_style has to be int between 0 and 2")

    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    G = nx.from_numpy_array(topo, create_using=nx.DiGraph)
    return G
