"""
Date: May 2022

This file implements the DSGD-CECA topologies from:
DSGD-CECA: Decentralized SGD with Communication-Optimal Exact Consensus Algorithm

"""
import numpy as np
import math
import networkx as nx
from itertools import chain
from typing import List, Tuple, Dict, Iterator, Optional


def GetOptTopoSendRecvRanks(
    topo: nx.DiGraph, self_rank: int
) -> Iterator[Tuple[List[int], List[int]]]:
    """A utility function to generate 1-outoging send rank and corresponding recieving rank(s).
        The generating method is Opt Topo.

    Args:
        topo (nx.DiGraph): The base topology to generate dynamic send and receive ranks.
        self_rank (int): The self rank.
    Yields:
        Iterator[Tuple[List[int], List[int]]]: send_ranks, recv_ranks.
    """
    n = topo.number_of_nodes()
    tau = int(math.ceil(math.log(n, 2.0)))
    bi = bin(n - 1)
    ite = 0
    n_ex = 0
    while True:
        r = ite % tau
        ite = ite + 1
        dig = int(bi[2 + r])
        if dig == 1:
            send_rank = (self_rank + n_ex + 1) % n
            recv_rank = (self_rank - n_ex - 1) % n
        else:
            send_rank = (self_rank + n_ex) % n
            recv_rank = (self_rank - n_ex) % n
        yield [send_rank], [recv_rank], n_ex, dig
        
        n_ex = 2 * n_ex + dig
        if n_ex >= (n - 1):
            n_ex = 0


def GetOptTopoSendRecvRanks1Port(
    topo: nx.DiGraph, self_rank: int
) -> Iterator[Tuple[List[int], List[int]]]:
    n = topo.number_of_nodes()
    assert n % 2 == 0
    
    tau = int(math.ceil(math.log(n, 2.0)))
    bi = bin(n - 1)
    ite = 0
    n_ex = 0
    while True:
        r = ite % tau
        ite = ite + 1
        dig = int(bi[2 + r])
        if self_rank % 2 == 1:
            send_rank = (self_rank + 2 * n_ex + 1) % n
            recv_rank = (self_rank + 2 * n_ex + 1) % n
        else:
            send_rank = (self_rank - 2 * n_ex - 1) % n
            recv_rank = (self_rank - 2 * n_ex - 1) % n
            
            
        yield [send_rank], [recv_rank], n_ex, dig
        
        n_ex = 2 * n_ex + dig
        if n_ex >= (n - 1):
            n_ex = 0
