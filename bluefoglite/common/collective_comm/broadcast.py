import math
from typing import List

from bluefoglite.common.tcp.agent import AgentContext
from bluefoglite.common.tcp.buffer import SpecifiedBuffer


def broadcast_one_to_all(
    buf: SpecifiedBuffer, root_rank: int, context: AgentContext, *, tag=0
):
    # assume the input are all well-defined and behaved.
    if context.rank != root_rank:
        buf.recv(root_rank)
        return

    handles: List[int] = []
    for i in range(context.size):
        if i == context.rank:
            continue
        handles.append(buf.isend(i))

    for h in handles:
        buf.waitCompletion(h)


def broadcast_ring(
    buf: SpecifiedBuffer, root_rank: int, context: AgentContext, *, tag=0
):
    virtual_rank = (context.rank - root_rank) % context.size
    next_rank = (context.rank + 1) % context.size
    prev_rank = (context.rank - 1) % context.size
    for r in range(context.size - 1):
        if virtual_rank == r:
            buf.send(next_rank)
        elif virtual_rank == r + 1:
            buf.recv(prev_rank)
        else:
            pass


def broadcast_spreading(
    buf: SpecifiedBuffer, root_rank: int, context: AgentContext, *, tag=0
):
    # Using the 0->1 | 0->2, 1->3 | 0->4, 1->5, 2->6, 3->9 style.
    virtual_rank = (context.rank - root_rank) % context.size
    rounds = math.ceil(math.log2(context.size))

    for r in range(rounds):
        rank_diff = 1 << r

        if virtual_rank < rank_diff:
            virutal_send_to = virtual_rank + rank_diff
            if virutal_send_to > context.size:
                continue
            real_send_to = (virutal_send_to + root_rank) % context.size
            buf.send(real_send_to)
        elif rank_diff <= virtual_rank < 2 * rank_diff:
            virutal_recv_from = virtual_rank - rank_diff
            if virutal_recv_from < 0:  # impossible.
                continue
            real_recv_from = (virutal_recv_from + root_rank) % context.size
            buf.recv(real_recv_from)
        else:
            pass
