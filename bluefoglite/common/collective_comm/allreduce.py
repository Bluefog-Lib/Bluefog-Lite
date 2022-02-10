import math

from bluefoglite.common.tcp.agent import AgentContext
from bluefoglite.common.tcp.buffer import NumpyBuffer


def allreduce_tree(
    buf: NumpyBuffer, context: AgentContext, *, agg_op="AVG", tag=0
):  # pylint: disable=too-many-branches
    if agg_op not in ("SUM", "AVG"):
        raise NotImplementedError(
            "Only support SUM or AVG as the aggregation operation now."
        )
    rounds = math.ceil(math.log2(context.size))
    tmp_buf = None
    # We aggregate the value in the reverse tree style (Reduce/Scatter):
    # Round 0: (0<-1) (2<-3) (4<-5)
    # Round 1: (0<-2) (4)
    # Round 2: (0<-4)
    # Now 0 should have the aggregation of all values
    for r in range(rounds):
        rank_diff = 1 << r

        if context.rank % rank_diff != 0:
            continue

        if (context.rank // rank_diff) % 2 == 1:
            buf.send(context.rank - rank_diff)

        if (context.rank // rank_diff) % 2 == 0:
            if context.rank + rank_diff >= context.size:
                continue
            if tmp_buf is None:
                # TODO Delegate to the buf do the allocation
                tmp_buf = buf.clone()

            tmp_buf.recv(context.rank + rank_diff)
            buf.add_(tmp_buf)

    if context.rank == 0:
        print(f"Before {buf.array}")
        buf.div_(context.size)
        print(f"After {buf.array}")

    # Broadcast it in reverse away
    for r in range(rounds - 1, -1, -1):
        rank_diff = 1 << r

        if context.rank % rank_diff != 0:
            continue

        if (context.rank // rank_diff) % 2 == 0:
            if context.rank + rank_diff >= context.size:
                continue
            buf.send(context.rank + rank_diff)

        if (context.rank // rank_diff) % 2 == 1:
            buf.recv(context.rank - rank_diff)
