"""Run simple consensus algorithm.

bflrun -np 8 python example/consensus.py
"""
import time
import numpy as np
import torch

use_bfl = True
if use_bfl:
    import bluefoglite as bfl
else:
    import bluefog.torch as bfl

bfl.init()
bfl.set_topology(topology=bfl.ExponentialGraph(bfl.size()))

dims = [100, 1000, 10000, 50_000, 80_000, 100_000, 200_000]
durations = []
for dim in dims:
    # Let each process create a vector containing a element equals to its rank.
    if use_bfl:
        x = np.array([bfl.rank()] * dim, dtype=np.float64)
        x_avg = np.array([(bfl.size() - 1) / 2] * dim)
    else:
        x = torch.Tensor([bfl.rank()] * dim).double()
        x_avg = torch.Tensor([(bfl.size() - 1) / 2] * dim).double()

    mse = [((x - x_avg) * (x - x_avg)).sum()]
    # print(f"{bfl.rank()}: before {x}")
    start = time.perf_counter()
    for _ in range(100):
        x = bfl.neighbor_allreduce(x)
        # mse.append(np.linalg.norm(x - x_avg))
    # print(f"{bfl.rank()}: after {x}")
    duration = time.perf_counter() - start
    if bfl.rank() == 0:
        print(f"dim: {dim} -- duration: {duration}")
    durations.append(duration)
    np.testing.assert_allclose(x, x_avg)
bfl.shutdown()
