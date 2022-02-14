"""Run simple consensus algorithm.

bflrun -np 8 python example/consensus.py
"""

import numpy as np
import bluefoglite as bfl

bfl.init()
bfl.set_topology(topology=bfl.ExponentialGraph(bfl.size()))

# Let each process create a vector containing a element equals to its rank.
x = np.array([bfl.rank()], dtype=np.float64)
x_avg = np.array([(bfl.size() - 1) / 2])
mse = [np.linalg.norm(x - x_avg)]

print(f"{bfl.rank()}: before {x}")
for _ in range(25):
    x = bfl.neighbor_allreduce(x)
    mse.append(np.linalg.norm(x - x_avg))
print(f"{bfl.rank()}: after {x}")

bfl.shutdown()