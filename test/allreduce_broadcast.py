import bluefoglite as bfl
import numpy as np  # type: ignore

bfl.init()
# print(f"{bfl.rank()}: init done")
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

array = np.array([1.0, 2.0, 3, 4]) + bfl.rank()
bfl.allreduce(array, agg_op="AVG")
expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
np.testing.assert_allclose(array, expected_array)

if bfl.rank() == 0:
    array += 1
bfl.broadcast(array=array, root_rank=0)
expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() + 1) / 2
np.testing.assert_allclose(array, expected_array)

# Why do we need to call shutdown explicitly?
bfl.shutdown()