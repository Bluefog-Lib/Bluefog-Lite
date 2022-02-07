import bluefoglite as bfl
import numpy as np  # type: ignore

bfl.init()
# print(f"{bfl.rank()}: init done")
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

array = np.array([1.0, 2, 3, 4]) + bfl.rank()
ret_array = bfl.allreduce(array, agg_op="AVG")
expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
np.testing.assert_allclose(ret_array, expected_array)

if bfl.rank() == 0:
    ret_array += 1
bfl.broadcast(array=ret_array, root_rank=0)
expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() + 1) / 2
np.testing.assert_allclose(ret_array, expected_array)
# Why do we need to call shutdown explicitly?
# We need to detect event_loop thread is_alive or not.
bfl.shutdown()
