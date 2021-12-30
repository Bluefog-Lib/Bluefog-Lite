import os
import time

import bluefoglite as bfl
import numpy as np  # type: ignore

bfl.init()
# print(f"{bfl.rank()}: init done")
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

if bfl.rank() == 0:
    data = np.array([1, 2, 3, 4])
    bfl.send(dst=1, obj_or_array=data)
elif bfl.rank() == 1:
    buf = np.array([9, 8, 7, 6])
    recv_data = bfl.recv(src=0, obj_or_array=buf)
    print("recv data:", recv_data)
else:
    pass
time.sleep(1.5)
bfl.shutdown()
