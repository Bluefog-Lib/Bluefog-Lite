import time

import bluefoglite as bfl
import numpy as np

bfl.init()
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
time.sleep(0.1)

if bfl.rank() == 0:
    data = {"x": [1, 2], "y": "s"}
    bfl.send(dst=1, obj_or_array=data)
elif bfl.rank() == 1:
    recv_data = bfl.recv(src=0)
    print("recv data:", recv_data)
else:
    pass
time.sleep(0.1)

bfl.shutdown()
