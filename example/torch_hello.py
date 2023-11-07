import torch
import torch.distributed as dist
import bluefoglite.torch as bfl

bfl.init()
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

tensor = torch.zeros(1) + (bfl.size() - bfl.rank())

if bfl.rank() == 0:
    bfl.send(tensor=tensor, dst=1)
elif bfl.rank() == 1:
    bfl.recv(tensor=tensor, src=0)

print("Rank ", bfl.rank(), " has data ", tensor[0])

if bfl.rank() == 0:
    dist.send(tensor=tensor, dst=1)
elif bfl.rank() == 1:
    dist.recv(tensor=tensor, src=0)

print("Rank ", bfl.rank(), " has data ", tensor[0])

bfl.broadcast(tensor, root_rank=2, inplace=True)
print("Rank ", bfl.rank(), " has data ", tensor[0])

origin_tensor = torch.randn(2)
tensor = bfl.allreduce(origin_tensor, op=bfl.ReduceOp.AVG, inplace=False)
print("Rank ", bfl.rank(), " has data ", tensor[0], " after origin ", origin_tensor[0])
bfl.shutdown()
