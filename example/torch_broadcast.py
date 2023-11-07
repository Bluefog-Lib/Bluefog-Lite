import torch
import torch.distributed as dist
import bluefoglite
import bluefoglite.torch as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.RingGraph(bfl.size()))
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

tensor = torch.zeros(2) + bfl.rank()
print("Rank ", bfl.rank(), " has data ", tensor)

bfl.broadcast(tensor, root_rank=1, inplace=True)
print("Rank ", bfl.rank(), " has data ", tensor, " and output ", tensor)
