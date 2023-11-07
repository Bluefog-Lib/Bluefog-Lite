import torch
import torch.distributed as dist
import bluefoglite
import bluefoglite.torch as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.ExponentialGraph(bfl.size()))
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

tensor = torch.zeros(2) + bfl.rank()
print("Rank ", bfl.rank(), " has data ", tensor)

output_tensor = bfl.neighbor_allreduce(tensor, inplace=False)
print("Rank ", bfl.rank(), " has data ", tensor, " and output ", output_tensor)