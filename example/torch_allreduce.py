import torch
import bluefoglite
import bluefoglite.torch_api as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.RingGraph(bfl.size()))
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

tensor = torch.zeros(2) + bfl.rank()
print("Rank ", bfl.rank(), " has data ", tensor)

bfl.allreduce(tensor, inplace=True)
print("Rank ", bfl.rank(), " has data ", tensor, " and output ", tensor)
