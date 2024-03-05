import torch
import bluefoglite
import bluefoglite.torch_api as bfl

bfl.init()
bfl.set_topology(topology=bluefoglite.RingGraph(bfl.size()))
print(f"I am rank {bfl.rank()} among size {bfl.size()}.")

x = torch.nn.Parameter(
    torch.arange(bfl.rank(), bfl.rank() + 4).float(), requires_grad=True
)
y = torch.dot(x, x)
y.backward()

print("Rank ", bfl.rank(), " x.data: ", x.data)
print("Rank ", bfl.rank(), " x.grad: ", x.grad)

bfl.neighbor_allreduce(x.data, inplace=True)
bfl.neighbor_allreduce(x.grad, inplace=True)
print("Rank ", bfl.rank(), " x.data: ", x.data)
print("Rank ", bfl.rank(), " x.grad: ", x.grad)
