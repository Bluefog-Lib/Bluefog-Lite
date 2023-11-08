import argparse
import copy
import os

from torchvision import datasets, transforms, models
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import bluefoglite.torch_api as bfl
import bluefoglite.utility as bfl_util
from bluefoglite.common import topology_util
from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp


# Training settings
def MLP():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


cuda = torch.cuda.is_available()
seed = 42
batch_size = 16
test_batch_size = 16
epochs = 10
lr = 0.001

bfl.init()
topo = topology_util.RingGraph(bfl.size())
bfl.set_topology(topo)
# topo_gen = GetOptTopoSendRecvRanks(topo, bfl.rank())

if bfl.rank() == 0:
    import wandb

    wandb.init(project="bfl-test", name="torch_mnist")

if cuda:
    print("using cuda.")
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
else:
    print("using cpu")

kwargs = {}
data_folder_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

train_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
)

test_dataset = datasets.MNIST(
    os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs
)


model = MLP()
# model_hat = MLP()

# Force hat and orignal model to be the samee and among all agents.
# model_hat.load_state_dict(copy.deepcopy(model.state_dict()))

if cuda:
    model.cuda()
    # model_hat.cuda()

# broadcast model
bfl_util.broadcast_parameters(model.state_dict(), root_rank=0)
# print("model parameters: ", model.state_dict())


def train(epoch):
    model.train()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        model.zero_grad()
        output = model(data.cuda())
        loss = F.cross_entropy(output, target.cuda())
        loss.backward()
        with torch.no_grad():
            for module in model.parameters():
                module.data.add_(module.grad.data, alpha=-lr)
                # print("before average: ", bfl.rank(), " ", list(model.parameters())[0].data[0][0])
            for module in model.parameters():
                bfl.allreduce(module.data, op=ReduceOp.AVG, inplace=True)
                # print("after average: ", bfl.rank(), " ", list(model.parameters())[0].data[0][0])
        correct += (output.argmax(dim=1) == target.cuda()).sum().item()
        total += len(target)
    print(
        "rank {}: Train Epoch: {}\tLoss: {:.6f}\tAccuracy: {:.4f}%".format(
            bfl.rank(),
            epoch,
            loss.item(),
            100.0 * correct / total,
        )
    )


def test():
    model.eval()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.cuda())
        correct += (output.argmax(dim=1) == target.cuda()).sum().item()
        total += len(target)
    # correct = bfl.allreduce(correct, op=ReduceOp.SUM)
    # total = bfl.allreduce(total, op=ReduceOp.SUM)
    print(
        "rank {}: Total test\tAccuracy: {:.4f}%".format(
            bfl.rank(),
            100.0 * correct / total,
        )
    )
    correct, total = 0, 0


for e in range(epochs):
    train(e)
print(
    f"rank {bfl.rank()} finished training, parameters: {list(model.parameters())[0].data[0][0]}"
)
test()
