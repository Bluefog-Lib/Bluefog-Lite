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
from bluefoglite.common import topology
from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp

# Args
cuda = torch.cuda.is_available()
seed = 42
batch_size = 16
test_batch_size = 16
log_interval = 100
epochs = 5
lr = 0.001

# Initialize topology
bfl.init()
topo = topology.RingGraph(bfl.size())
bfl.set_topology(topo)

# Device
if cuda:
    print("using cuda.")
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
else:
    print("using cpu")

# Dataloader
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


# Model
def MLP():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


model = MLP()
if cuda:
    model.cuda()


# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Broadcast parameters & optimizer state
bfl_util.broadcast_parameters(model.state_dict(), root_rank=0)


def metric_average(val):
    tensor = torch.tensor(val)
    avg_tensor = bfl.allreduce(tensor)
    return avg_tensor.item()


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # TODO[1]: Implement unit test to check whether params in different workers are the same
            # TODO[2]: Write a function to sychronize the parameters in different workers
            for module in model.parameters():
                bfl.allreduce(module.data, op=ReduceOp.AVG, inplace=True)
        if batch_idx % log_interval == 0:
            print(
                "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                    bfl.rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(epoch):
    model.eval()
    test_loss, test_accuracy, total = 0.0, 0.0, 0.0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        total += len(target)
    test_loss /= total
    test_accuracy /= total
    # Bluefog: average metric values across workers.
    test_loss = metric_average(test_loss)
    test_accuracy = metric_average(test_accuracy)
    if bfl.rank() == 0:
        print(
            "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, test_loss, 100.0 * test_accuracy
            ),
            flush=True,
        )


for e in range(epochs):
    train(e)
    test(e)
bfl.barrier()
print(f"rank {bfl.rank()} finished.")
