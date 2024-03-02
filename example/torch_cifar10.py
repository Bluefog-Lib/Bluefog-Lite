import argparse
import os

from torchvision import datasets, transforms
import torch.utils.data.distributed
import torch.nn.functional as F

import bluefoglite.torch_api as bfl
import bluefoglite.utility as bfl_util
from bluefoglite.common import topology
from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp
from model import ResNet20, ResNet32, ResNet44, ResNet56

# Args
parser = argparse.ArgumentParser(
    description="Bluefog-Lite Example on MNIST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", type=str, default="resnet20", help="model to use")
parser.add_argument(
    "--batch_size", type=int, default=16, help="input batch size for training"
)
parser.add_argument(
    "--test_batch_size", type=int, default=16, help="input batch size for testing"
)
parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Initialize topology
bfl.init()
topo = topology.RingGraph(bfl.size())
bfl.set_topology(topo)

# Device
if args.cuda:
    print("using cuda.")
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)
else:
    print("using cpu")
    torch.manual_seed(args.seed)

# Dataloader
kwargs = {}
data_folder_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
)


# model
if args.model == "resnet20":
    model = ResNet20()
elif args.model == "resnet32":
    model = ResNet32()
elif args.model == "resnet44":
    model = ResNet44()
elif args.model == "resnet56":
    model = ResNet56()
else:
    raise NotImplementedError("model not implemented")
if args.cuda:
    model.cuda()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# Broadcast parameters & optimizer state
bfl_util.broadcast_parameters(model.state_dict(), root_rank=0)


def metric_average(val):
    tensor = torch.tensor(val)
    avg_tensor = bfl.allreduce(tensor)
    return avg_tensor.item()


def train(epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # TODO[1]: Implement unit test to check whether params in different workers are same after allreduce
            # TODO[2]: Write a function to sychronize the parameters in different workers
            bfl_util.neighbor_allreduce_parameters(model.state_dict())
        # Calculate metric
        train_loss += loss.item()
        _, pred = outputs.max(dim=1)
        total += targets.size(dim=0)
        correct += pred.eq(targets).sum().item()

        if (batch_idx + 1) % args.log_interval == 0:
            print(
                "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                    bfl.rank(),
                    epoch,
                    total,
                    len(train_sampler),
                    100.0 * total / len(train_sampler),
                    train_loss / (batch_idx + 1),
                )
            )
    train_accuracy = correct / total
    # Bluefog: average metric values across workers.
    train_loss = metric_average(train_loss)
    train_accuracy = metric_average(train_accuracy)
    if bfl.rank() == 0:
        print(
            "\nTrain Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, train_loss / len(train_loader), 100.0 * train_accuracy
            ),
            flush=True,
        )


def test(epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    for data, targets in test_loader:
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, pred = outputs.max(dim=1)
        total += targets.size(dim=0)
        correct += pred.eq(targets).sum().item()

    test_accuracy = correct / total
    # Bluefog: average metric values across workers.
    test_loss = metric_average(test_loss)
    test_accuracy = metric_average(test_accuracy)
    if bfl.rank() == 0:
        print(
            "\nTest Epoch: {}\tAverage loss: {:.6f}\tAccuracy: {:.4f}%\n".format(
                epoch, test_loss / len(test_loader), 100.0 * test_accuracy
            ),
            flush=True,
        )


for e in range(args.epochs):
    train(e)
    test(e)
    scheduler.step()
bfl.barrier()
print(f"rank {bfl.rank()} finished.")
