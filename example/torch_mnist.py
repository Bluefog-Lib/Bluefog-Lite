import argparse
import copy
import os

from torchvision import datasets, transforms, models
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import bluefoglite.torch as bfl
from bluefoglite.common import topology_util

from opt_topo import GetOptTopoSendRecvRanks, GetOptTopoSendRecvRanks1Port

import collections
from scipy.io import savemat

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.5,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=20,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
)
parser.add_argument('--dataset', type=str, default='mnist',
                    help='The dataset to train with.')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


def opt_topo_avg(x, x_hat, send_ranks, recv_ranks, n_ex, dig):
    self_weight = 0.0
    src_weights = {recv_ranks[0]: 1.0}
    dst_weights = {send_ranks[0]: 1.0}

    # TODO some strange bug of using cuda tensor
    send_x = x.cpu() if dig == 1 else x_hat.cpu()
    comb_x = bfl.neighbor_allreduce(
        send_x,
        # name="x",
        self_weight=self_weight,
        # neighbor_weights=src_weights,
        # send_neighbors=send_ranks,
        src_weights=src_weights,
        dst_weights=dst_weights,
        # enable_topo_check=False,
    )
    if x.device != "cpu":
        comb_x = comb_x.to(x.device)
    
    if dig == 1:
        new_x = 0.5 * x + 0.5 * comb_x
        new_x_hat = n_ex / (2 * n_ex + 1) * x_hat + (n_ex + 1) / (2 * n_ex + 1) * comb_x
    else:
        new_x = (n_ex + 1) / (2 * n_ex + 1) * x + n_ex / (2 * n_ex + 1) * comb_x
        new_x_hat = 0.5 * x_hat + 0.5 * comb_x
    return new_x, new_x_hat


def set_all_param_zero(model):
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()


def set_flatten_model_back(model, x_flattern):
    with torch.no_grad():
        start = 0
        for p in model.parameters():
            p_extract = x_flattern[start : (start + p.numel())]
            p.set_(p_extract.view(p.shape))
            p.grad.zero_()
            start += p.numel()


def get_flatten_model_param(model):
    with torch.no_grad():
        return torch.cat(
            [p.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )


def get_flatten_model_grad(model):
    with torch.no_grad():
        return torch.cat(
            [p.grad.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )

# Alg:
# x     = a(x-\gamma g_hat) + (1-a) P(z-\gamma e)
# x_hat = b(x_hat-\gamma g_hat) + (1-b) P(z-\gamma e)

args = parser.parse_args()
args.cuda = (not args.no_cuda) and (torch.cuda.is_available())

bfl.init()
topo = topology_util.ExponentialGraph(bfl.size())
bfl.set_topology(topo)
topo_gen = GetOptTopoSendRecvRanks(topo, bfl.rank())
# topo_gen = GetOptTopoSendRecvRanks1Port(topo, bfl.rank()) 

if bfl.rank() == 0:
    import wandb
    wandb.init(project='bfl-test', name='torch_mnist', config=args)

if args.cuda:
    print("using cuda.")
    # Bluefog: pin GPU to local rank.
    # device_id = bfl.local_rank() if bfl.nccl_built() else bfl.local_rank() % torch.cuda.device_count()
    device_id = bfl.rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(args.seed)
else:
    print("using cpu")


kwargs = {}
data_folder_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if args.dataset == "cifar10":
    train_dataset = datasets.CIFAR10(
        os.path.join(data_folder_loc, "..", "data", "data-%d" % bfl.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "mnist":
    train_dataset = datasets.MNIST(
        os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

# Bluefog: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs
)

if args.dataset == "cifar10":
    test_dataset = datasets.CIFAR10(
        os.path.join(data_folder_loc, "..", "data", "data-%d" % bfl.rank()),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
elif args.dataset == "mnist":
    test_dataset = datasets.MNIST(
        os.path.join(data_folder_loc, "data", "data-%d" % bfl.rank()),
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
test_sampler = None
# Bluefog: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bfl.size(), rank=bfl.rank()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs
)

if args.dataset == "cifar10":
    model = models.resnet18()
    model_hat = models.resnet18()
elif args.dataset == "mnist":  
    model = Net()
    model_hat = Net()

# Force hat and orignal model to be the samee and among all agents.
model_hat.load_state_dict(copy.deepcopy(model.state_dict()))  

if args.cuda:
    # Move model to GPU.
    model.cuda()
    model_hat.cuda()
    
bfl.broadcast_parameters(model.state_dict(), root_rank=0)

# Bluefog: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)  # pylint: disable=not-callable
        self.n = torch.tensor(0.0)  # pylint: disable=not-callable

    def update(self, val):
        # self.sum += bfl.allreduce(val.detach().cpu(), name=self.name)
        self.sum += bfl.allreduce(val.detach().cpu())
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().float().mean()

def train(epoch):
    model.train()
    # Bluefog: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    train_accuracy = Metric("train_accuracy")

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Dynamic one peer topology
        send_ranks, recv_ranks, n_ex, dig = next(topo_gen)

        # optimizer.zero_grad()
        output = model(data)
        output_hat = model_hat(data)

        train_accuracy.update(accuracy(output, target))

        if args.dataset == 'mnist':
            loss = F.nll_loss(output, target)
            loss_hat = F.nll_loss(output_hat, target)
        else:
            loss = F.cross_entropy(output, target)
            loss_hat = F.cross_entropy(output_hat, target)

        train_loss.update(loss)

        loss.backward()
        loss_hat.backward()
        # optimizer.step()

        with torch.no_grad():
            flatten_x = get_flatten_model_param(model)
            flatten_x_hat = get_flatten_model_param(model_hat)
            flatten_grad = get_flatten_model_grad(model)
            flatten_grad_hat = get_flatten_model_grad(model_hat)
            grad = flatten_grad if dig == 1 else flatten_grad_hat
            x = flatten_x - args.lr * bfl.size() * grad
            x_hat = flatten_x_hat - args.lr * bfl.size() * grad

        x_update, x_update_hat = opt_topo_avg(x, x_hat, send_ranks, recv_ranks, n_ex, dig)
        set_flatten_model_back(model, x_update)
        set_flatten_model_back(model_hat, x_update_hat)

        if batch_idx % args.log_interval == 0:
            # Bluefog: use train_sampler to determine the number of examples in
            # this worker's partition.
            print(
                "[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    bfl.rank(),
                    epoch,
                    batch_idx * len(data),
                    len(train_sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return train_loss.avg, train_accuracy.avg


def test(record):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0


    val_accuracy = Metric("val_accuracy")

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        output = model(data)

        val_accuracy.update(accuracy(output, target))

        # sum up batch loss
        if args.dataset == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()
        else:
            test_loss += F.cross_entropy(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()

    # Bluefog: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler) if test_sampler else len(test_dataset)
    test_accuracy /= len(test_sampler) if test_sampler else len(test_dataset)

    # Bluefog: average metric values across workers.
    # test_loss = metric_average(test_loss, "avg_loss")
    # test_accuracy = metric_average(test_accuracy, "avg_accuracy")

    # Bluefog: print output only on first rank.
    if bfl.rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                test_loss, 100.0 * test_accuracy
            ),
            flush=True,
        )
    record.append((test_loss, 100.0 * test_accuracy))

    return val_accuracy.avg


test_record = []
result_dict = collections.defaultdict(list)

for i in range(args.epochs):
    train_loss_record, train_acc_record = train(i)
    test_acc_record = test(test_record)
    result_dict['train_loss'].append(train_loss_record)
    result_dict['train_accuracy'].append(train_acc_record)
    result_dict['val_accuracy'].append(test_acc_record)


if bfl.rank() == 0:

    fname = "Opt_"
    fname += "Nodes" + str(bfl.size())
    if args.dataset == "mnist":
        fname += "_mnist"
    else:
        fname += "_cifar10"
    
    fname += "_lr" + str(args.lr)
    fname += "_epochs" + str(args.epochs)
    fname += "_batchsize" + str(args.batch_size)
    fname += "_seed" +str(args.seed)
    savemat('output/' + fname + ".mat", result_dict)