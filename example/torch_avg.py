import torch
import torch.distributed as dist
import bluefoglite
import bluefoglite.torch_api as bfl

import argparse
from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--data-size", type=int, default=4)
parser.add_argument("--max-iters", type=int, default=200)
parser.add_argument("--plot-interactive", action="store_true")
parser.add_argument("--backend", type=str, default="gloo")
parser.add_argument("--consensus-method", type=str, default="neighbor_allreduce")
# setattr(args, "data_size", 4)
# setattr(args, "max_iters", 200)
# setattr(args, "plot_interactive", True)
# setattr(args, "backend", "gloo")
# setattr(args, "consensus_method", "neighbor_allreduce")
args = parser.parse_args()

# choices: gloo, mpi, nccl
bfl.init(backend="gloo")
bfl.set_topology(bluefoglite.RingGraph(bfl.size(), connect_style=0))

device = bfl.rank() % torch.cuda.device_count()
x = torch.randn(args.data_size, device=device, dtype=torch.double)
x_bar = bfl.allreduce(x, op=ReduceOp.AVG)
mse = [torch.norm(x - x_bar, p=2) / torch.norm(x_bar, p=2)]


for ite in range(args.max_iters):
    x = eval(f"bfl.{args.consensus_method}(x, inplace=False)")
    mse.append(torch.norm(x - x_bar, p=2) / torch.norm(x_bar, p=2))


mse = [m.item() for m in mse]
print("MSE at last iteration: ", mse[-1])
if args.plot_interactive and bfl.rank() == 0:
    import matplotlib.pyplot as plt

    plt.semilogy(mse)
    plt.savefig(f"./img/torch_avg_{args.consensus_method}.png")
    plt.show()
    plt.close()
bfl.shutdown()
