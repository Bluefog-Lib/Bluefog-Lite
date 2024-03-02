import torch
import bluefoglite.torch_api as bfl


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the ``model.state_dict()``,
    ``model.named_parameters()``, or ``model.parameters()``.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # Run asynchronous broadcasts.
    async_works = []
    for name, p in params:
        async_work = bfl.broadcast_nonblocking(p, inplace=True, root_rank=root_rank)
        async_works.append(async_work)

    # Wait for completion.
    for async_work in async_works:
        async_work.wait()


def neighbor_allreduce_parameters(params):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError("invalid params of type: %s" % type(params))

    # Run asynchronous broadcasts.
    async_works = []
    for name, p in params:
        if torch.is_floating_point(p):
            async_work = bfl.neighbor_allreduce_nonblocking(p, inplace=True)
            async_works.append(async_work)

    # Wait for completion.
    for async_work in async_works:
        async_work.wait()
