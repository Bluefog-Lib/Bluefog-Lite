import bluefoglite.torch as bfl

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
    # handles = []
    for name, p in params:
        bfl.broadcast_nonblocking(p, root_rank=root_rank)
        # handle = bfl.broadcast_nonblocking_(p, root_rank, name)
        # handles.append(handle)

    # Wait for completion.
    # for handle in handles:
    #     bfl.synchronize(handle)