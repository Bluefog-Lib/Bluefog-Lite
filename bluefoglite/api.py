from bluefoglite.common.basics import BlueFogLiteGroup

_global_group = BlueFogLiteGroup()

# import basic methods and wrap it with default global group.


def init(group=None):
    if group is None:
        group = _global_group
    group.init()


def shutdown(group=None):
    if group is None:
        group = _global_group
    group.shutdown()


def size(group=None):
    if group is None:
        group = _global_group
    return group.size()


def rank(group=None):
    if group is None:
        group = _global_group
    return group.rank()


def send(dst, obj_or_array, *, tag=0, group=None):
    if group is None:
        group = _global_group
    group.send(dst=dst, obj_or_array=obj_or_array,  tag=tag)


def recv(src, obj_or_array, *, tag=0, group=None):
    if group is None:
        group = _global_group
    return group.recv(src=src, obj_or_array=obj_or_array,  tag=tag)
