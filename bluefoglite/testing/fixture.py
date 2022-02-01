import datetime
import os

from bluefoglite.common.store import FileStore


def fixture_store(name):
    runtime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    shared_file_dir = os.path.join("/tmp", ".bluefoglite", name, runtime_str)
    if not os.path.exists(shared_file_dir):
        os.makedirs(shared_file_dir)
    f_store = FileStore(shared_file_dir)
    yield f_store
    f_store.close()
