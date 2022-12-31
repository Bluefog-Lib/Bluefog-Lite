#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

mypy --install-types
# I don't know why this approach cannot revmove the check on *_pb2.py files
# in Github CI. But it works perfectly in local machine
mypy_files=$(find bluefoglite -name "*.py" ! -name '*_pb2.py')
mypy --config-file=tools/.mypy.ini "$@" ${mypy_files}
