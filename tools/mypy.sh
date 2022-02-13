#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

files=$(find bluefoglite -name "*.py" ! -name '*_pb2.py')
mypy --config-file=tools/.mypy.in "$@" ${files}
