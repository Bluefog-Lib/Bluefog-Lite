#!/usr/bin/env bash
#
# Compiled .proto code to python file

SRC_DIR=bluefoglite/common/tcp
protoc -I=$SRC_DIR --python_out=$SRC_DIR $SRC_DIR/message.proto
