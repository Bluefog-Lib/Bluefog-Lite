# BlueFogLite

![ci_badge](https://github.com/Bluefog-Lib/Bluefog-Lite/actions/workflows/ci.yml/badge.svg)
![Liscense](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

BlueFogLite is a lite implementation of BlueFog functionality using pure python implementation
without mandotary MPI or NCCL dependency.

The goal of BlueFogLite is to make your experimental trail easier, multi-process debugging simpler, and 
not worrying the hardward/software dependency. The performance in BlueFogLite may not be
optimized as BlueFog.  But we plan to make the interface of BlueFogLite the same as BlueFog.
So after you have verifed your concept on BlueFogLite, simply switch from BlueFogLite to 
BlueFog will automatically boost the performance.

Warning: this is still work in progress.
