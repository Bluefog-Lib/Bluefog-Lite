#!/bin/bash
now=$(date +"%T")
prof_file=.profile/test.prof
mkdir .profile/
touch $prof_file

python -m cProfile -o $prof_file -m pytest -s

# Run it on local machine
snakeviz $prof_file