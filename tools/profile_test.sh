#!/bin/bash
#
# Usage:  (under the root directory)
#   tools/profile_test.sh [--visualize]

visualize=0

for arg in $@; do
    if [[ "${arg}" == "--visualize" ]]; then
        visualize=1
    else
        echo -e "\033[31mUnknown arguments. Expected tools/profile_test.sh [--visualize]\033[0m" >&2
        exit 1
    fi
done

now=$(date +"%T")
prof_file=.profile/test.prof
mkdir .profile/
touch $prof_file

python -m cProfile -o $prof_file -m pytest -s

# Run it on local machine
if (( visualize == 1 )); then
    snakeviz $prof_file
fi
