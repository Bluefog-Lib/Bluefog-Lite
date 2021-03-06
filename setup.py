# Copyright 2021 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import find_packages, setup  # type: ignore
from bluefoglite.version import __version__

NAME = "bluefoglite"
DESCRIPTION = "A lite implementation for Bluefog"
EMAIL = "bichengying@gmail.com"
AUTHOR = "Bicheng Ying"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = __version__

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt") as f:
    requirements = list(f.read().strip().split("\n"))


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url="https://github.com/bluefog-lib/bluefog",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=">=3.7.0",
    install_requires=requirements,
    packages=find_packages(exclude=["test", "examples"]),
    license=license,
    entry_points={"console_scripts": ["bflrun = bluefoglite.launch.run:main"]},
)
