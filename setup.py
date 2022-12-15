# Copyright 2022 JaxGaussianProcesses Contributors. All Rights Reserved.
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

import io
import os
import re

from setuptools import find_packages, setup


# Get version number (function from GPyTorch)
def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


version = find_version("jaxutils", "__init__.py")
readme = open("README.md").read()


REQUIRES = [
    "jax>=0.1.67",
    "jaxlib>=0.1.47",
    "jaxtyping"
]

EXTRAS = {
    "dev": [
        "black",
        "isort",
        "pylint",
        "flake8",
        "pytest",
        "pytest-cov",
    ],
    "cuda": ["jax[cuda]"],
}


setup(
    name="JaxUtils",
    version=version,
    author="Daniel Dodd and Thomas Pinder",
    author_email="tompinder@live.co.uk",
    packages=find_packages(".", exclude=["tests"]),
    license="LICENSE",
    description="Utility functions for JaxGaussianProcesses",
    long_description=readme,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://JaxUitls.readthedocs.io/en/latest/",
        "Source": "https://github.com/JaxGaussianProcesses/JaxUitls",
    },
    install_requires=REQUIRES,
    tests_require=EXTRAS["dev"],
    extras_require=EXTRAS,
    keywords=["gaussian-processes jax machine-learning bayesian"],
)
