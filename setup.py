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
from setuptools import find_packages, setup
import versioneer
import os


readme = open("README.md").read()
NAME = "jaxutils"


# Handle builds of nightly release
if "BUILD_JAXUTILS_NIGHTLY" in os.environ:
    NAME += "-nightly"

    from versioneer import get_versions as original_get_versions

    def get_versions():
        from datetime import datetime, timezone

        suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
        versions = original_get_versions()
        versions["version"] = versions["version"].split("+")[0] + suffix
        return versions

    versioneer.get_versions = get_versions


REQUIRES = ["jax>=0.4.0", 
            "jaxlib>=0.4.0", 
            "jaxtyping", 
            "ml-collections==0.1.0",
            "distrax>=0.1.2",
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
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
