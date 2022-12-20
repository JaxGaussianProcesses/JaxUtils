from setuptools import find_packages, setup
import versioneer


readme = open("README.md").read()


REQUIRES = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
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
