from setuptools import find_packages, setup

setup(
    name="fst",
    version="0.0.1",
    author="Elliot Fosong",
    author_email="e.fosong@ed.ac.uk",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy>=1.23.1",
        "gym>=0.26.1",
        "pettingzoo==1.22.0",
        "hydra-core>=1.2.0",
        "pandas>=1.4.0",
        "git+https://github.com/efosong/VectorizedMultiAgentSimulator.git@aamas24"
        "git+https://github.com/efosong/gym-cooking.git@aamas24"
        "git+https://github.com/efosong/SuperSuit.git@aamas24"
        ]
)
