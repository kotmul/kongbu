from setuptools import setup, find_packages

setup(
    name='kongbu',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "matplotlib",
        "hydra-core",
        "wandb",
        "deepspeed",
        "flash-attn",
    ],
)