from setuptools import setup, find_packages

setup(
    name="federated-gnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "ray",
        "numpy",
        "pandas",
        "pyyaml",
        "matplotlib",
        "scikit-learn",
    ],
    author="Brian Bosho",
    author_email="briankipkirui03@gmail.com",
    description="Federated Learning for Graph Neural Networks",
    keywords="federated-learning, gnn, graph-neural-networks",
    python_requires=">=3.7",
)
