from distutils.core import setup

setup(
    name="Tiny Ml",
    version="1.0",
    packages=[
        "src.data_loaders",
        "src.models",
        "src.quantization",
        "src.training",
    ],
    install_requires=[
        'numpy',  # Replace with actual dependencies
        'torch',
        'tqdm',
        'pandas',
        'brevitas'
        # Add any other dependencies that your package needs
    ],
)
