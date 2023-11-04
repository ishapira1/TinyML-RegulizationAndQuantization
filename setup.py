from setuptools import setup, find_packages

setup(
    name='tinyml',
    version='0.1.0',
    author='Itai Shapira',
    author_email='itaishapira@g.harvard.edu',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add your package dependencies here
        'torch',
        'torchvision',
        # 'numpy', 'matplotlib', etc.
    ],
    python_requires='>=3.6',
)
