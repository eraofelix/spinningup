from os.path import join, dirname, realpath
from setuptools import setup
import sys

with open(join("spinup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup',
    py_modules=['spinup'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle',
        'ipython',
        'joblib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn',
        'tqdm',
        'gymnasium',
        'gymnasium_robotics',
        'tensorboard',
        'swig',
        'tensorboard',
        'scipy'
        # 'gymnasium[box2d]'  需要最后单独安装
    ],
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)
