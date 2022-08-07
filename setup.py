#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='drllib',
    version='0.1',
    description='This repo covers various classical and deep reinforcement learning algorithm implementations.',
    url='https://github.com/khajash/deep-rl-algs',
    author='Kate Hajash',
    author_email='kshajash@gmail.com',
    packages=find_packages(exclude=['wandb', 'notebooks']),
)