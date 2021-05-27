from setuptools import setup, find_packages
import os

# find version
with open('sciml_bench/__init__.py') as f:
    version = f.read().splitlines()[-1].split("'")[-2]

# framework dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# setup
setup(
    name='sciml_bench',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': ['sciml-bench=sciml_bench.core.command:cli'],
    },
    url='https://github.com/stfc-sciml/sciml-bench',
    license='MIT License',
    author='Jeyan Thiyagalingam, Juri Papay, Kuangdai Leng, \
            Samuel Jackson, Mallikarjun Shankar, \
            Geoffrey Fox, Tony Hey',
    author_email='t.jeyan@stfc.ac.uk',
    description='SciMLBench: A Benchmarking Suite for AI for Science',
)
