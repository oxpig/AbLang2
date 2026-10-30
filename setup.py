# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name='ablang2',
    version='0.1.1',
    license='BSD 3-clause license',
    description='AbLang2: An antibody-specific language model focusing on NGL prediction.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Tobias Hegelund Olsen",
    maintainer='Tobias Hegelund Olsen',
    maintainer_email='tobiasheol@gmail.com',
    url="",
    include_package_data=True,
    packages=find_packages(include=('ablang2', 'ablang2.*')),
    install_requires=[
        'torch>1.9',
        'requests',
        'einops',
        'rotary-embedding-torch',
        'numpy',
    ],
    extras_require={
        # Needed only for `align=True` and the `--align` option.  ANARCI is
        # also needed, but it is deliberately absent here: the `anarci`
        # package on PyPI is not an official release, so ANARCI must come
        # from Bioconda (`bioconda::anarci`).
        'align': ['pandas'],
    },
    entry_points={
        'console_scripts': ['ablang2=ablang2.cli:main'],
    },
)