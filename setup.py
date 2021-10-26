#!/usr/bin/env python

from setuptools import setup, find_packages
from gpmap.settings import VERSION


def main():
    description = 'Tools for inference and visualization of genotype-phenotype'
    description += ' maps'
    setup(
        name='gpmap_tools',
        version=VERSION,
        description=description,
        author_email='martigo@cshl.edu',
        url='https://bitbucket.org/cmartiga/gpmap_tools',
        packages=find_packages(),
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'fit_SeqDEFT = bin.fit_seqdeft:main',
            ]},
        install_requires=['numpy', 'cython', 'pandas', 'scipy', 'pysam', 
                          'seaborn', 'matplotlib', 'pystan==2.19', 'tqdm',
                          'statsmodels'],
        platforms='ALL',
        keywords=['genotype-phenotyp maps', 'fitness landscape'],
        classifiers=[
            "Programming Language :: Python :: 3",
            'Intended Audience :: Science/Research',
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
    return


if __name__ == '__main__':
    main()
