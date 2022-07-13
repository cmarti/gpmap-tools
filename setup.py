#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.0'


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
                'calc_visualization = bin.calc_visualization:main',
                'calc_tpt = bin.calc_tpt:main',
                'plot_visualization = bin.plot_visualization:main',
                'plot_decay_rates = bin.plot_decay_rates:main',
                'filter_genotypes = bin.filter_genotypes:main',
            ]},
        install_requires=['biopython==1.79', 'datashader', 'holoviews',
                          'plotly==5.6.0', 'logomaker==0.8',
                          'seaborn==0.11.2', 'matplotlib==3.5.1',
                          'tqdm==4.63.0',
                          'pandas==1.3.5', 'scipy==1.7.3', 'numpy==1.21.5'],
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
