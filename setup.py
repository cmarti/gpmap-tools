#!/usr/bin/env python
from setuptools import setup, find_packages

VERSION = '0.3.0'


def main():
    description = 'Tools for inference and visualization of complex '
    description += 'genotype-phenotype maps'
    setup(
        name='gpmap_tools',
        version=VERSION,
        license='MIT',
        description=description,
        author='Carlos Martí-Gómez',
        author_email='martigo@cshl.edu',
        url='https://github.com/cmarti/gpmap-tools',
        packages=find_packages(),
        include_package_data=True,
        package_data = {'': ['datasets/*/serine*',
                             'datasets/*/gb1*',
                             'datasets/*/f1u*',
                             'datasets/*/smn1*',
                             'datasets/*/dmsc*',
                             'datasets/*/pard*',
                             'datasets/*/5ss*',
                             'datasets/*/trna*']},
        entry_points={
            'console_scripts': [
                'fit_SeqDEFT = bin.fit_seqdeft:main',
                'vc_regression = bin.vc_regression:main',
                'split_data = bin.split_data:main',
                'evaluate_split_fits = bin.evaluate_split_fits:main',
                'calc_visualization = bin.calc_visualization:main',
                'calc_tpt = bin.calc_tpt:main',
                'plot_visualization = bin.plot_visualization:main',
                'plot_decay_rates = bin.plot_decay_rates:main',
                'filter_genotypes = bin.filter_genotypes:main',
            ]},
        install_requires=['biopython', 'matplotlib',
                          'tqdm', 'numpy', 'scipy',
                          'pandas', 'pyarrow', 'networkx',
                          'datashader>=0.13.0',
                          'holoviews>=1.15.0',
                          'plotly>=5.9.0'],
        python_requires='>=3.8',
        platforms='ALL',
        keywords=['genotype-phenotype maps', 'fitness landscape',
                  'exact gaussian process regression'],
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
