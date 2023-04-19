#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.1'


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
        url='https://bitbucket.org/cmartiga/gpmap_tools',
        packages=find_packages(),
        include_package_data=True,
        package_data = {'': ['datasets/*/gb1*',
                             'datasets/*/f1u*',
                             'datasets/*/smn1*']},
        entry_points={
            'console_scripts': [
                'fit_SeqDEFT = bin.fit_seqdeft:main',
                'vc_regression = bin.vc_regression:main',
                'split_data = bin.split_data:main',
                'calc_split_data_r2 = bin.calc_split_data_r2:main',
                'calc_visualization = bin.calc_visualization:main',
                'calc_tpt = bin.calc_tpt:main',
                'plot_visualization = bin.plot_visualization:main',
                'plot_decay_rates = bin.plot_decay_rates:main',
                'filter_genotypes = bin.filter_genotypes:main',
            ]},
        install_requires=['biopython',
                          'datashader', 'holoviews', 'plotly', 'logomaker',
                          'seaborn', 'matplotlib', 'tqdm',
                          'fastparquet', 'pandas', 'scipy', 'numpy'],
        python_requires='>=3',
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
