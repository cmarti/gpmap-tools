#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.2'


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
        package_data = {'': ['datasets/*/serine*',
                             'datasets/*/gb1*',
                             'datasets/*/f1u*',
                             'datasets/*/smn1*',
                             'datasets/*/dmsc*',
                             'datasets/*/pard*',
                             'datasets/*/5ss*']},
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
        install_requires=['biopython==1.79',
                          'datashader==0.14.1', 'holoviews==1.15.0', 'plotly==5.9.0',
                          'seaborn==0.11.1', 'matplotlib==3.4.1',
                          'tqdm==4.64.0', 'jellyfish==0.11.2',
                          'fastparquet==2023.2.0', 'pandas==1.5.3',
                          'scipy==1.6.3', 'numpy==1.22.4'],
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
