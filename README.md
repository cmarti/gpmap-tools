# Visualizing fitness landscapes


## Installation

Create a python3 conda environment and activate it

```bash
conda create -n gpmap python=3.7
conda activate gpmap
```

Download the repository using git and cd into it

```bash
git clone git@bitbucket.org:cmartiga/gpmap_tools.git
```

Install using setuptools
```bash
cd gpmap_tools
python setup.py install
```

Test installation

```bash
python -m unittest gpmap/test/*py
```

# Command line usage

The basic functionality is available through few utilities in the command line after installation. 


## Calculating diffusion map

To check all the options available run
```bash
calc_visualization -h
```

We provide with a small case use for the Serine fitness landscape. We have to provide a few options:

- input file: csv separated file with sequence and function relationship sorted in a lexicographic order
- alphabet type: -A rna, dna, protein or custom. In the latter case, we also have to specify number of alleles with -a
- number of diffusion axis to calculate (5)
- effective population size: it can be provided directly with -Ns option, or through setting the mean function at stationarity at a certain value (-m) or percetile (-p)
- store also edges connecting the different genotypes for later plotting. This only needs to be done once even if we change the effective population size, as the connectivity among genotypes remains unchanged

```bash
cd gpmap/test/data
calc_visualization serine.csv -o serine -A rna -p 90 -e -k 20
```

The script outputs several files with different information:

- serine.nodes.csv: CSV file containing coordinates in each of the calculated diffusion axis for each genotype
- serine.edges.npz: npz file containing the sparse matrix encoding genotype connectivity. It could be given in CSV format also but it is less efficient for storage, reading and writting
- serine.decay_rates.csv: CSV file containing the decay rates of the different components

These files use standard formats so one could use any other custom script or language to plot the diffusion map

## Plotting the decay rates

Generally, for the visualization to be as representative as possible, we want the diffusion axis that we plot to decay as slower as possible compared with the remaining dimensions. This way we ensure that most of the long term deviations from stationarity are shown in the visualization. While this is a very simple plot to do, we also provide a command line utility to do so.

```bash
plot_decay_rates serine.decay_rates.csv -o serine.decay_rates
```

![Decay rates](https://bitbucket.org/cmartiga/gpmap_tools/raw/master/gpmap/test/data/serine.decay_rates.png)


## Plotting the fitness landscape

To check all the options available run
```bash
plot_visualization -h
```

For our case

```bash
plot_visualization serine.nodes.csv -e serine.edges.npz -o serine -nc function -s function
```

![Serine landscape](https://bitbucket.org/cmartiga/gpmap_tools/raw/master/gpmap/test/data/serine.plot.png)

Additionally, one can highlight specific genotypes with a comma separated list IUPAC encoded genotypes. For instance, if we want to highlight the different types of codons encoding Serine:

```bash
plot_visualization serine.nodes.csv -e serine.edges.npz -o serine.plot.2sets -nc function -s function -g UCN,AGY
```

![Serine landscape](https://bitbucket.org/cmartiga/gpmap_tools/raw/master/gpmap/test/data/serine.plot.2sets.png)

Or highlight directly the sequences that encode a particular aminoacid under a specific genetic code

```bash
plot_visualization serine.nodes.csv -e serine.edges.npz -o serine.plot.aa -nc function -s function -g S -A protein --protein_seq
```

![Serine landscape](https://bitbucket.org/cmartiga/gpmap_tools/raw/master/gpmap/test/data/serine.plot.aa.png)


## Interactive 3D maps

We make use of [plotly](https://plotly.com/python/) to make an interactive 3D plot of the fitness landscape that we can rotate and hover over the genotypes to show their corresponding sequences.

```bash
plot_visualization serine.nodes.csv -e serine.edges.npz -o serine.plot -nc function -s function --interactive
```

Click [here](https://bitbucket.org/cmartiga/gpmap_tools/raw/master/gpmap/test/data/serine.plot.html) for interactive visualization of the Serine landscape


## Plotting very big landscapes

For bigger landscapes, rendering can become very time consuming, specially due to the high number of edges. For this cases, one may want to explore different population sizes using only the nodes for plotting. Alternatively, we have implemented the --datashader option to use datashader library to pre-render the edges into bins for faster plotting. This option still has limited functionality compared to matplotlib based plotting e.g. genotype highlighting is not available yet



# Cite

- Mccandlish, D. M. (2011). Visualizing fitness landscapes. Evolution, 65(6), 1544–1558. [https://doi.org/10.1111/j.1558-5646.2011.01236.x](https://doi.org/10.1111/j.1558-5646.2011.01236.x)
