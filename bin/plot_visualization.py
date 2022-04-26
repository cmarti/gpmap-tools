#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.plot import figure_visualization
from gpmap.src.plot import plot_holoview
from scipy.sparse._matrix_io import load_npz

        
def main():
    description = 'Plot the low dimensional representation of a pre-processed'
    description += 'landscape with calc_visualization'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    help_msg = 'CSV file containing nodes coordinates and attributes'
    input_group.add_argument('nodes', help=help_msg)

    nodes_group = parser.add_argument_group('Nodes options')
    nodes_group.add_argument('-nc', '--nodes_color', default='f',
                             help='Color genotypes according to this property (f)')
    nodes_group.add_argument('-l', '--label', default=None,
                             help='Function label to show on colorbar')
    nodes_group.add_argument('--cmap', default='viridis',
                             help='Colormap to use (viridis)')
    nodes_group.add_argument('-ns', '--nodes_size', default=50, type=float,
                             help='Nodes size (50)')
    help_msg = 'Sort nodes according to this property before plotting (f)'
    nodes_group.add_argument('-s', '--sort_by', default='f', help=help_msg)
    nodes_group.add_argument('--ascending', default=False, action='store_true',
                             help='Sort nodes for plotting in ascending order')
    
    edges_group = parser.add_argument_group('Edges options')
    edges_group.add_argument('-e', '--edges', default=None,
                             help='npz file containing edges data for plotting')
    edges_group.add_argument('-ec', '--edges_color', default='grey',
                             help='Edges color (grey)')
    edges_group.add_argument('-ea', '--edges_alpha', default=0.1, type=float,
                             help='Edges transparency (0.1)')
    edges_group.add_argument('-ew', '--edges_width', default=0.5, type=float,
                             help='Edges width (0.5)')
    
    fig_group = parser.add_argument_group('Figure options')
    help_msg = 'Comma separated list of diffusion axis to display (1,2)'
    fig_group.add_argument('-a', '--axis', default='1,2', help=help_msg)
    fig_group.add_argument('--interactive', default=False, action='store_true',
                           help='Make interactive html')
    fig_group.add_argument('--datashader', default=False, action='store_true',
                           help='Use datashader for plotting. Recommended for big landscapes')
    fig_group.add_argument('-r', '--resolution', default=600, type=int,
                           help='Resolution for datashader plotting (600)')
    
    highlight_group = parser.add_argument_group('Highlight genotypes options')    
    help_msg = 'Comma separated list of IUPAC codes to highlight genotypes'
    highlight_group.add_argument('-g', '--genotypes', default=None, 
                           help=help_msg)
    highlight_group.add_argument('-A', '--alphabet_type', default='dna',
                                 help='Alphabet type [dna, rna, protein, custom] (dna)')
    help_msg = 'Sequences to highlight are the encoded protein sequences'
    highlight_group.add_argument('--protein_seq', default=False, action='store_true', 
                                 help=help_msg)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    nodes_fpath = parsed_args.nodes
    
    nodes_color = parsed_args.nodes_color
    label = parsed_args.label
    nodes_cmap = parsed_args.cmap
    nodes_size = parsed_args.nodes_size
    sort_by = parsed_args.sort_by
    ascending = parsed_args.ascending
    
    edges_fpath = parsed_args.edges
    edges_color = parsed_args.edges_color
    edges_alpha = parsed_args.edges_alpha
    edges_width = parsed_args.edges_width
    
    axis = parsed_args.axis.split(',')
    if len(axis) == 2:
        axis.append(None)
    x, y, z = axis
    interactive = parsed_args.interactive
    use_datashader = parsed_args.datashader
    resolution = parsed_args.resolution 
    
    genotypes = parsed_args.genotypes
    alphabet_type = parsed_args.alphabet_type
    is_prot = parsed_args.protein_seq
    
    out_fpath = parsed_args.output
    
    # Load data
    log = LogTrack()
    log.write('Start analysis')
    
    log.write('Reading genotype data from {}'.format(nodes_fpath))
    nodes_df = pd.read_csv(nodes_fpath, index_col=0)
    
    if edges_fpath is not None:
        log.write('Reading edges data from {}'.format(edges_fpath))
        A = load_npz(edges_fpath).tocoo()
        edges_df = pd.DataFrame({'i': A.row, 'j': A.col}) 
    else:
        log.write('No edges provided for plotting')
        edges_df = None
    
    if genotypes is not None:
        genotypes = str(genotypes).split(',')
    
    log.write('Plot visualization')
    if use_datashader:
        plot_holoview(nodes_df, out_fpath, x=x, y=y, edges_df=edges_df,
                      nodes_color=nodes_color, nodes_cmap=nodes_cmap,
                      edges_cmap='grey', background_color='white',
                      resolution=resolution)
    else:
        figure_visualization(nodes_df, edges_df=edges_df,
                             fpath=out_fpath, x=x, y=y, z=z, 
                             nodes_color=nodes_color, nodes_cmap=nodes_cmap, 
                             nodes_size=nodes_size, nodes_cmap_label=label,
                             edges_color=edges_color,
                             edges_width=edges_width, edges_alpha=edges_alpha,
                             sort_by=sort_by, ascending=ascending, sort_nodes=True,
                             highlight_genotypes=genotypes,
                             is_prot=is_prot, interactive=interactive,
                             alphabet_type=alphabet_type)
    
    log.finish()


if __name__ == '__main__':
    main()
