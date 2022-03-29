#!/usr/bin/env python
import argparse

import pandas as pd

from gpmap.utils import LogTrack
from gpmap.visualization import Visualization

        
def main():
    description = 'Plot the low dimensional representation of a pre-processed'
    description += 'landscape with calc_visualization'
    
    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group('Input')
    help_msg = 'Pickle file containing calc_visualization output'
    input_group.add_argument('input', help=help_msg)

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
    edges_group.add_argument('--edges', default=False, action='store_true',
                             help='Plot edges connecting neighboring genotypes')
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
    
    highlight_group = parser.add_argument_group('Highlight genotypes options')    
    help_msg = 'Comma separated list of IUPAC codes to highlight genotypes'
    highlight_group.add_argument('-g', '--genotypes', default=None, 
                           help=help_msg)
    help_msg = 'Sequences to highlight are the encoded protein sequences'
    highlight_group.add_argument('--protein_seq', default=False, action='store_true', 
                                 help=help_msg)
    highlight_group.add_argument('-c', '--codon_table', default='Standard', 
                                 help='NCBI Codon table to use for translation (Standard)')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    
    nodes_color = parsed_args.nodes_color
    label = parsed_args.label
    nodes_cmap = parsed_args.cmap
    nodes_size = parsed_args.nodes_size
    sort_by = parsed_args.sort_by
    ascending = parsed_args.ascending
    
    show_edges = parsed_args.edges
    edges_color = parsed_args.edges_color
    edges_alpha = parsed_args.edges_alpha
    edges_width = parsed_args.edges_width
    
    axis = [int(x) for x in parsed_args.axis.split(',')]
    if len(axis) == 2:
        axis.append(None)
    x, y, z = axis
    interactive = parsed_args.interactive
    
    genotypes = parsed_args.genotypes
    is_prot = parsed_args.protein_seq
    codon_table = parsed_args.codon_table
    
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    landscape = Visualization(fpath=data_fpath, log=log)
    if is_prot:
        landscape.prot = pd.Series(landscape.get_protein_seq(table=codon_table),
                                   index=landscape.genotype_labels)

    if genotypes is not None:
        genotypes = str(genotypes).split(',')
    
    landscape.figure(fpath=out_fpath, x=x, y=y, z=z, 
                     nodes_color=nodes_color, nodes_cmap=nodes_cmap, 
                     nodes_size=nodes_size, nodes_cmap_label=label,
                     show_edges=show_edges, edges_color=edges_color,
                     edges_width=edges_width, edges_alpha=edges_alpha,
                     sort_by=sort_by, ascending=ascending, sort_nodes=True,
                     highlight_genotypes=genotypes,
                     is_prot=is_prot, interactive=interactive)
    
    log.finish()


if __name__ == '__main__':
    main()
