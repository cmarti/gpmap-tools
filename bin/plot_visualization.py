#!/usr/bin/env python
import argparse

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

    viz_group = parser.add_argument_group('Plotting options')
    viz_group.add_argument('-l', '--label', default=None,
                           help='Function label to show on colorbar')
    help_msg = 'Comma separated list of IUPAC codes to highlight genotypes'
    viz_group.add_argument('-g', '--genotypes', default=None, 
                           help=help_msg)
    
    viz_group.add_argument('--edges', default=False, action='store_true',
                           help='Plot edges connecting neighboring genotypes')
    viz_group.add_argument('--interactive', default=False, action='store_true',
                           help='Make interactive html')
    viz_group.add_argument('--plot3d', default=False, action='store_true',
                           help='Make 3D plot with an additional 3rd dimension')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('-o', '--output', required=True,
                              help='Output file')

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.input
    
    label = parsed_args.label
    genotypes = parsed_args.genotypes
    
    show_edges = parsed_args.edges
    interactive = parsed_args.interactive
    plot3d = parsed_args.plot3d
    out_fpath = parsed_args.output
    
    # Load counts data
    log = LogTrack()
    log.write('Start analysis')
    landscape = Visualization(fpath=data_fpath, log=log)

    if genotypes is not None:
        genotypes = str(genotypes).split(',')
    print(genotypes)
    
    if interactive:
        if plot3d:
            landscape.plot_interactive_3d(show_edges=show_edges,
                                          fpath=out_fpath)
        else:
            landscape.plot_interactive_2d(show_edges=show_edges,
                                          fpath=out_fpath)
    else:
        if plot3d:
            landscape.figure(fpath=out_fpath, x=1, y=2, z=3, show_edges=show_edges,
                             label=label)
        else:
            
            landscape.figure(fpath=out_fpath, x=1, y=2, show_edges=show_edges,
                             nodes_cmap_label=label, highlight_genotypes=genotypes)
    
    log.finish()


if __name__ == '__main__':
    main()
