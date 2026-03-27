.. _api:

API Reference
==============

Inference
---------

Regression
^^^^^^^^^^
.. autoclass:: gpmap.inference.MinimumEpistasisInterpolator
    :members: fit, predict, make_contrasts, sample_prior, simulate

.. autoclass:: gpmap.inference.VCregression
    :members: fit, predict, make_contrasts, sample_prior, simulate, get_variance_components

Density estimation
^^^^^^^^^^^^^^^^^^
.. autoclass:: gpmap.inference.SeqDEFT
    :members: fit, predict, make_contrasts, sample_prior, simulate

Summary statistics
------------------

.. autoclass:: gpmap.summary.GPmapSummarizer
    :members: calc_V_k_variance_components, calc_root_mean_squared_epistatic_coeff,
        calc_V_U_variance_components, calc_sites_variance_perc, 
        calc_site_pairs_variance_perc

Visualization
-------------

Discrete Spaces
^^^^^^^^^^^^^^^
.. autoclass:: gpmap.space.DiscreteSpace
    :members: get_edges_df, get_state_idxs, get_neighbors, get_neighbor_pairs

.. autoclass:: gpmap.space.GridSpace
    :members: get_edges_df, get_state_idxs, get_neighbors, get_neighbor_pairs, set_peaks

.. autoclass:: gpmap.space.CodonSpace
   :members: get_edges_df, get_state_idxs, get_neighbors, get_neighbor_pairs,

.. autoclass:: gpmap.space.SequenceSpace
   :members: get_edges_df, get_state_idxs, get_neighbors, get_neighbor_pairs,
        get_single_mutant_matrix, to_nucleotide_space, remove_codon_incompatible_transitions

Random walks
^^^^^^^^^^^^

.. autoclass:: gpmap.randwalk.WMWalk
    :members: set_Ns, calc_stationary_frequencies, calc_rate_matrix,
        calc_visualization, write_tables

Plotting
--------

Summary statistics
^^^^^^^^^^^^^^^^^^
.. autofunction:: gpmap.plot.mpl.plot_correlation_distance
.. autofunction:: gpmap.plot.mpl.plot_correlation_U_sites
.. autofunction:: gpmap.plot.mpl.plot_interaction_matrix
.. autofunction:: gpmap.plot.mpl.plot_kth_variance_components
.. autofunction:: gpmap.plot.mpl.plot_sites_variance_components
.. autofunction:: gpmap.plot.mpl.plot_site_pairs_variance_components

Matplotlib Backend
^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmap.plot.mpl.plot_nodes
.. autofunction:: gpmap.plot.mpl.plot_edges
.. autofunction:: gpmap.plot.mpl.plot_visualization
.. autofunction:: gpmap.plot.mpl.plot_relaxation_times
.. autofunction:: gpmap.plot.mpl.figure_Ns_grid
.. autofunction:: gpmap.plot.mpl.figure_allele_grid
.. autofunction:: gpmap.plot.mpl.figure_SeqDEFT_summary

Plotly Backend
^^^^^^^^^^^^^^

.. autofunction:: gpmap.plot.ply.plot_visualization

Datashader Backend
^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmap.plot.ds.plot_nodes
.. autofunction:: gpmap.plot.ds.plot_edges
.. autofunction:: gpmap.plot.ds.plot_visualization
.. autofunction:: gpmap.plot.ds.dsg_to_fig
.. autofunction:: gpmap.plot.ds.figure_allele_grid

Datasets
--------

.. autofunction:: gpmap.datasets.list_available_datasets
.. autoclass:: gpmap.datasets.DataSet
    :members: data, landscape, to_sequence_space, calc_visualization,
              nodes, edges, relaxation_times, plot, save

Utilities
---------

Input/Output
^^^^^^^^^^^^

.. autofunction:: gpmap.utils.read_dataframe
.. autofunction:: gpmap.utils.read_edges

Genotype dataframes
^^^^^^^^^^^^^^^^^^^

.. autofunction:: gpmap.genotypes.select_genotypes
.. autofunction:: gpmap.genotypes.get_genotypes_from_region
.. autofunction:: gpmap.genotypes.marginalize_landscape_positions

Sequence handling
^^^^^^^^^^^^^^^^^

.. autofunction:: gpmap.seq.get_custom_codon_table
.. autofunction:: gpmap.seq.generate_freq_reduced_code
.. autofunction:: gpmap.seq.msa_to_counts