.. _api:

API Reference
==============

Discrete spaces
---------------

.. autoclass:: gpmap.space.DiscreteSpace
    :members: get_neighbors, get_edges_df, write_edges_npz, write_edges_csv,
        get_state_idxs, get_neighbors, get_neighbor_pairs

.. autoclass:: gpmap.space.ProductSpace

.. autoclass:: gpmap.space.GridSpace
    :members: set_peaks

.. autoclass:: gpmap.space.SequenceSpace
   :members: to_nucleotide_space, remove_codon_incompatible_transitions,
        get_single_mutant_matrix, calc_variance_components,
        calc_vjs_variance_components

.. autoclass:: gpmap.space.HammingBallSpace

Random walks
------------

.. autoclass:: gpmap.randwalk.TimeReversibleRandomWalk
    :members: calc_visualization, write_tables

.. autoclass:: gpmap.randwalk.WMWalk
    :members: calc_neutral_mixing_rates, calc_neutral_rate_matrix, 
        calc_stationary_frequencies, calc_rate_matrix, calc_model_neutral_rate_matrix

Landscape Inference
-------------------

.. autoclass:: gpmap.inference.VCregression
    :members: fit, predict, make_contrasts, lambdas_to_variance, simulate

.. autoclass:: gpmap.inference.SeqDEFT
    :members: fit, simulate_phi, simulate

Sequence utils
--------------

.. autofunction:: gpmap.seq.guess_space_configuration
.. autofunction:: gpmap.seq.get_custom_codon_table
.. autofunction:: gpmap.seq.get_one_hot_from_alleles
.. autofunction:: gpmap.seq.get_alphabet
.. autofunction:: gpmap.seq.generate_freq_reduced_alphabet
.. autofunction:: gpmap.seq.msa_to_counts

Genotypes handling
------------------

.. autofunction:: gpmap.genotypes.select_genotypes
.. autofunction:: gpmap.genotypes.read_edges
.. autofunction:: gpmap.genotypes.get_genotypes_from_region
.. autofunction:: gpmap.genotypes.marginalize_landscape_positions

Plotting
--------

.. autofunction:: gpmap.plot.plot_relaxation_times
.. autofunction:: gpmap.plot.plot_edges
.. autofunction:: gpmap.plot.plot_nodes
.. autofunction:: gpmap.plot.plot_interactive
.. autofunction:: gpmap.plot.plot_SeqDEFT_summary

Datasets
--------

.. autofunction:: gpmap.datasets.list_available_datasets
.. autoclass:: gpmap.datasets.DataSet
    :members: data, landscape, to_sequence_space, calc_visualization,
              nodes, edges, relaxation_times, plot, save