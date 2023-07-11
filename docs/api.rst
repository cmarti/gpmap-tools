.. _api:

API Reference
==============

Discrete spaces
---------------

.. autoclass:: gpmap.src.space.DiscreteSpace
    :members: get_neighbors, get_edges_df, write_edges_npz, write_edges_csv,
        get_state_idxs, get_neighbors, get_neighbor_pairs

.. autoclass:: gpmap.src.space.SequenceSpace
   :members: to_nucleotide_space, remove_codon_incompatible_transitions,
        get_single_mutant_matrix, calc_variance_components,
        calc_vjs_variance_components

.. autoclass:: gpmap.src.space.HammingBallSpace
            
Random walks
------------

.. autoclass:: gpmap.src.randwalk.TimeReversibleRandomWalk
    :members: calc_visualization, write_tables

.. autoclass:: gpmap.src.randwalk.WMWalk
    :members: calc_neutral_mixing_rates, calc_neutral_rate_matrix, 
        calc_stationary_frequencies, calc_rate_matrix, calc_model_neutral_rate_matrix

Landscape Inference
-------------------

.. autoclass:: gpmap.src.inference.VCregression
    :members: fit, predict, lambdas_to_variance, project, simulate, calc_L_polynomial_coeffs

.. autoclass:: gpmap.src.inference.SeqDEFT
    :members: fit, simulate_phi, simulate

Sequence utils
--------------

.. autofunction:: gpmap.src.seq.guess_space_configuration
.. autofunction:: gpmap.src.seq.get_custom_codon_table
.. autofunction:: gpmap.src.seq.get_one_hot_from_alleles
.. autofunction:: gpmap.src.seq.get_alphabet
.. autofunction:: gpmap.src.seq.generate_freq_reduced_alphabet
.. autofunction:: gpmap.src.seq.msa_to_counts

Genotypes handling
------------------

.. autofunction:: gpmap.src.genotypes.select_genotypes
.. autofunction:: gpmap.src.genotypes.read_edges
.. autofunction:: gpmap.src.genotypes.get_genotypes_from_region
.. autofunction:: gpmap.src.genotypes.marginalize_landscape_positions

Plotting
--------

.. autofunction:: gpmap.src.plot.plot_relaxation_times
.. autofunction:: gpmap.src.plot.plot_edges
.. autofunction:: gpmap.src.plot.plot_nodes
.. autofunction:: gpmap.src.plot.plot_interactive
.. autofunction:: gpmap.src.plot.plot_SeqDEFT_summary

