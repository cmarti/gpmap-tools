.. _api:

API Reference
==============

Discrete spaces
---------------

.. autoclass:: gpmap.src.space.DiscreteSpace
    :members: get_neighbors, get_edges_df, write_edges_npz, write_edges_csv,
        get_state_idxs, get_neighbors, get_neighbor_pairs

.. autoclass:: gpmap.src.space.SequenceSpace
   :members: to_nucleotide_space, remove_codon_incompatible_transitions
            
Random walks
------------

.. autoclass:: gpmap.src.randwalk.TimeReversibleRandomWalk
    :members: calc_visualization, write_tables

.. autoclass:: gpmap.src.randwalk.WMWSWalk
    :members: calc_neutral_mixing_rates, calc_neutral_rate_matrix, 
        calc_stationary_frequencies, calc_rate_matrix

Landscape Inference
-------------------

.. autoclass:: gpmap.src.inference.VCregression
    :members: fit, predict, lambdas_to_variance

.. autoclass:: gpmap.src.inference.SeqDEFT
    :members: fit

Sequence utils
--------------

.. autofunction:: gpmap.src.seq.guess_space_configuration
.. autofunction:: gpmap.src.seq.get_custom_codon_table

Genotypes handling
------------------

.. autofunction:: gpmap.src.genotypes.select_genotypes

Plotting
--------

.. autofunction:: gpmap.src.plot.plot_relaxation_times
.. autofunction:: gpmap.src.plot.plot_edges
.. autofunction:: gpmap.src.plot.plot_nodes
