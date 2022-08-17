.. _api:

API Reference
==============

Discrete spaces
---------------

.. autoclass:: gpmap.src.space.DiscreteSpace
    :members: get_neighbors, get_edges_df, write_edges_npz, write_edges_csv

.. autoclass:: gpmap.src.space.SequenceSpace
   :members: to_nucleotide_space, remove_codon_incompatible_transitions
            
Random walks
------------

.. autoclass:: gpmap.src.randwalk.TimeReversibleRandomWalk
    :members: calc_visualization, write_tables

.. autoclass:: gpmap.src.randwalk.WMWSWalk
    :members: calc_neutral_mixing_rates, calc_neutral_rate_matrix, 
        calc_stationary_frequencies, calc_rate_matrix
