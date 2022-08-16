.. _api:

API Reference
==============

Discrete spaces
---------------

.. autoclass:: gpmap.src.space.DiscreteSpace
    :members: get_neighbors, get_edges_df, write_edges_npz, write_edges_csv

.. autoclass:: gpmap.src.space.SequenceSpace
   :members: to_nucleotide_space, remove_codon_incompatible_transitions
             
