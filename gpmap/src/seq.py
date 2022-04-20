#!/usr/bin/env python
import itertools

import numpy as np
from Bio.Seq import Seq 

from gpmap.settings import NUCLEOTIDES, COMPLEMENT


def extend_ambigous_seq(seq, mapping):
    if not seq:
        yield('')

    else:
        character, next_seq = seq[0], seq[1:]
        pos_mapping, next_mapping = mapping[0], mapping[1:]
        
        for allele in pos_mapping[character]:
            for seq in extend_ambigous_seq(next_seq, next_mapping):
                yield(allele + seq)


def generate_possible_sequences(l, alphabet=NUCLEOTIDES):
    for seq in itertools.product(alphabet, repeat=l):
        yield(''.join(seq))


def reverse_complement(seq):
    return(''.join(COMPLEMENT.get(x, x) for x in seq[::-1]))


def get_random_seq(length):
    return(''.join(np.random.choice(NUCLEOTIDES, size=length)))


def add_random_flanks(seq, length, only_upstream=False):
    if only_upstream:
        flank = get_random_seq(length)
        new_seq = flank + seq
    else:
        flanks = get_random_seq(2 * length)
        new_seq = flanks[:length] + seq + flanks[length:]
    return(new_seq)


def translate_seqs(seqs, codon_table='Standard'):
    prot_genotypes = np.array([str(Seq(seq).translate(table=codon_table))
                               for seq in seqs])
    return(prot_genotypes)
