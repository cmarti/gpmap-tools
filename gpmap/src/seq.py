#!/usr/bin/env python
import itertools
from _collections import defaultdict

import numpy as np
from Bio.Seq import Seq 

from gpmap.settings import NUCLEOTIDES, COMPLEMENT
from gpmap.src.utils import check_error
from gpmap.src.settings import DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET
from itertools import chain
from Bio.Data.CodonTable import CodonTable, standard_dna_table


def extend_ambigous_seq(seq, mapping):
    if not seq:
        yield('')

    else:
        character, next_seq = seq[0], seq[1:]
        if isinstance(mapping, dict):
            pos_mapping, next_mapping = mapping, mapping
        else:
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


def guess_alphabet_type(alphabet):
    set_alphabet = set(chain(*alphabet))
    if len(set_alphabet - set(DNA_ALPHABET)) == 0:
        alphabet_type = 'dna'
    elif len(set_alphabet - set(RNA_ALPHABET)) == 0:
        alphabet_type = 'rna'
    elif len(set_alphabet - set(PROTEIN_ALPHABET)) == 0:
        alphabet_type = 'protein'
    else:
        alphabet_type = 'custom'
    return(alphabet_type)
    

def guess_space_configuration(seqs, ensure_full_space=True):
    alleles = defaultdict(dict)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i][a] = 1 
    length = len(alleles)
    config = {'length': length,
              'n_alleles': [len(alleles[i]) for i in range(length)],
              'alphabet': [[a for a in alleles[i].keys()] for i in range(length)]}
    config['alphabet_type'] = guess_alphabet_type(config['alphabet'])
    
    if ensure_full_space:
        msg = 'Number of genotypes does not match the expected from guessed configuration.'
        msg += ' Ensure that genotypes span the whole sequence space'
        check_error(np.prod(config['n_alleles']) == seqs.shape[0], msg)
    return(config)


def get_custom_codon_table(aa_mapping):
    '''
    Builds a biopython CodonTable to use for translation with a custom genetic code
    
    aa_mapping: pd.DataFrame with columns "Codon" and "Letter" representing the
    genetic code correspondence. Stop codons should appear as "*"
    
    '''
    aa_mapping['Codon'] = [x.replace('U', 'T') for x in aa_mapping['Codon']]
    stop_codons = aa_mapping.loc[aa_mapping['Letter'] == '*', 'Codon'].tolist()
    aa_mapping = aa_mapping.loc[aa_mapping['Letter'] != '*', :]
    forward_table = aa_mapping.set_index('Codon')['Letter'].to_dict()
    codon_table = CodonTable(forward_table=forward_table, stop_codons=stop_codons,
                             nucleotide_alphabet='ACGT')
    codon_table.id = -1
    codon_table.names = ['Custom']
    return(codon_table)
