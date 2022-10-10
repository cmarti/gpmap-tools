#!/usr/bin/env python
import itertools
from _collections import defaultdict

import numpy as np
from Bio.Seq import Seq 

from gpmap.src.settings import NUCLEOTIDES, COMPLEMENT, ALPHABET
from gpmap.src.utils import check_error
from gpmap.src.settings import DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET
from itertools import chain
from Bio.Data.CodonTable import CodonTable


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


def transcribe_seqs(seqs, code):
    new_seqs = np.array([''.join([sdict[a] for a, sdict in zip(seq, code)])
                         for seq in seqs])
    return(new_seqs)


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
    '''
    Guess the sequence space configuration from a collection of sequences
    This allows to have different number of alleles per site and maintain 
    the order in which alleles appear in the sequences when enumerating the 
    alleles per position
    
    Parameters
    ----------
    seqs: array-like of shape (n_genotypes,)
        Vector or list containing the sequences from which we want to infer
        the space configuration
        
    ensure_full_space: bool
        Option to ensure that the whole sequence space must be represented by 
        the set of provided sequences. This is a useful feature to identify
        whether there are missing genotypes before defining the space and
        random walk to visualize the full landscape.
    
       
    Returns
    -------
    
    config: dict with keys {'length', 'n_alleles', 'alphabet'}
            Returns a dictionary with the inferred configuration of the discrete
            space where the sequences come from.
    
    '''
    
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
        msg += ' Ensure that genotypes span the whole sequence space or use'
        msg += '`ensure_full_space` option to avoid this error'
        check_error(np.prod(config['n_alleles']) == seqs.shape[0], msg)
    return(config)


def generate_freq_reduced_code(seqs, n_alleles, counts=None,
                               keep_allele_names=True, last_character='X'):
    '''
    Returns a list of dictionaries with the mapping from each allele in the
    observed sequences to a reduced alphabet with at most ``n_alleles`` per site.
    The least frequent alleles are pooled together into a single allele
    
    Parameters
    ----------
    seqs : array-like of shape (n_genotypes,) or (n_obs,)
        Observed sequences. If ``counts=None``, then every sequence is counted
        once. Otherwise, frequencies are calculated using the counts as the
        number of times a certain sequence appears in the data
    
    n_alleles : int or array-like of shape (seq_length, )
        Maximal number of alleles per site allowed. If a list or array is provided
        each site will use the specified number of alleles. Otherwise, all
        sites will have the same maximum number of alleles
        
    counts : None or array-like of shape (n_genotypes, )
        Number of times every sequence in ``seqs`` appears in the data. If
        not provided, every provided sequence is assumed to appear exactly once
        
    keep_allele_names : bool
        If ``keep_allele_names=True``, then allele names are preserved. Otherwise
        they are replace by new alleles taken from the alphabet
        
    last_character : str
        Character to use for remaining alleles when ``keep_allele_names=True``
        
    Returns
    -------
    code : list of dict of length seq_length
        List of dictionaries containing the new allele corresponding to each
        of the original alleles for each site.
    
    '''

    if counts is None:
        counts = itertools.cycle([1])
    else:
        msg = 'counts must have the same shape as seqs'
        check_error(counts.shape == seqs.shape, msg=msg)
    
    seq_length = len(seqs[0])
    freqs = [defaultdict(lambda: 0) for _ in range(seq_length)]
    for seq, c in zip(seqs, counts):
        for i, allele in enumerate(seq):
            freqs[i][allele] += c
    
    alleles = [sorted(site_freqs.keys(), key=lambda x: site_freqs[x], reverse=True)
               for site_freqs in freqs]
    
    if keep_allele_names:
        new_alleles = [a[:n_alleles] + [last_character] for a in alleles]
    else:
        new_alleles = [ALPHABET[:n_alleles]] * seq_length
        
    reduced_alphabet = []
    for site_alleles, site_new_alleles in zip(alleles, new_alleles):
        site_dict = defaultdict(lambda : last_character)
        for a1, a2 in zip(site_alleles[:n_alleles], site_new_alleles):
            site_dict[a1] = a2
        reduced_alphabet.append(site_dict)
    
    return(reduced_alphabet)


def get_custom_codon_table(aa_mapping):
    '''
    Builds a biopython CodonTable to use for translation with a custom genetic code
    
    
    Parameters
    ----------
    aa_mapping: pd.DataFrame
        pandas DataFrame with columns "Codon" and "Letter" representing the
        genetic code correspondence. Stop codons should appear as "*"
        
    Returns
    -------
    codon_table: Bio.Data.CodonTable.CodonTable object
        Standard bioptython codon table object to use for translating sequences
    
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
