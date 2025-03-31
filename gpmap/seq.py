#!/usr/bin/env python
import itertools
import pandas as pd
import numpy as np

from itertools import chain
from collections import defaultdict

from Bio.Seq import Seq
from Bio.Data.CodonTable import CodonTable
from scipy.sparse import csr_matrix, hstack, vstack

from gpmap.settings import NUCLEOTIDES, COMPLEMENT, ALPHABETS, ALPHABET
from gpmap.utils import check_error
from gpmap.matrix import get_sparse_diag_matrix
from gpmap.settings import DNA_ALPHABET, RNA_ALPHABET, PROTEIN_ALPHABET


def hamming_distance(s1, s2):
    distance = len(s1) - len(s2)
    distance += np.sum([c1 != c2 for c1, c2 in zip(s1, s2)])
    return distance


def extend_ambigous_seq(seq, mapping):
    if not seq:
        yield ("")

    else:
        character, next_seq = seq[0], seq[1:]
        if isinstance(mapping, dict):
            pos_mapping, next_mapping = mapping, mapping
        else:
            pos_mapping, next_mapping = mapping[0], mapping[1:]

        for allele in pos_mapping[character]:
            for seq in extend_ambigous_seq(next_seq, next_mapping):
                yield (allele + seq)


def generate_possible_sequences(seq_length, alphabet=NUCLEOTIDES):
    for seq in itertools.product(alphabet, repeat=seq_length):
        yield ("".join(seq))


def reverse_complement(seq):
    return "".join(COMPLEMENT.get(x, x) for x in seq[::-1])


def get_random_seq(length):
    return "".join(np.random.choice(NUCLEOTIDES, size=length))


def add_random_flanks(seq, length, only_upstream=False):
    if only_upstream:
        flank = get_random_seq(length)
        new_seq = flank + seq
    else:
        flanks = get_random_seq(2 * length)
        new_seq = flanks[:length] + seq + flanks[length:]
    return new_seq


def transcribe_seqs(seqs, code):
    new_seqs = np.array(
        ["".join([sdict[a] for a, sdict in zip(seq, code)]) for seq in seqs]
    )
    return new_seqs


def translate_seqs(seqs, codon_table="Standard"):
    prot_genotypes = np.array(
        [str(Seq(seq).translate(table=codon_table)) for seq in seqs]
    )
    return prot_genotypes


def guess_alphabet_type(alphabet):
    set_alphabet = set(chain(*alphabet))
    if len(set_alphabet - set(DNA_ALPHABET)) == 0:
        alphabet_type = "dna"
    elif len(set_alphabet - set(RNA_ALPHABET)) == 0:
        alphabet_type = "rna"
    elif len(set_alphabet - set(PROTEIN_ALPHABET)) == 0:
        alphabet_type = "protein"
    else:
        alphabet_type = "custom"
    return alphabet_type


def guess_space_configuration(
    seqs,
    ensure_full_space=True,
    force_regular=False,
    force_regular_alleles=False,
):
    """
    Infer the sequence space configuration from a collection of sequences.

    This function determines the sequence space configuration, allowing for
    different numbers of alleles per site while maintaining the order in which
    alleles appear in the sequences. It can also enforce constraints such as
    ensuring a full sequence space or a constant number of alleles per site.

    Parameters
    ----------
    seqs : array-like of shape (n_genotypes,)
        A list or array containing the sequences from which the space
        configuration is to be inferred.

    ensure_full_space : bool, optional, default=True
        If True, ensures that the entire sequence space is represented by
        the provided sequences. This is useful for identifying missing
        genotypes before defining the space.

    force_regular : bool, optional, default=False
        If True, ensures that all sites have the same number of alleles.
        New allele names will be added to sites with fewer alleles than
        the maximum across all sites.

    force_regular_alleles : bool, optional, default=False
        If True, ensures that the same alleles are used across all sites,
        in addition to enforcing the same number of alleles per site.

    Returns
    -------
    config : dict
        A dictionary with the inferred configuration of the sequence space.
        Keys include:
        - 'length': The length of the sequences.
        - 'n_alleles': A list containing the number of alleles per site.
        - 'alphabet': A list of lists, where each inner list contains the
          alleles for a specific site.
        - 'alphabet_type': The inferred type of alphabet ('dna', 'rna',
          'protein', or 'custom').
    """

    alleles = defaultdict(dict)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i][a] = 1
    seq_length = len(alleles)
    n_alleles = [len(alleles[i]) for i in range(seq_length)]
    alphabet = [[a for a in alleles[i].keys()] for i in range(seq_length)]

    if ensure_full_space:
        n_exp = np.prod(n_alleles)
        n_obs = seqs.shape[0]
        msg = "Number of genotypes ({}) does not match ".format(n_obs)
        msg += "the expected from the observed alleles in the "
        msg += "provided sequences ({}).".format(n_exp)
        msg += "Provide phenotypes for every possible "
        msg += "sequence in the space or "
        msg += "set `ensure_full_space=False` to avoid this error"
        check_error(n_exp == n_obs, msg)

    if force_regular:
        if force_regular_alleles:
            new_alphabet = set()
            for alleles in alphabet:
                new_alphabet = new_alphabet.union(alleles)
            n_alleles = [len(new_alphabet)] * seq_length
            alphabet = [sorted(new_alphabet)] * seq_length

        else:
            if np.unique(n_alleles).shape[0] > 1:
                new_alphabet = []
                max_alleles = max(n_alleles)
                for site_alleles in alphabet:
                    i = 0
                    while len(site_alleles) < max_alleles:
                        new_allele = str(i)
                        if new_allele not in site_alleles:
                            site_alleles.append(new_allele)
                        i += 1
                    new_alphabet.append(site_alleles)
                alphabet = [sorted(x) for x in new_alphabet]
                n_alleles = [max_alleles] * seq_length

    config = {
        "length": seq_length,
        "n_alleles": n_alleles,
        "alphabet": alphabet,
    }
    config["alphabet_type"] = guess_alphabet_type(alphabet)

    return config


def generate_freq_reduced_code(
    seqs, n_alleles, counts=None, keep_allele_names=True, last_character="X"
):
    """
    Generate a mapping from each allele in the observed sequences to a reduced
    alphabet with at most ``n_alleles`` per site. The least frequent alleles
    are grouped into a single allele.

    Parameters
    ----------
    seqs : array-like of shape (n_genotypes,) or (n_obs,)
        Observed sequences. If ``counts`` is None, each sequence is assumed to
        appear once. Otherwise, frequencies are calculated using the counts
        as the number of times a sequence appears in the data.

    n_alleles : int or array-like of shape (seq_length,)
        Maximum number of alleles allowed per site. If an array is provided,
        each site will use the specified number of alleles. Otherwise, all
        sites will have the same maximum number of alleles.

    counts : None or array-like of shape (n_genotypes,)
        Number of times each sequence in ``seqs`` appears in the data. If not
        provided, each sequence is assumed to appear exactly once.

    keep_allele_names : bool, optional
        If True, allele names are preserved. Otherwise, they are replaced by
        new alleles taken from the alphabet. Default is True.

    last_character : str, optional
        Character to use for pooled alleles when ``keep_allele_names`` is True.
        Default is "X".

    Returns
    -------
    code : list of dict of length seq_length
        A list of dictionaries, where each dictionary maps the original alleles
        to the new reduced alphabet for each site.
    """

    if counts is None:
        counts = itertools.cycle([1])
    else:
        msg = "counts must have the same shape as seqs"
        check_error(counts.shape == seqs.shape, msg=msg)

    seq_length = len(seqs[0])
    freqs = [defaultdict(lambda: 0) for _ in range(seq_length)]
    for seq, c in zip(seqs, counts):
        for i, allele in enumerate(seq):
            freqs[i][allele] += c

    alleles = [
        sorted(site_freqs.keys(), key=lambda x: site_freqs[x], reverse=True)
        for site_freqs in freqs
    ]

    if keep_allele_names:
        new_alleles = [a[:n_alleles] + [last_character] for a in alleles]
    else:
        new_alleles = [ALPHABET[:n_alleles]] * seq_length

    reduced_alphabet = []
    for site_alleles, site_new_alleles in zip(alleles, new_alleles):
        site_dict = defaultdict(lambda: last_character)
        for a1, a2 in zip(site_alleles[:n_alleles], site_new_alleles):
            site_dict[a1] = a2
        reduced_alphabet.append(site_dict)

    return reduced_alphabet


def get_custom_codon_table(aa_mapping):
    """
    Constructs a Biopython CodonTable for translation using a custom genetic code.

    Parameters
    ----------
    aa_mapping : pd.DataFrame
        A pandas DataFrame with columns "Codon" and "Letter" representing the
        genetic code mapping. Stop codons should be denoted with "*".

    Returns
    -------
    codon_table : Bio.Data.CodonTable.CodonTable
        A Biopython CodonTable object that can be used for translating sequences
        with the specified custom genetic code.
    """
    aa_mapping["Codon"] = [x.replace("U", "T") for x in aa_mapping["Codon"]]
    stop_codons = aa_mapping.loc[aa_mapping["Letter"] == "*", "Codon"].tolist()
    aa_mapping = aa_mapping.loc[aa_mapping["Letter"] != "*", :]
    forward_table = aa_mapping.set_index("Codon")["Letter"].to_dict()
    codon_table = CodonTable(
        forward_table=forward_table,
        stop_codons=stop_codons,
        nucleotide_alphabet="ACGT",
    )
    codon_table.id = -1
    codon_table.names = ["Custom"]
    return codon_table


def get_product_states(state_labels):
    if not state_labels:
        yield ([])
    else:
        for state in state_labels[0]:
            for seq in get_product_states(state_labels[1:]):
                yield ([state] + seq)


def get_seqs_from_alleles(alphabet):
    if not alphabet:
        yield ("")
    else:
        for allele in alphabet[0]:
            for seq in get_seqs_from_alleles(alphabet[1:]):
                yield (allele + seq)


def get_one_hot_from_alleles(alphabet):
    """
    Generate a one-hot encoding CSR matrix for a complete combinatorial space.

    This function uses a fast recursive method to construct the one-hot
    encoding matrix, avoiding redundant computations for common blocks
    in the full matrix.

    Parameters
    ----------
    alphabet : list of list
        A list where each inner list contains the alleles for a specific
        site in the sequence space.

    Returns
    -------
    scipy.sparse.csr_matrix
        A CSR matrix of shape (n_genotypes, total_n_alleles), where
        `n_genotypes` is the total number of genotypes in the sequence
        space and `total_n_alleles` is the sum of alleles across all sites.
        The matrix contains the one-hot encoding of the full sequence space,
        with genotypes sorted lexicographically.
    """

    if not alphabet:
        raise ValueError("alphabet must not be empty")

    n_alleles = len(alphabet[0])
    if len(alphabet) == 1:
        m = get_sparse_diag_matrix(np.ones(n_alleles))
        return m
    else:
        m1 = get_one_hot_from_alleles(alphabet[1:])
        nrows = m1.shape[0]
        row_idxs = np.arange(nrows * n_alleles)
        col_idxs = np.hstack(
            [i * np.ones(nrows, dtype=int) for i in range(n_alleles)]
        )
        data = np.ones(nrows * n_alleles)
        m0 = csr_matrix((data, (row_idxs, col_idxs)))
        m = hstack([m0, vstack([m1] * n_alleles)])
        return m


def get_alphabet(n_alleles=None, alphabet_type=None):
    """
    Generate an alphabet based on the specified number of alleles or alphabet type.

    Parameters
    ----------
    n_alleles : int
        The number of alleles per site. If `alphabet_type` is not specified,
        this determines the size of the custom alphabet.

    alphabet_type : str, optional
        The type of alphabet to use. Must be one of {None, 'dna', 'rna', 'protein'}.
        If None or 'custom', a custom alphabet is generated based on `n_alleles`.

    Returns
    -------
    alphabet : list
        A list containing the alleles in the desired alphabet. For custom alphabets,
        the alleles are represented as strings of numbers or characters.
    """

    if alphabet_type is None or alphabet_type == "custom":
        if n_alleles <= 10:
            alphabet = [str(x) for x in np.arange(n_alleles)]
        else:
            alphabet = [ALPHABET for x in np.arange(n_alleles)]

    elif alphabet_type in ALPHABETS:
        alphabet = ALPHABETS[alphabet_type]

    else:
        raise ValueError(
            "Unknwon alphabet type. Try any of: {}".format(ALPHABETS.keys())
        )
    return alphabet


def get_alleles(c, alleles=None):
    if alleles is not None:
        return alleles
    else:
        return np.unique(c)


def seq_to_one_hot(X, alleles=None):
    m = np.array([[a for a in x] for x in X])
    onehot = []
    for i in range(m.shape[1]):
        c = m[:, i]
        for allele in get_alleles(c, alleles=alleles):
            onehot.append(c == allele)
    onehot = np.staalphabetck(onehot, 1)
    return onehot


def calc_msa_weights(X, phylo_correction=False, max_dist=0.2):
    sl = len(X[0])
    if phylo_correction:
        y = []
        for seq1 in X:
            d = np.array([hamming_distance(seq1, seq2) for seq2 in X]) / sl
            y.append(1 / (d < max_dist).sum())
        y = np.array(y)
    else:
        y = np.ones(X.shape[0])
    return y


def get_subsequences(X, positions=None):
    if positions is None:
        return X
    return np.array(["".join([seq[i] for i in positions]) for seq in X])


def msa_to_counts(
    X, y=None, positions=None, phylo_correction=False, max_dist=0.2
):
    """
    Extracts unique sequences and their counts from a Multiple Sequence
    Alignment (MSA). Optionally, subsequences can be selected based on
    specific positions, and sequence identity re-weighting can be applied
    to account for sequence similarities across the full alignment.

    Parameters
    ----------
    X : array-like of aligned sequences
        Input sequences from which to extract unique sequences and counts.

    y : array-like of weights, optional (default=None)
        Pre-calculated weights associated with the input sequences. If not
        provided, weights are calculated based on sequence identity.

    positions : array-like of int, optional (default=None)
        Subset of positions to extract subsequences from the MSA. If not
        provided, the full sequences are used.

    phylo_correction : bool, optional (default=False)
        If True, applies sequence identity re-weighting. Observations are
        weighted as 1 divided by the number of similar sequences in the MSA.
        Similar sequences are defined based on the `max_dist` parameter.

    max_dist : float, optional (default=0.2)
        Maximum sequence identity distance for considering sequences as
        similar during re-weighting. Only used if `phylo_correction` is True.

    Returns
    -------
    X : np.array of shape (n_unique_seqs,)
        Unique subsequences at the specified positions in the MSA.

    y : np.array of shape (n_unique_seqs,)
        Counts or re-weighted counts for each unique subsequence in the MSA.
    """
    if phylo_correction:
        if not positions:
            raise ValueError(
                '"positions" must be provided for phylogenetic correction'
            )
        if y is not None:
            msg = "phylogenetic correction can not be calculated "
            msg += 'when "y" is provided'
            raise ValueError(msg)

    if y is None:
        y = calc_msa_weights(
            X, phylo_correction=phylo_correction, max_dist=max_dist
        )

    if positions:
        X = get_subsequences(X, positions=positions)

    data = (
        pd.DataFrame({"x": X, "y": y}).groupby(["x"])["y"].sum().reset_index()
    )
    X, counts = data["x"].values, data["y"].values
    return (X, counts)


def calc_allele_frequencies(X, y=None):
    counts = {}

    if y is None:
        y = np.ones(X.shape)

    for x, w in zip(X, y):
        for c in x:
            try:
                counts[c] += w
            except KeyError:
                counts[c] = w

    total = np.sum([v for v in counts.values()])
    freqs = {a: c / total for a, c in counts.items()}
    return freqs


def calc_genetic_code_aa_freqs(codon_table="Standard"):
    dna = get_seqs_from_alleles([DNA_ALPHABET] * 3)
    aa = translate_seqs(dna, codon_table)
    aa = np.array([a for a in aa if a in PROTEIN_ALPHABET])
    return calc_allele_frequencies(aa)


def calc_expected_logp(X, allele_freqs):
    log_freqs = {a: np.log(freq) for a, freq in allele_freqs.items()}
    exp_logp = np.sum([[log_freqs[c] for c in x] for x in X], axis=1)
    return exp_logp
