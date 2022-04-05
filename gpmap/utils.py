import itertools
import sys
import time
import pystan

from time import ctime
from _pickle import dump, load
from os.path import join, exists

import numpy as np

from gpmap.settings import NUCLEOTIDES, COMPLEMENT, MODELS_DIR, CODE_DIR
from Bio.Seq import Seq
from _collections import defaultdict


def logit(p):
    return(np.log(p  /(1 - p)))


def invlogit(x):
    return(np.exp(x) / (1  + np.exp(x)))


class LogTrack(object):
    '''Logger class'''

    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = '[ {} ] {}\n'.format(ctime(), msg)
        else:
            msg += '\n'
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write('Finished succesfully. Time elapsed: {:.1f} s'.format(t))


def write_log(log, msg):
    if log is not None:
        log.write(msg)


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


def write_landscape(landscape_iter, fpath, verbose=False):
    with open(fpath, 'w') as fhand:
        fhand.write('sequence\tdG\tKa\tP_helix\n')
        for i, (seq, dG, Ka, p_helix) in enumerate(landscape_iter):
            fhand.write('\t'.join([seq, str(dG), str(Ka), str(p_helix)]) + '\n')
            if verbose and i % 10000 == 0:
                total_seqs = len(NUCLEOTIDES) ** len(seq)
                print('Sequences processed: {} out of {}'.format(i, total_seqs)) 


def get_constraints_idx(constraints):
    c1, c2 = constraints
    idx1 = [i for i, x in enumerate(c1) if x == '(']
    idx2 = [i for i, x in enumerate(c2[::-1]) if x == ')']
    return(idx1, idx2)


def get_seq(genome, chrom, start, end, strand):
    seq = genome.fetch(chrom, start, end)
    if strand == '-':
        seq = reverse_complement(seq)
    
    return(seq.upper())


def load_pickle(fpath):
    with open(fpath, 'rb') as fhand:
        data = load(fhand)
    return(data)


def write_pickle(data, fpath):
    with open(fpath, 'wb') as fhand:
        dump(data, fhand)
    return(data)


def get_model(model_label, recompile=False):
    code_fpath = join(CODE_DIR, '{}.stan'.format(model_label))
    compiled_fpath = join(MODELS_DIR, '{}.p'.format(model_label))

    if not exists(compiled_fpath) or recompile:
        # STAN model compilation
        model = pystan.StanModel(file=code_fpath)
        write_pickle(model, compiled_fpath)
    else:
        # Load stored model
        model = load_pickle(compiled_fpath)
    return(model)


def translante_seqs(seqs, codon_table='Standard'):
    prot_genotypes = np.array([str(Seq(seq).translate(table=codon_table))
                               for seq in seqs])
    return(prot_genotypes)


def guess_configuration(seqs):
    alleles = defaultdict(set)
    for seq in seqs:
        for i, a in enumerate(seq):
            alleles[i].add(a)
    length = len(alleles)
    config = {'length': length,
              'n_alleles': [len(alleles[i]) for i in range(length)],
              'alphabet': [sorted(alleles[i]) for i in range(length)]}
    if np.prod(config['n_alleles']) != seqs.shape[0]:
        msg = 'Number of genotypes does not match the expected from guessed'
        msg += ' configuration. Ensure that genotypes span the whole sequence '
        msg += 'space'
        raise ValueError(msg)
    return(config)
