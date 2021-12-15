import itertools

import numpy as np
import pandas as pd

from scipy.linalg.decomp_cholesky import cholesky
from scipy.special._logsumexp import logsumexp

from gpmap.base import BaseGPMap
from gpmap.utils import get_model


class ConvolutionalModel(BaseGPMap):
    def set_parameters(self, filter_size, alphabet_type='rna',
                       n_alleles=4, model_label='conv0', recompile=False):
        self.filter_size = filter_size
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles)
        self.model = get_model(model_label, recompile=recompile)
            
    def simulate_random_seqs(self, length, n_seqs):
        if length is None or n_seqs is None:
            raise ValueError('length and n_seqs must be specified')
        
        total = length * n_seqs
        letters = np.random.choice(self.alphabet, size=total, replace=True)
        x = [''.join(x) for x in letters.reshape(n_seqs, length)]
        return(x)
    
    def get_single_mutants(self, seq):
        for i in range(len(seq)):
            for nc in self.alphabet:
                if nc == seq[i]:
                    continue
                yield(seq[:i] + nc + seq[i+1:])
    
    def get_single_and_double_mutants(self, seq):
        if seq is None:
            raise ValueError('seq argument must be specified')
        
        singles = list(self.get_single_mutants(seq))
        mutants = set([seq] + singles)
        for s in singles:
            for double in self.get_single_mutants(s):
                mutants.add(double)
        return(mutants)
    
    def get_random_mutation(self, letter):
        new_letter = np.random.choice([x for x in self.alphabet if x != letter])
        return(new_letter)
    
    def get_random_mutant(self, seq, mut_positions):
        new_seq = ''.join([self.get_random_mutation(x) if pos in mut_positions else x
                           for pos, x in enumerate(seq)])
        return(new_seq)
    
    def simulate_random_mutants(self, seq, n_seqs, p_mut):
        if seq is None or n_seqs is None or p_mut is None:
            raise ValueError('seq, n_seqs and p_mut must all be specified')
        
        length = len(seq)
        positions = np.arange(length)
        for _ in range(n_seqs):
            
            n_mut = np.random.binomial(length, p_mut)
            while n_mut == 0:
                n_mut = np.random.binomial(length, p_mut)
                
            mut_positions = np.random.choice(positions, size=n_mut, replace=False)
            mutant = self.get_random_mutant(seq, mut_positions)
            yield(mutant)
    
    def get_all_possible_seqs(self, length):
        for seq in itertools.product(self.alphabet, repeat=length):
            yield(''.join(seq))
    
    def simulate_combinatorial_mutants(self, length, p=1):
        if length is None:
            raise ValueError('sequence "length" must be specified')
            
        seqs = np.array(list(self.get_all_possible_seqs(length)))
        n_seqs = seqs.shape[0]
        if p < 1:
            seqs = seqs[np.random.uniform(size=n_seqs) < p]
        return(seqs)

    def simulate_sequences(self, length=None, seq=None, mode='random',
                           n_seqs=None, p=1, p_mut=None):
        if mode == 'random': 
            seqs = self.simulate_random_seqs(length, n_seqs)
        elif mode == 'single_and_doubles':
            seqs = self.get_single_and_double_mutants(seq)
        elif mode == 'combinatorial':
            seqs = self.simulate_combinatorial_mutants(length, p)
        elif mode == 'error_pcr':
            seqs = self.simulate_random_mutants(seq, n_seqs, p_mut)
        else:
            raise ValueError('mode {} is not compatible'.format(mode))
        return(seqs)
    
    def add_flanking_seqs(self, seqs, n_backgrounds=1):
        upstream = self.simulate_random_seqs(self.filter_size - 1, n_seqs=n_backgrounds)
        downstream = self.simulate_random_seqs(self.filter_size - 1, n_seqs=n_backgrounds)
        embedded = []
        for u, d in zip(upstream, downstream):
            for seq in seqs:
                embedded.append(u + seq + d)
        return(embedded)
    
    def get_L_K(self, x, a, r):
        d = []
        for x1 in x:
            row = []
            for x2 in x:
                row.append(np.abs(x1 - x2))
            d.append(row)
        d = np.array(d)
        K = a * np.exp(-d**2 / r)
        L = cholesky(K + 1e-5 * np.eye(K.shape[0]))
        return(L)
    
    def simulate_parameters(self, n_positions_filter, n_features, theta0, rho,
                            position_variable=False):
        pos = np.arange(n_positions_filter)
        mu = theta0 - rho * np.abs(pos - pos.mean())    
        theta = np.random.normal(0, 1, size=n_features)
        theta = np.vstack([theta] * n_positions_filter).T
        theta[0] = mu
        
        if position_variable:
            a, r = 0.5, 1
            L_K = self.get_L_K(pos, a, r)
            theta[1:] += np.dot(L_K, np.random.normal(0, 1, size=theta[1:].shape).T).T
            
        return(theta)
    
    def calc_logf(self, encoding, theta, log_rt=0, background=0):
        log_ki = np.vstack([np.dot(features, t)
                            for features, t in zip(encoding, theta.T)])
        log_ki_sum = logsumexp(log_ki, axis=0)
        v = np.vstack([np.ones(log_ki_sum.shape), log_ki_sum])
        
        logf = log_rt + log_ki_sum - logsumexp(v, axis=0)
        if background > 0:
            logf = logsumexp(np.vstack([np.full(log_ki_sum.shape, np.log(background)), logf]), axis=0)
        return(logf)
    
    def get_conv_encoding(self, seqs, ref_seq):
        length = len(seqs[0])
        n_positions_filter = length - self.filter_size + 1
        positions = np.arange(n_positions_filter)
        encoding = [self.get_encoding(seqs, ref_seq, frame=i) for i in positions]
        return(encoding)
    
    def simulate_data(self, seqs, ref_seq=None, log_rt=0, background=0,
                      theta0=0, rho=0.5, position_variable=False, sigma=0.2):
        if ref_seq is None:
            ref_seq = self.simulate_random_seqs(self.filter_size, n_seqs=1)[0]
            
        encoding = self.get_conv_encoding(seqs, ref_seq)
        n_positions_filter, n_features = len(encoding), encoding[0].shape[1]
        
        theta = self.simulate_parameters(n_positions_filter, n_features,
                                         theta0=theta0, rho=rho,
                                         position_variable=position_variable)
        logf = self.calc_logf(encoding, theta, log_rt, background)
        y = logf + np.random.normal(0, sigma)
        
        data = {'x': seqs, 'y': y, 'yhat': logf,
                'encoding': encoding,  'theta': theta,
                'ref_seq': ref_seq,
                
                'L': len(seqs[0]), 'F': n_features, 
                'C': self.n_alleles, 'S': n_positions_filter
                }
        return(data)
    
    def fit(self, data):
        X = np.stack(data['encoding'], axis=2).transpose([2, 0, 1])
        
        data = {'G': len(data['x']), 'F': data['F'], 'S': data['S'],
                'X': X, 'log_gfp': data['y']}
        self.estimates = self.model.optimizing(data)
        theta = self.estimates['theta']
        sigma = self.estimates['sigma']
        yhat = self.estimates['yhat']
        results = {'theta': theta, 'sigma': sigma, 'yhat': yhat}
        return(results)
        

class AdditiveConvolutionalModel(ConvolutionalModel):
    def __init__(self, filter_size, alphabet_type='rna',
                 n_alleles=4):
        self.set_parameters(filter_size=filter_size, alphabet_type=alphabet_type,
                            n_alleles=n_alleles)
    
    def seq_to_encoding(self, seq, wt):
        features = {wt: 1}
        for i in range(min(len(seq), len(wt))):
            for nc in self.alphabet:
                if nc != wt[i]:
                    features['{}{}{}'.format(wt[i], i+1, nc)] = int(seq[i] == nc)
        return(features)
    
    def get_encoding(self, seqs, wt, frame=0):
        l = len(wt)
        full = pd.DataFrame([self.seq_to_encoding(seq[frame:frame+l], wt)
                             for seq in seqs], index=seqs)
        sorted_cols = sorted(full.columns[1:],
                             key=lambda x:(int(x[1:-1]), x[0], x[-1]))
        cols = [full.columns[0]] + sorted_cols
        full = full[cols]
        return(full)
    
    
class BPStacksConvolutionalModel(ConvolutionalModel):
    def __init__(self, filter_size, template,
                 model_label='conv0', recompile=False):
        self.set_parameters(filter_size=filter_size, alphabet_type='rna',
                            model_label=model_label, recompile=recompile)
        self.set_template_seq(template)
        
    def set_template_seq(self, template):
        if self.alphabet_type != 'rna': 
            raise ValueError('Template sequence can only be specified for RNA')

        dinucleotides = [template[i:i+2] for i in range(len(template)-1)]
        stacks = {'baseline': 1}
        for dinc in dinucleotides:
            c = []
            for b1 in self.complements[dinc[0]]:
                for b2 in self.complements[dinc[1]]:
                    c.append(b1 + b2)
                    stacks['{}|{}'.format(dinc, c[-1])] = 0
        self.stacks = stacks
        
    def seq_to_encoding(self, seq, template):
        counts = self.stacks.copy()
        for i in range(len(template)-1):
            stack = '{}|{}'.format(template[i:i+2], seq[i:i+2])
            try:
                counts[stack] += 1
            except KeyError:
                continue
        return(counts)
    
    def get_encoding(self, seqs, template, frame=0):
        self.set_template_seq(template)
        l = len(template)
        full = pd.DataFrame([self.seq_to_encoding(seq[frame:frame+l], template)
                             for seq in seqs], index=seqs)
        return(full)
