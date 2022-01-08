import itertools

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.linalg.decomp_cholesky import cholesky
from scipy.special._logsumexp import logsumexp
from scipy.stats.stats import pearsonr

from gpmap.base import BaseGPMap
from gpmap.utils import get_model
from gpmap.plot_utils import arrange_plot, savefig, init_fig
import logomaker


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
    
    def embed_seqs(self, seqs, upstream='', downstream=''):
        if not upstream and not downstream:
            return(seqs)
        embedded = []
        for seq in seqs:
            embedded.append(upstream + seq + downstream)
        return(embedded)
    
    def add_flanking_seqs(self, seqs, n_backgrounds=1):
        upstream = self.simulate_random_seqs(self.filter_size - 1, n_seqs=n_backgrounds)
        downstream = self.simulate_random_seqs(self.filter_size - 1, n_seqs=n_backgrounds)
        embedded = []
        for u, d in zip(upstream, downstream):
            embedded.extend(self.embed_seqs(seqs, u, d))
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
                            position_variable=False, position_seq_variable=False):
        pos = np.arange(n_positions_filter)
        
        if position_variable:
            mu = theta0 - rho * np.abs(pos - pos.mean())    
        else:
            mu = np.full(n_positions_filter, theta0)
            
        theta = np.random.normal(0, 1, size=n_features)
        theta = np.vstack([theta] * n_positions_filter).T
        
        if position_seq_variable:
            a, r = 0.5, 1
            L_K = self.get_L_K(pos, a, r)
            theta += np.dot(L_K, np.random.normal(0, 1, size=theta.shape).T).T
            
        return(mu, theta)
    
    def calc_total_protein(self, encoding, mu, theta, theta0=0, background=0):
        log_ki = np.vstack([-(m + np.dot(features, t))
                            for features, t, m in zip(encoding, theta.T, mu)])
        log_ki_sum = logsumexp(log_ki, axis=0)
        logf = theta0 + log_ki_sum
        
        if background > 0:
            logf = logsumexp(np.vstack([np.full(log_ki_sum.shape, np.log(background)), logf]), axis=0)
        return(logf)
    
    def calc_translation_rate(self, encoding, mu, theta, theta0=0, background=0):
        log_ki = np.vstack([-(m + np.dot(features, t))
                            for features, t, m in zip(encoding, theta.T, mu)])
        log_ki_sum = logsumexp(log_ki, axis=0)
        v = np.vstack([np.ones(log_ki_sum.shape), log_ki_sum])
        
        logf = theta0 + log_ki_sum - logsumexp(v, axis=0)
        if background > 0:
            logf = logsumexp(np.vstack([np.full(log_ki_sum.shape, np.log(background)), logf]), axis=0)
        return(logf)
    
    def to_stan_data(self, seqs, y, ref_seq=None, encoding=None,
                     upstream='', downstream='', y_sd=None):
        seqs = self.embed_seqs(seqs, upstream=upstream, downstream=downstream)
        if encoding is None:
            encoding = self.get_conv_encoding(seqs, ref_seq=ref_seq)
        n_features = encoding[0].shape[1]
        n_positions_filter = len(encoding)
        
        data = {'x': seqs, 'y': y,
                'encoding': encoding,
                'theta_labels': encoding[0].columns,
                'L': len(seqs[0]), 'F': n_features, 
                'C': self.n_alleles, 'S': n_positions_filter}
        if y_sd is not None:
            data['log_gfp_sd'] = y_sd
        return(data)
    
    def simulate_data(self, seqs, ref_seq=None, log_rt=0, background=0,
                      theta0=2, rho=0.5, position_variable=False,
                      position_seq_variable=False, sigma=0.2):
        if ref_seq is None:
            ref_seq = self.simulate_random_seqs(self.filter_size, n_seqs=1)[0]
            
        encoding = self.get_conv_encoding(seqs, ref_seq)
        n_positions_filter, n_features = len(encoding), encoding[0].shape[1]
        
        mu, theta = self.simulate_parameters(n_positions_filter, n_features,
                                             theta0=theta0, rho=rho,
                                             position_variable=position_variable,
                                             position_seq_variable=position_seq_variable)
        logf = self.calc_total_protein(encoding, mu, theta, log_rt, background)
        y = logf + np.random.normal(0, sigma)
        data = self.to_stan_data(seqs, y, ref_seq, encoding=encoding)
        data['yhat'] = logf
        data['theta'] = theta
        
        return(data)
    
    def fit(self, data):
        X = np.stack(data['encoding'], axis=2).transpose([2, 0, 1])
        stan_data = {'G': len(data['x']), 'F': data['F'], 'S': data['S'],
                     'X': X, 'log_gfp': data['y']}
        if 'log_gfp_sd' in data:
            stan_data['log_gfp_sd'] = data['log_gfp_sd']
            
        self.estimates = self.model.optimizing(stan_data)
        theta = self.estimates['theta']
        sigma = self.estimates['sigma']
        yhat = self.estimates['yhat']
        mu = self.estimates['mu']
        labels = data['encoding'][0].columns
        df = pd.DataFrame({'theta': theta, 'label': labels})
        results = {'theta': theta, 'sigma': sigma, 'yhat': yhat,
                   'mu': mu, 'theta_labels': labels,
                   'df': df, 'y': data['y']}
        
        if 'background' in self.estimates:
            results['background'] = self.estimates['background']
            
        return(results)
    
    def plot_y_distribution(self, data, axes, xlabel='Phenotype', islog=False):
        y = data['y']
        if islog:
            y = np.exp(y)
        sns.histplot(y, ax=axes)
        arrange_plot(axes, xlabel=xlabel, ylabel='# genotypes',
                     title='Phenotype distribution')
    
    def plot_mut_eff(self, data, fit, axes):
        x, y = data['theta'][1:], fit['theta'][1:]
        if x.shape != y.shape:
            try:
                x = np.vstack([x] * y.shape[1]).transpose()
            except IndexError:
                y = np.vstack([y] * x.shape[1]).transpose()
        
        axes.scatter(x, y, s=10, lw=0.2, edgecolor='black')
        xlims, ylims = axes.get_xlim(), axes.get_ylim() 
        lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
        axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--')
        arrange_plot(axes, xlabel=r'$\theta_{real}$',
                     ylabel=r'$\hat\theta$', xlims=lims, ylims=lims,
                     title='Mutational effects')
    
    def theta_to_matrix(self, fit):
        df = fit['df']
        df['pos'] = [int(x[1:-1]) for x in df['label']]
        df['letter'] = [x[-1] for x in df['label']]
        m = pd.pivot_table(df, index='letter', columns='pos',
                           values='theta', fill_value=0)
        return(m)
    
    def plot_theta_heatmap(self, fit, axes, label=r'$\theta$'):
        m = self.theta_to_matrix(fit)
        sns.heatmap(m, cmap='coolwarm', ax=axes, center=0,
                    cbar_kws={'label': label})
        arrange_plot(axes, ylabel='Allele', xlabel='Position')
    
    def plot_theta_logo(self, fit, axes, label=r'$\theta$'):
        m = self.theta_to_matrix(fit)
        m = m - m.mean()
        logomaker.Logo(m.T, ax=axes)
        arrange_plot(axes, ylabel=label)
    
    def plot_theta_barplot(self, fit, axes, label=r'$\theta$'):
        df = fit['df']
        sns.barplot(x='label', y='theta', data=df, ax=axes, linewidth=1, 
                    edgecolor='black', color='purple')
        arrange_plot(axes, ylabel=label, xlabel='Feature', rotate_xlabels=True)
    
    def plot_mu(self, fit, axes, color='purple', label=None, x=None):
        mu = fit['mu']
        if not isinstance(mu, float):
            if x is None:
                x = np.arange(mu.shape[0])
            axes.plot(x, mu, lw=1, c=color)
            axes.scatter(x, mu, lw=1, c=color, label=label)
            arrange_plot(axes, xlabel='Position', ylabel=r'$\mu$')
    
    def plot_predictions(self, fit, axes, hist=False):
        if hist:
            sns.histplot(x=fit['yhat'], y=fit['y'], cmap='viridis', ax=axes,
                         cbar_kws={'label': '# genotypes'}, cbar=True, vmax=200)
        else:
            axes.scatter(fit['yhat'], fit['y'],
                         s=10, lw=0.2, edgecolor='black')
        r = pearsonr(fit['yhat'], fit['y'])[0]
        xlims, ylims = axes.get_xlim(), axes.get_ylim() 
        lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
        axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--')
        axes.text(xlims[0] + 0.1 * (xlims[1] - xlims[0]),
                  ylims[0] + 0.9 * (ylims[1] - ylims[0]),
                  'r={:.2f}'.format(r))
        arrange_plot(axes, xlabel=r'$\hat{y}$',
                     ylabel=r'$y$', xlims=lims, ylims=lims,
                     title='Phenotypic values')
    
    def figure_data_distribution(self, data, fname, xlabel='Phenotype'):
        fig, subplots = init_fig(1, 2)
        axes = subplots[0]
        self.plot_y_distribution(data, axes, xlabel=xlabel)
        axes = subplots[1]
        self.plot_y_distribution(data, axes, xlabel=xlabel, islog=True)
        savefig(fig, fname)
    

class AdditiveConvolutionalModel(ConvolutionalModel):
    def __init__(self, filter_size, alphabet_type='rna',
                 n_alleles=4, model_label='conv0', recompile=False):
        self.set_parameters(filter_size=filter_size, alphabet_type=alphabet_type,
                            n_alleles=n_alleles, 
                            model_label=model_label, recompile=recompile)
    
    def seq_to_encoding(self, seq, ref_seq):
        features = {}
        for i in range(min(len(seq), self.filter_size)):
            for nc in self.alphabet:
                if nc != ref_seq[i]:
                    features['{}{}{}'.format(ref_seq[i], i+1, nc)] = int(seq[i] == nc)
        return(features)
    
    def get_encoding(self, seqs, ref_seq=None, frame=0):
        full = pd.DataFrame([self.seq_to_encoding(seq[frame:frame+self.filter_size],
                                                  ref_seq=ref_seq)
                             for seq in seqs], index=seqs)
        sorted_cols = sorted(full.columns[1:],
                             key=lambda x:(int(x[1:-1]), x[0], x[-1]))
        cols = [full.columns[0]] + sorted_cols
        full = full[cols]
        return(full)
    
    def get_conv_encoding(self, seqs, ref_seq):
        length = len(seqs[0])
        n_positions_filter = length - self.filter_size + 1
        positions = np.arange(n_positions_filter)
        encoding = [self.get_encoding(seqs, ref_seq=ref_seq, frame=i) for i in positions]
        return(encoding)
    
    
class BPStacksConvolutionalModel(ConvolutionalModel):
    def __init__(self, template, allow_bulges=False,
                 model_label='conv0', recompile=False):
        filter_size = len(template)
        self.set_parameters(filter_size=filter_size, alphabet_type='rna',
                            model_label=model_label, recompile=recompile)
        self.allow_bulges = allow_bulges
        self.set_template_seq(template)
        
    def set_template_seq(self, template):
        if self.alphabet_type != 'rna': 
            raise ValueError('Template sequence can only be specified for RNA')

        dinucleotides = [template[i:i+2] for i in range(len(template)-1)]
        stacks = {}
        for dinc in dinucleotides:
            c = []
            for b1 in self.complements[dinc[0]]:
                for b2 in self.complements[dinc[1]]:
                    c.append(b1 + b2)
                    stacks['{}|{}'.format(dinc, c[-1])] = 0
        self.stacks = stacks
        self.template = template
        
    def seq_to_encoding(self, seq, bulge_pos=None):
        if bulge_pos is None:
            target = seq
            bulge = None
        else:
            bulge = seq[bulge_pos]
            target = seq[:bulge_pos] + seq[bulge_pos+1:]
            
        counts = self.stacks.copy()
        for i in range(self.filter_size - 1):
            stack = '{}|{}'.format(self.template[i:i+2], target[i:i+2])
            try:
                counts[stack] += 1
            except KeyError:
                continue
        if bulge is not None:
            counts['b{}'.format(bulge)] = 1
        return(counts)
    
    def get_encoding(self, seqs, frame=0, bulge_pos=None, **kwargs):
        full = pd.DataFrame([self.seq_to_encoding(seq[frame:frame+self.filter_size], bulge_pos=bulge_pos)
                             for seq in seqs], index=seqs)
        return(full)
    
    def get_possible_positions(self, seqs, bulge=False):
        length = len(seqs[0])
        n_positions_filter = length - self.filter_size + 1 - int(bulge)
        positions = np.arange(n_positions_filter)
        return(positions)
    
    def get_possible_configurations(self, seqs):
        # Bulges need at least to bases at each side to allow counting some stacks
        # Limits also the number of useful configurations to take into account
        bulge_positions = np.array([None])
        if self.allow_bulges:
            bulge_positions = np.hstack([bulge_positions,
                                         np.arange(2, self.filter_size-1)])
        
        for bulge_pos in bulge_positions: 
            for pos in self.get_possible_positions(seqs, bulge=bulge_pos is not None):
                yield(pos, bulge_pos)

    def get_conv_encoding(self, seqs):
        encoding = [self.get_encoding(seqs, frame=frame, bulge_pos=bulge_pos)
                    for frame, bulge_pos in self.get_possible_configurations(seqs)]
        return(encoding)
