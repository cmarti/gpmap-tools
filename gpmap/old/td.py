import itertools

import logomaker
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.special._logsumexp import logsumexp
from scipy.stats.stats import pearsonr

from gpmap.base import BaseGPMap
from gpmap.utils import get_model
from gpmap.plot_utils import arrange_plot, savefig, init_fig
from scipy.stats._continuous_distns import norm


class _ConvolutionalModel(BaseGPMap):
    '''
    This class has only methods for simulating sequences for experimental data
    using different schemes
    '''
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
    
    def add_flanking_seqs(self, seqs, n_backgrounds=1, flank_size=None):
        if flank_size is None:
            flank_size = self.filter_size - 1
        upstream = self.simulate_random_seqs(flank_size, n_seqs=n_backgrounds)
        downstream = self.simulate_random_seqs(flank_size, n_seqs=n_backgrounds)
        embedded = []
        for u, d in zip(upstream, downstream):
            embedded.extend(self.embed_seqs(seqs, u, d))
        return(embedded)
    
    
class ConvolutionalModel(_ConvolutionalModel):
    def get_model_label(self):
        model_label = None
        if self.multiple_backgrounds:
            if self.variable_theta:
                model_label = 'full_model'
            elif self.fixed_theta:
                model_label = 'full_model.fixed_theta.gpmu'
            else:
                if self.positional_effects:
                    if self.allow_bulges:
                        model_label = 'full_model.fixed.bulges'
                    else:
                        if self.n_filters > 1:
                            model_label = 'full_model.fixed.n_filters'
                        else:
                            model_label = 'full_model.fixed.gpmu'
                else:
                    model_label = 'full_model.fixed_pos'
        else:
            if self.positional_effects:
                if self.alpha_type == 1:
                    model_label = 'conv_pos_eff_fixed_alpha'
                elif self.alpha_type == 0:
                    model_label = 'conv_pos_eff'
                elif self.alpha_type == 'free':
                    if self.intercept is not None and self.slope is not None:
                        model_label = 'conv_pos_eff_free_alpha_known_cts'
                    else:
                        model_label = 'conv_pos_eff_free_alpha'
                else:
                    raise ValueError('alpha_type {} not allowed'.format(self.alpha_type))
                
            else:
                if self.alpha_type == 1:
                    model_label = 'conv_fixed_fixed_alpha'
                elif self.alpha_type == 0:
                    model_label = 'conv_fixed'
                elif self.alpha_type == 'free':
                    if self.intercept is not None and self.slope is not None:
                        model_label = 'conv_fixed_free_alpha_known_cts'
                    else:
                        model_label = 'conv_fixed_free_alpha'
                else:
                    raise ValueError('alpha_type {} not allowed'.format(self.alpha_type))
        print(model_label)
        if model_label is None:
            raise ValueError('Conditions not available')
        
        return(model_label)
        
    def set_parameters(self, filter_size, alphabet_type='rna',
                       n_filters=1, positional_effects=False,
                       allow_bulges=False, base_bulges=False,
                       intercept=None, slope=None,
                       fixed_theta=False,
                       position_bulges=False,
                       selected_bulges_positions=None,
                       multiple_backgrounds=False,
                       variable_theta=False,
                       alpha_type=0, n_alleles=4, recompile=False):
        self.set_alphabet_type(alphabet_type, n_alleles=n_alleles)
        
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.alpha_type = alpha_type
        self.multiple_backgrounds = multiple_backgrounds
        self.variable_theta = variable_theta
        self.intercept = intercept
        self.slope = slope
        self.positional_effects = positional_effects
        self.fixed_theta = fixed_theta
        
        self.allow_bulges = allow_bulges
        self.base_bulges = base_bulges
        self.position_bulges = position_bulges
        self.selected_bulges_positions = selected_bulges_positions
        self.set_bulges_params()
        
        self.model = get_model(self.get_model_label(), recompile=recompile)
        self.params = ['mu', 'theta', 'sigma', 'yhat', 'log_ki', 'background',
                       'theta_0', 'log_alpha', 'alpha']
    
    def get_bulges_positions(self):
        if self.selected_bulges_positions is None:
            selected_bulges_positions = np.arange(2, self.filter_size - 1)
        else:
            selected_bulges_positions = self.selected_bulges_positions
        return(selected_bulges_positions)
    
    def get_bulge_nc_and_pos(self):
        for p in self.get_bulges_positions():
            for a in self.alphabet:
                yield(p, a)
    
    def set_bulges_params(self):
        if self.allow_bulges:
            if self.base_bulges:
                if self.position_bulges:
                    self.bulges_params = {'b{}{}'.format(p, a): 0
                                          for p, a in self.get_bulge_nc_and_pos()}
                else:
                    self.bulges_params = {'b{}'.format(a): 0
                                          for a in self.alphabet}
            else:
                if self.position_bulges:
                    self.bulges_params = {'b{}'.format(a): 0
                                          for a in self.get_bulges_positions()}
                else:
                    self.bulges_params = {'b': 0}
        else:
            self.bulges_params = {}
            
    def update_bulge_features(self, features, bulge, bulge_pos):
        if bulge is not None:
            if self.base_bulges:
                if self.position_bulges:
                    features['b{}{}'.format(bulge_pos, bulge)] = 1
                else:
                    features['b{}'.format(bulge)] = 1
            else:
                if self.position_bulges:
                    features['b{}'.format(bulge_pos)] = 1
                else:
                    features['b'] = 1
    
    def get_target_and_bulge(self, seq, bulge_pos=None):
        if bulge_pos is None:
            target = seq
            bulge = None
        else:
            bulge = seq[bulge_pos]
            target = seq[:bulge_pos] + seq[bulge_pos+1:]
        return(target, bulge)
    
    def get_encoding(self, seqs, frame=0, bulge_pos=None):
        start = frame
        end = start + self.filter_size + int(bulge_pos is not None)
        full = pd.DataFrame([self.seq_to_encoding(seq[start:end], bulge_pos=bulge_pos)
                             for seq in seqs], index=seqs)
        if hasattr(self, 'feature_names'):
            full = full[self.feature_names]
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
                                         self.get_bulges_positions()])
        
        for bulge_pos in bulge_positions: 
            for pos in self.get_possible_positions(seqs, bulge=bulge_pos is not None):
                yield(pos, bulge_pos)

    def get_conv_encoding(self, seqs):
        configurations = list(self.get_possible_configurations(seqs))
        encoding = [self.get_encoding(seqs, frame=frame, bulge_pos=bulge_pos)
                    for frame, bulge_pos in configurations]
        frames = np.array([frame for frame, bulge_pos in configurations])
        
        encoding = self.fix_encoding(encoding)
        features = encoding[0].columns
        
        n_positions = np.unique(frames).shape[0]
        if not self.positional_effects:
            frames = np.full(frames.shape, 0)
        
        return({'X': encoding, 'positions': frames,
                'n_positions': n_positions,
                'n_configurations': len(configurations),
                'features': features, 'n_features': features.shape[0]})
    
    def fix_encoding(self, encoding):
        features = np.unique(np.hstack([x.columns for x in encoding]))
        xs = []
        for x in encoding:
            for feature in features:
                if feature not in x.columns:
                    x[feature] = 0
            xs.append(x[features].fillna(0).astype(int))
        return(xs)
    
    def simulate_parameters(self, encoding, mu_0, rho):
        n_positions_filter = encoding['n_positions']
        n_features = encoding['n_features']
        
        theta_0 = np.random.normal(2, 1)
        theta = np.random.normal(0, 1, size=n_features)
        alpha = np.exp(np.random.normal(-2, 1)) if self.alpha_type == 'free' else self.alpha_type
        if self.positional_effects:
            pos = np.arange(n_positions_filter)
            mu = mu_0 - rho * np.abs(pos - pos.mean())    
        else:
            mu = np.array([mu_0])
        return({'mu': mu, 'theta': theta, 'theta_0': theta_0, 'alpha': alpha})
    
    def calc_total_log_protein(self, encoding, params):
        if self.n_filters > 1:
            log_ki = np.vstack([logsumexp(-(params['mu'][p] + np.dot(f, params['theta'])), axis=1)
                                for f, p in zip(encoding['X'], encoding['positions'])])
        else:
            log_ki = np.vstack([-(params['mu'][p] + np.dot(f, params['theta']))
                                for f, p in zip(encoding['X'], encoding['positions'])])
            
        log_ki_sum = logsumexp(log_ki, axis=0)
        
        if self.alpha_type != 0:
            ki_sum = np.exp(log_ki_sum)
            yhat = params['theta_0'] + log_ki_sum - np.log(1 + params['alpha'] * ki_sum)
        else:
            yhat = log_ki_sum
        
        if params['background'] > 0:
            background = np.full(log_ki_sum.shape, np.log(params['background']))
            yhat = logsumexp(np.vstack([background, yhat]), axis=0)
            
        return({'yhat': yhat, 'log_ki': log_ki})
    
    def to_stan_data(self, seqs, y, encoding=None, y_sd=None,
                     upstream='', downstream=''):
        seqs = self.embed_seqs(seqs, upstream=upstream, downstream=downstream)
        if encoding is None:
            encoding = self.get_conv_encoding(seqs)
        else:
            encoding = encoding
            
        bulge_idx = np.array([i+1 for i, f in enumerate(encoding['features'])
                              if f.startswith('b')]).astype(int)
        mut_idx = np.array([i+1 for i, f in enumerate(encoding['features'])
                            if not f.startswith('b')]).astype(int)
            
        X = np.stack(encoding['X'], axis=2).transpose([2, 0, 1])
        data = {'seqs': seqs, 'y': y, 'X': X,
                'theta_labels': encoding['features'],
                'positions': encoding['positions'] + 1,
                'distances': encoding['positions'] + 1,
                
                'bulges_features': bulge_idx,
                'mut_features': mut_idx,
                
                'L': len(seqs[0]), 'F': encoding['n_features'], 
                'C': self.n_alleles, 'S': encoding['n_configurations'],
                'P': encoding['n_positions'], 'G': len(seqs),
                'N': self.n_filters, 'B': bulge_idx.shape[0]}
        
        if y_sd is not None:
            data['y_sd'] = y_sd
        if self.alpha_type == 1:
            data['alpha'] = 1
        
        if self.intercept is not None and self.slope is not None:
            data['intercept'] = self.intercept
            data['slope'] = self.slope
            
        return(data)

    def join_data(self, data_dict):
        X = []
        y = []
        
        for bc_data in data_dict.values():
            X.append(bc_data['X'])
            y.append(bc_data['y'])
        
        X = np.stack(X, axis=3).transpose([3, 0, 1, 2])
        y = np.vstack(y).T
        print(X.shape, y.shape)
        data = bc_data.copy()
        data['X'] = X
        data['y'] = y
        data['C'] = len(data_dict)
        data['distances'] = bc_data['positions']
        return(data)
    
    def simulate_data(self, seqs, theta_0=None, background=0,
                      mu_0=2, rho=0.5, sigma=0.2):
        encoding = self.get_conv_encoding(seqs)
        params = self.simulate_parameters(encoding, mu_0=mu_0, rho=rho)
        params['background'] = background
        if theta_0 is not None:
            params['theta_0'] = theta_0
        
        output = self.calc_total_log_protein(encoding, params)
        y = np.random.normal(output['yhat'], sigma)
        
        data = self.to_stan_data(seqs, y, encoding=encoding)
        data['yhat'] = output['yhat']
        data['theta'] = params['theta']
        data['mu'] = params['mu']
        data['background'] = background
        
        return(data)
    
    def fit(self, data, n_iter=2000):
        input_data = {k: v for k, v in data.items()
                      if k not in ['seqs', 'theta_labels']}
        results = self.model.optimizing(input_data, iter=n_iter)
        # results = {param: estimates[param] for param in self.params
        #            if param in estimates}
        # df = pd.DataFrame({'theta': results['theta'],
        #                    'label': data['theta_labels']})
        loglikelihood = norm.logpdf(data['y'], results['yhat'], results['sigma']).sum()
        results.update({'theta_labels': data['theta_labels'],
                        # 'df': df,
                        'y': data['y'], 'seqs': data['seqs'],
                        'loglikelihood': loglikelihood})

        if not self.positional_effects:
            results['mu'] = np.array([results['mu']])
        if 'log_alpha' in  results:
            results['alpha'] = np.exp(results['log_alpha'])
            
        return(results)
    
    def predict(self, seqs, fit):
        encoding = self.get_conv_encoding(seqs)
        output = self.calc_total_log_protein(encoding, fit)
        return(output)
    
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
    
    def theta_to_matrix(self, fit, bulges=False):
        df = fit['df']
        rows = np.array([x.startswith('b') == bulges for x in df['label']])
        df = df.loc[rows, :]
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
    
    def plot_predictions(self, fit, axes, hist=False, vmax=None):
        if hist:
            sns.histplot(x=fit['yhat'], y=fit['y'], cmap='viridis', ax=axes,
                         cbar_kws={'label': '# genotypes'},
                         cbar=True, vmax=vmax)
        else:
            axes.scatter(fit['yhat'], fit['y'],
                         s=10, lw=0.2, edgecolor='black')
        r = pearsonr(fit['yhat'], fit['y'])[0]
        xlims, ylims = axes.get_xlim(), axes.get_ylim() 
        lims = min(xlims[0], ylims[0]), max(xlims[1], ylims[1])
        axes.plot(lims, lims, lw=0.5, c='grey', linestyle='--')
        arrange_plot(axes, xlabel=r'$\hat{y}$',
                     ylabel=r'$y$', xlims=lims, ylims=lims,
                     title='Phenotypic values')
        axes.text(xlims[0] + 0.1 * (xlims[1] - xlims[0]),
                  ylims[0] + 0.9 * (ylims[1] - ylims[0]),
                  'r={:.2f}'.format(r))
    
    def figure_data_distribution(self, data, fname, xlabel='Phenotype'):
        fig, subplots = init_fig(1, 2)
        axes = subplots[0]
        self.plot_y_distribution(data, axes, xlabel=xlabel)
        axes = subplots[1]
        self.plot_y_distribution(data, axes, xlabel=xlabel, islog=True)
        savefig(fig, fname)
    

class AdditiveConvolutionalModel(ConvolutionalModel):
    def __init__(self, ref_seq, alphabet_type='rna',
                 n_filters=1, positional_effects=False,
                 intercept=None, slope=None,
                 multiple_backgrounds=False,
                 variable_theta=False,
                 fixed_theta=False,
                 alpha_type=False, position_bulges=False,
                 allow_bulges=False, base_bulges=False,
                 selected_bulges_positions=None,
                 n_alleles=4, recompile=False):
        filter_size = len(ref_seq)
        self.set_parameters(filter_size=filter_size, alphabet_type=alphabet_type,
                            n_alleles=n_alleles, n_filters=n_filters,
                            alpha_type=alpha_type,
                            multiple_backgrounds=multiple_backgrounds,
                            variable_theta=variable_theta,
                            fixed_theta=fixed_theta,
                            intercept=intercept, slope=slope,
                            allow_bulges=allow_bulges, base_bulges=base_bulges,
                            selected_bulges_positions=selected_bulges_positions,
                            position_bulges=position_bulges,
                            positional_effects=positional_effects,
                            recompile=recompile)
        self.allow_bulges = allow_bulges
        self.base_bulges = base_bulges
        self.ref_seq = ref_seq
    
    def seq_to_encoding(self, seq, bulge_pos=None):
        features = self.bulges_params.copy()
        target, bulge = self.get_target_and_bulge(seq, bulge_pos)
        
        if self.filter_size != len(target):
            raise ValueError('Sequence must have filter size')
        
        for i, (a1, a2) in enumerate(zip(self.ref_seq, target)):
            if a1 != a2:
                features['{}{}{}'.format(a1, i+1, a2)] = 1
    
        self.update_bulge_features(features, bulge, bulge_pos)
        
        return(features)
    
    
class BPStacksConvolutionalModel(ConvolutionalModel):
    def __init__(self, template, allow_bulges=False, base_bulges=False,
                 intercept=None, slope=None,
                 alpha_type=False, position_bulges=False,
                 selected_bulges_positions=None,
                 recompile=False, positional_effects=False):
        filter_size = len(template)
        self.set_parameters(filter_size=filter_size, alphabet_type='rna',
                            alpha_type=alpha_type,
                            intercept=intercept, slope=slope,
                            selected_bulges_positions=selected_bulges_positions,
                            allow_bulges=allow_bulges, base_bulges=base_bulges,
                            position_bulges=position_bulges,
                            positional_effects=positional_effects,
                            recompile=recompile)
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
        self.bulges_params.update(stacks)
        
        self.feature_names = sorted(self.bulges_params.keys()) 
        self.template = template
        
    def seq_to_encoding(self, seq, bulge_pos=None):
        target, bulge = self.get_target_and_bulge(seq, bulge_pos)
            
        features = self.bulges_params.copy()
        for i in range(self.filter_size - 1):
            stack = '{}|{}'.format(self.template[i:i+2], target[i:i+2])
            try:
                features[stack] += 1
            except KeyError:
                continue
            
        self.update_bulge_features(features, bulge, bulge_pos)
        
        return(features)
