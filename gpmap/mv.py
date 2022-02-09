#!/usr/bin/env python
from os.path import join

import numpy as np
import seaborn as sns
import mavenn as mv
import logomaker
from sklearn.isotonic import IsotonicRegression

from gpmap.plot_utils import init_fig, savefig, arrange_plot
from gpmap.settings import CACHE_DIR
from gpmap.visualization import Visualization


class MaveNN(Visualization):
    @property
    def confs(self):
        return([
                ('additive', False),
                # ('neighbor', False),
                ('pairwise', False),
                ('additive', True),
                # ('neighbor', True),
                ('pairwise', True),
                # ('blackbox', True)
                ])
        
    def get_mavenn_model_fname(self, gpmap_type, global_epistasis):
        suffix = 'GE' if global_epistasis else 'linear'
        fname = '{}.{}.{}'.format(self.cache_prefix, gpmap_type, suffix)
        return(fname)
    
    def get_mavenn_model_fpath(self, gpmap_type, global_epistasis):
        fname = self.get_mavenn_model_fname(gpmap_type, global_epistasis)
        fpath = join(CACHE_DIR, fname)
        return(fpath)
    
    def fit_mavenn(self, seqs, f, f_sigma=None, gpmap_type='additive', global_epistasis=True,
                   ge_heteroskedasticity_order=0, epochs=1000, early_stopping=False,
                   regression_type='GE', theta_regularization=1e-3):
        ge_type = 'nonlinear' if global_epistasis else 'linear'
        self.report('\tFitting {} - {} model'.format(gpmap_type, ge_type))
        model = mv.Model(L=self.length, alphabet=self.alphabet_type,
                         ge_nonlinearity_hidden_nodes=20,
                         gpmap_type=gpmap_type,
                         ge_nonlinearity_type=ge_type,
                         regression_type=regression_type,
                         ge_heteroskedasticity_order=ge_heteroskedasticity_order,
                         ge_noise_model_type='Empirical' if f_sigma is not None else 'Gaussian',
                         Y=f.shape[1] if len(f.shape) > 1 else 2,
                         theta_regularization=theta_regularization)
        model.set_data(x=seqs, y=f, dy=f_sigma)
            
        model.fit(learning_rate=.0005, epochs=epochs, batch_size=5000,
                  early_stopping=early_stopping, early_stopping_patience=5, 
                  linear_initialization=False)
        model.save(self.get_mavenn_model_fpath(gpmap_type, global_epistasis))
        self.mavenn_model = model
    
    def fit_all_models(self, seqs, f, epochs=1000, early_stopping=False):
        self.report('Fitting all simple models with MAVE-NN')
        for gpmap_type, global_epistasis in self.confs:
            self.fit_mavenn(seqs, f, gpmap_type, global_epistasis,
                            epochs=epochs, early_stopping=early_stopping)
    
    def load_mavenn_model(self, gpmap_type, global_epistasis, force=False):
        fpath = self.get_mavenn_model_fpath(gpmap_type, global_epistasis)
        if not hasattr(self, 'mavenn_model') or force:
            self.mavenn_model = mv.load(fpath)
        else:
            self.report('Model was already loaded')
    
    def predict(self, seqs):
        return(self.mavenn_model.x_to_yhat(seqs))
    
    def plot_information_history(self, axes, x_test, y_test):
        I_var = self.mavenn_model.I_variational(x=x_test, y=y_test)[0]
        I_pred = self.mavenn_model.I_predictive(x=x_test, y=y_test)[0]
        I_var_hist = self.mavenn_model.history['I_var']
        val_I_var_hist = self.mavenn_model.history['val_I_var']
        
        axes.plot(I_var_hist, label='I_var_train')
        axes.plot(val_I_var_hist, label='I_var_val')
        axes.axhline(I_var, color='C2', linestyle=':', label='I_var_test')
        axes.axhline(I_pred, color='C3', linestyle=':', label='I_pred_test')
        arrange_plot(axes, xlabel='epochs', ylabel='bits',
                     title='Information history', showlegend=True, legend_loc=4)
    
    def plot_loss_history(self, axes):
        axes.plot(self.mavenn_model.history['loss'])
        axes.plot(self.mavenn_model.history['val_loss'])
        arrange_plot(axes, xlabel='epochs', ylabel='$\mathcal{L}_{\text{like}}$',
                     title='Loss history')
    
    def plot_sequence_logo(self, axes):
        theta_logo = self.mavenn_model.get_theta()['logomaker_df']
        for c in theta_logo.columns:
            theta_logo[c] = theta_logo[c] - theta_logo.mean(1)
        logomaker.Logo(theta_logo.fillna(0), ax=axes)
    
    def plot_Rsq(self, axes, seqs, f, xfactor=0.9, yfactor=0.9):
        pred_f = self.mavenn_model.x_to_yhat(seqs)
        Rsq = np.corrcoef(pred_f.ravel(), f)[0, 1]**2
        xlims = axes.get_xlim()
        ylims = axes.get_ylim()
        x = xlims[0] + xfactor * (xlims[1] - xlims[0])
        y = ylims[0] + yfactor * (ylims[1] - ylims[0])
        axes.text(x, y, f'$R^2$={Rsq:.3}', ha='right')
    
    def plot_residuals(self, axes, seqs, f, ylims=None):
        # Predict measurement values (yhat) on test data
        pred_f = self.mavenn_model.x_to_yhat(seqs)
        res = f - pred_f
        axes.scatter(pred_f, res, color='C0', s=5, alpha=1)
        axes.axhline(0, color='C1', linestyle='--', lw=2)
        arrange_plot(axes, xlabel='Predicted $\hat{y}$', ylabel='Residues',
                     ylims=ylims)
        self.plot_Rsq(axes, seqs, f)
    
    def get_phi_grid(self, phi):
        phi_lim = [min(phi)-.5, max(phi)+.5]
        phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
        return(phi_grid)
    
    def plot_global_epistasis_pred(self, axes, seqs):
        phi = self.mavenn_model.x_to_phi(seqs)
        phi_grid = self.get_phi_grid(phi)
        yhat_grid = self.mavenn_model.phi_to_yhat(phi_grid)
        
        # Compute 90% CI for each yhat
        q = [0.05, 0.95] #[0.16, 0.84]
        yqs_grid = self.mavenn_model.yhat_to_yq(yhat_grid, q=q)
        
        axes.plot(phi_grid, yhat_grid, linewidth=2, color='C1',
                label='$\hat{y} = g(\phi)$')
        axes.plot(phi_grid, yqs_grid[:, 0], linestyle='--', color='C1',
                  label='90% CI')
        axes.plot(phi_grid, yqs_grid[:, 1], linestyle='--', color='C1')
    
    def plot_isotonic(self, axes, seqs, f):
        reg = IsotonicRegression()
        phi = self.mavenn_model.x_to_phi(seqs)
        phi_grid = self.get_phi_grid(phi)
        reg.fit(phi, f)
        pred = reg.predict(phi_grid)
        axes.plot(phi_grid, pred, linewidth=2, color='black',
                  label='Isotonic regression')
    
    def plot_global_epistasis(self, axes, seqs, f, add_isotonic=True,
                              title='Global epistasis', ref_seq=None,
                              color='C0' , alpha=1, add_pred=True):
        phi = self.mavenn_model.x_to_phi(seqs)
        phi_lim = [min(phi)-.5, max(phi)+.5]
        
        # Plot inferred phi and observed values
        axes.scatter(phi, f, color=color, s=5, alpha=alpha, zorder=1)
        if add_pred:
            self.plot_global_epistasis_pred(axes, seqs)
        
        if ref_seq:
            x, y = self.mavenn_model.x_to_phi(ref_seq), f[ref_seq]
            axes.scatter(x, y, color='black', s=15, alpha=1, label=ref_seq,
                         zorder=2, lw=1, edgecolor='white')

        # Plot isotonic regression to ensure convergence
        if add_isotonic:        
            self.plot_isotonic(axes, seqs, f)
        
        arrange_plot(axes, xlims=phi_lim, xlabel='latent phenotype ($\phi$)',
                     ylabel='measurement ($y$)', title=title)
    
    def figure_model(self, seqs, f, gpmap_type, global_epistasis,
                     logo=True, fname=None):
        
        self.load_mavenn_model(gpmap_type, global_epistasis)
        
        fig, subplots = init_fig(1, 4 if logo else 3, colsize=4, rowsize=3.2)
        
        axes = subplots[0]
        self.plot_loss_history(axes)
        
        axes = subplots[1]
        self.plot_information_history(axes, seqs, f)
        
        axes = subplots[2]
        self.plot_global_epistasis(axes, seqs, f, add_isotonic=True, title='',
                                   ref_seq=None)
        self.plot_Rsq(axes, seqs, f)
        
        if logo:
            axes = subplots[3]
            self.plot_sequence_logo(axes)
    
        if fname is None:
            fname = '{}.{}.{}.fit'.format(self.cache_prefix, gpmap_type, global_epistasis)    
        
        savefig(fig, fname)
    
    def figure_global_epistasis(self, seqs, f, add_isotonic=True,
                                res_ylims=None):
        
        fig, subplots = init_fig(5, len(self.confs), colsize=4, rowsize=3.2)
        
        for i, (gpmap_type, global_epistasis) in enumerate(self.confs):
            title = '{} - {}'.format(gpmap_type,
                                     'global epistasis' if global_epistasis
                                     else 'linear')
            self.load_mavenn_model(gpmap_type, global_epistasis)
            
            axes = subplots[0][i]
            self.plot_residuals(axes, seqs, f, ylims=res_ylims)
            axes.set_title(title)
            
            axes = subplots[1][i]
            self.plot_global_epistasis(axes, seqs, f, add_isotonic=add_isotonic,
                                       title='')
            
            axes = subplots[2][i]
            self.plot_additive_effects(axes, gauge='min')
            
            axes = subplots[3][i]
            self.plot_additive_effects(axes, gauge='max', cmap='Blues')
            
            axes = subplots[4][i]
            self.plot_pairwise_effects(axes)
            
        fname = '{}.global_epistasis.fit'.format(self.cache_prefix)
        savefig(fig, fname)
    
    def change_additive_gauge(self, theta, gauge):
        if gauge == 'standard':
            return(theta)
        elif gauge == 'min':
            return(theta - np.vstack([theta.min(1)] * 4).transpose())
        elif gauge == 'max':
            return(theta - np.vstack([theta.max(1)] * 4).transpose())
        else:
            raise ValueError('Gauge not allowed')
    
    def plot_additive_effects(self, axes, gauge='standard', cmap='Reds'):
        # TODO: plot gauge = 'consensus'
        # TODO: plot logo for additive models
        theta = self.mavenn_model.get_theta()
        t = self.change_additive_gauge(theta['theta_lc'], gauge)
        if np.all(np.isnan(t)):
            sns.despine(ax=axes, left=True, bottom=True)
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            cb = mv.heatmap(values=t, alphabet=theta['alphabet'], ax=axes,
                            cmap=cmap)[1]
            cb.set_label('Phenotypic effects')
            axes.set_xlabel('Pentamer position')
            axes.set_ylabel('Nucleotide')
            axes.set_xticks(np.arange(1, self.length + 1))
            axes.set_xticklabels(np.arange(1, self.length + 1))
    
    def plot_pairwise_effects(self, axes, gpmap_type='pairwise'):
        theta = self.mavenn_model.get_theta()
        if np.all(np.isnan(theta['theta_lclc'])):
            sns.despine(ax=axes, left=True, bottom=True)
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            mv.heatmap_pairwise(values=theta['theta_lclc'],
                                gpmap_type=gpmap_type,
                                alphabet=theta['alphabet'], ax=axes)
            axes.set_xticklabels(np.arange(1, self.length + 1))
