#!/usr/bin/env python
import os
import warnings
from itertools import product
from os.path import exists, join

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logomaker

from tqdm import tqdm
from Bio import motifs
from Bio.Seq import Seq
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh
from scipy.sparse._matrix_io import save_npz, load_npz
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp

from gpmap.base import SequenceSpace, get_sparse_diag_matrix
from gpmap.plot_utils import init_fig, savefig, arrange_plot, init_single_fig
from gpmap.settings import CACHE_DIR, CMAP, PLOTS_DIR
from scipy.sparse.linalg.dsolve.linsolve import spsolve


class Visualization(SequenceSpace):
    def __init__(self, length, n_alleles=None, ns=1,
                 log=None, cache_prefix=None, alphabet_type='dna', 
                 label=None):
        self.init(length, n_alleles, log=log, alphabet_type=alphabet_type)
        
        self.cache_prefix = cache_prefix
        self.cached_T = False
        self.cached_eigenvectors = False
        self.cached_eigenvalues = False
        self.ns = ns
        self.label = label
        self.calc_adjacency()
    
    def set_random_function(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.load_function(f=np.random.normal(size=self.n_genotypes))
    
    def load_function(self, f, label=None):
        self.report('Loading function data')
        self.f = np.array(f)
        self.label_function = label
    
    def calc_delta_f(self, rows, cols):
        return(self.f[cols] - self.f[rows])
    
    def calc_average_function(self):
        return(np.mean(self.f))
    
    def calc_stationary_frequencies(self, silent=False):
        if not silent:
            self.report('Calculating stationary frequencies')
            
        log_freqs = self.ns * self.f
        self.log_total = logsumexp(log_freqs)
        self.log_genotypes_stationary_frequencies = log_freqs - self.log_total
        freqs = np.exp(self.log_genotypes_stationary_frequencies)
        
        # Store equilibrium frequencies
        self.genotypes_stationary_frequencies = freqs
        return(freqs)
    
    def get_stationary_frequencies(self):
        if not hasattr(self, 'genotypes_stationary_frequencies'):
            self.calc_stationary_frequencies(silent=True)
        return(self.genotypes_stationary_frequencies)
    
    def calc_stationary_function(self):
        self.fmean = np.sum(self.f * self.get_stationary_frequencies())
        return(self.fmean)
    
    def tune_ns(self, stationary_function=None, perc=None, tol=1e-8, maxiter=100):
        if perc is not None:
            stationary_function = np.percentile(self.f, perc)
        elif stationary_function is None:
            msg = 'Either stationary_function or percentile must be provided'
            raise ValueError(msg)
            
        msg = 'Optimizing Ns to reach a stationary state with mean(f)={}'
        self.report(msg.format(stationary_function))
        
        def f(param):
            self.ns = np.exp(param)
            self.calc_stationary_frequencies(silent=True)
            x = self.calc_stationary_function()
            rel_error = (stationary_function - x) ** 2
            return(rel_error)
        
        for _ in range(10):
            result = minimize(f, x0=self.ns, tol=tol,
                              options={'maxiter': maxiter})
            if f(result.x[0]) < tol:
                break

        self.ns = np.exp(result.x[0])
        self.calc_stationary_frequencies(silent=True)
        self.calc_stationary_function()
        self.report('\tOptimal Ns value: {}'.format(self.ns))
        self.report('\tExpected average function: {}'.format(self.fmean))
        return(self.fmean)
    
    def calc_rate(self, delta_f):
        S = self.ns * delta_f
        return(S / (1 - np.exp(-S)))

    def calc_rate_vector(self, delta_f):
        rate = np.ones(delta_f.shape[0])
        idxs = np.isclose(delta_f, 0) == False
        rate[idxs] = self.calc_rate(delta_f[idxs])
        return(rate)
    
    def check_symmetric(self, m, tol):
        if not (abs(m - m.T)> tol).nnz == 0:
            raise ValueError('Re-scaled rate matrix is not symmetric')
    
    def calc_rate_matrix(self):
        A = self.get_adjacency_matrix()
        rows, cols = A.row, A.col
        delta_f = self.calc_delta_f(rows, cols)
        size = (self.n_genotypes, self.n_genotypes)
        t = self.calc_rate_vector(delta_f)
        t = csr_matrix((t, (rows, cols)), shape=size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t.setdiag(-t.sum(1).A1)
        return(t)
        
    def calc_sparse_reweighted_rate_matrix(self, tol=1e-8, 
                                                 cache_matrix=False):
        self.report('Calculating re-weigthed symmetric rate matrix')
        rate_matrix = self.calc_rate_matrix()
        t = self.diag_freq_sparse.dot(rate_matrix).dot(self.diag_freq_inv_sparse)
        self.T = (t + t.T) / 2
        self.check_symmetric(self.T, tol=tol)

        if cache_matrix:
            self.save_sparse_matrix()
        
        # Recalculate rate matrix after ensuring time reversible process
        self.rate_matrix = self.diag_freq_inv_sparse.dot(self.T).dot(self.diag_freq_sparse)
        
        return(self.T)
    
    def calc_committor_probability(self, genotypes1, genotypes2):
        if np.intersect1d(genotypes1, genotypes2).shape[0] > 0:
            raise ValueError('The two sets of genotypes cannot overlap')
        
        # Select (AUB)^c genotypes
        idx = np.full(self.n_genotypes, True)
        all_gts = np.hstack([genotypes1, genotypes2])
        idx[self.genotype_idxs.loc[all_gts]] = False
        no_ab_genotypes = np.where(idx)[0]
        
        # Select B genotypes
        b_genotypes = self.genotype_idxs.loc[genotypes2]
        
        # Solve system Uq = v 
        partial_rate_matrix = self.rate_matrix[no_ab_genotypes, :]
        U = partial_rate_matrix[:, no_ab_genotypes]
        v = -partial_rate_matrix[:, b_genotypes].sum(1)
        q_reduced = spsolve(U, v)
                
        q = np.zeros(self.n_genotypes)
        q[no_ab_genotypes] = q_reduced
        q[b_genotypes] = 1
        return(q)

    def calc_genotypes_reactive_p(self, genotypes1, genotypes2):
        q = self.calc_committor_probability(genotypes1, genotypes2)
        log_stationary_freqs = self.log_genotypes_stationary_frequencies
        
        gt_log_p_reactive = log_stationary_freqs + np.log(q) + np.log(1-q)
        log_p_reactive = logsumexp(gt_log_p_reactive)
        gt_log_p_reactive = gt_log_p_reactive - log_p_reactive
        
        p_reactive = np.exp(log_p_reactive)
        gt_p = np.exp(gt_log_p_reactive)
        return(p_reactive, gt_p)
    
    def calc_edges_flow(self, genotypes1, genotypes2):
        q = self.calc_committor_probability(genotypes1, genotypes2)
        stationary_freqs = self.genotypes_stationary_frequencies
        
        i, j = self.get_neighbor_pairs()
        rate_ij = self.calc_rate_vector(self.calc_delta_f(i, j))
        flow = stationary_freqs[i] * (1-q)[i] * rate_ij * q[j]
        return(flow)
    
    def save_sparse_matrix(self):
        if self.cache_prefix is not None:
            fpath = join(CACHE_DIR, '{}.T'.format(self.cache_prefix))
            msg = 'Saving re-weighted rate matrix at {}'.format(fpath)
            self.report(msg)
            save_npz(fpath, self.T)
    
    def _load_csv(self, label):
        fpath = self._get_cache_fpath(label)
        if exists(fpath):
            self.report('Loading {} from {}'.format(label, fpath))
            df = pd.read_csv(fpath, index_col=0)
            return(df)
        else:
            self.report('Could not find rate matrix {}'.format(fpath))
    
    def _get_cache_fpath(self, label):
        fname = '{}.{}.csv'.format(self.cache_prefix, label)
        fpath = join(CACHE_DIR, fname)
        return(fpath)
    
    def load_eigendecomposition(self):
        self.report('Loading eigendecomposition')
        u = self._load_csv('right_eigenvectors')
        l = self._load_csv('eigenvalues')

        if u is None or l is None:
            raise ValueError('Error loading eigendecomposition: check tmp files')
        self.right_eigenvectors = u.values
        self.eigenvalues = l.values[:, 0]
        self.n_components = l.shape[0]
        self.cached_eigenvectors = True
        self.cached_eigenvalues = True

    @property
    def T_fpath(self):
        return(join(CACHE_DIR, '{}.T.npz'.format(self.cache_prefix)))
    
    def load_sparse_reweighted_rate_matrix(self):
        if self.cache_prefix is not None:
            if exists(self.T_fpath):
                msg = 'Loading re-weighted rate matrix from {}'
                self.report(msg.format(self.T_fpath))
                
                self.T = load_npz(self.T_fpath)
                self.cached_T = True
            else:
                self.report('Could not find rate matrix {}'.format(self.T_fpath))
    
    def get_sparse_reweighted_rate_matrix(self, recalculate=False,
                                                cache_matrix=True):
        if recalculate:
            self.calc_sparse_reweighted_rate_matrix(cache_matrix=cache_matrix)
        else:
            if hasattr(self, 'T'):
                pass
            elif cache_matrix:
                self.load_sparse_reweighted_rate_matrix()
                self.cached_T = True
            if not hasattr(self, 'T'):
                self.calc_sparse_reweighted_rate_matrix(cache_matrix=cache_matrix)

    def calc_reweighting_diag_matrices(self):
        # New basis and re-weighting genotypes by frequency
        sqrt_freqs = np.sqrt(self.genotypes_stationary_frequencies)
        self.diag_freq_sparse = get_sparse_diag_matrix(sqrt_freqs) 
        self.diag_freq_inv_sparse = get_sparse_diag_matrix(1 / sqrt_freqs)
    
    def calc_A_sparse(self):
        self.report('Calculating A = I - 1/c * T for eigen-decomposition')
        self.c = np.abs(self.T).sum(1).max()
        self.report('\tc = {:.3f}'.format(self.c))
        I_sparse = identity(self.n_genotypes) 
        self.A_sparse = I_sparse + 1 / self.c * self.T
    
    def calc_eigendecomposition(self, n_components=10, tol=1e-12,
                                cache_matrix=True):
        n_components = min(n_components, self.n_genotypes-1)
        self.report('Calculating eigen-decomposition of re-weighted rate matrix')
        self.calc_A_sparse()
        lambdas, q = eigsh(self.A_sparse, n_components, which='LM', tol=tol)
        self.n_components = n_components
        self.eigenvalues = self.c * (lambdas - 1)
        self.q = q
        self.left_eigenvectors = self.diag_freq_sparse.dot(q)
        self.right_eigenvectors = self.diag_freq_inv_sparse.dot(q)
        if cache_matrix:
            self.save_eigendecomposition()
    
    def _save_eigenvalues(self):
        df = pd.DataFrame({'eigenvalues': self.eigenvalues})
        fpath = self._get_cache_fpath('eigenvalues')
        self.report('Saving eigenvalues at {}'.format(fpath))
        df.to_csv(fpath)
    
    def _save_right_eigenvectors(self):
        df = pd.DataFrame(self.right_eigenvectors)
        fpath = self._get_cache_fpath('right_eigenvectors')
        self.report('Saving eigenvectors at {}'.format(fpath))
        df.to_csv(fpath)
    
    def save_eigendecomposition(self):
        if self.cache_prefix is not None:
            self._save_eigenvalues()
            self._save_right_eigenvectors()
    
    def check_eigendecomposition(self, tol=1e-3):
        self.report('Testing eigendecomposition of T')
        T = self.diag_freq_inv_sparse.dot(self.T).dot(self.diag_freq_sparse)
        for i in range(self.eigenvalues.shape[0]):
            u = self.right_eigenvectors[:, i]
            v1 = T.dot(u)
            v2 = self.eigenvalues[i] * u
            abs_err = np.mean(np.abs(v1 - v2)) 
            if abs_err > tol:
                msg = 'Numeric error in eigendecomposition: abs error = {:.5f}'
                self.report(msg.format(abs_err))
                raise ValueError()
        self.report('Eigendecomposition is correct')
    
    def get_eigendecomposition(self, n_components=10, tol=1e-9,
                               recalculate=False, cache_matrix=True, save=True):
        if not recalculate and not hasattr(self, 'eigenvalues') and self.cache_prefix is not None:
            try:
                self.load_eigendecomposition()
                self.check_eigendecomposition()
            except ValueError:
                msg ='Loaded eigenvectors are not eigenvectors of T: '
                msg += 'they will be recalculated'
                self.report(msg)
                recalculate = True
        else:
            recalculate = True
                
        if recalculate:
            self.calc_eigendecomposition(n_components=n_components, tol=tol,
                                         cache_matrix=cache_matrix)
            if save:
                self.save_eigendecomposition()
    
    def get_rescaled_projection(self):
        self.report('Scaling projection axis')
        projection = []
        for i in range(1, self.n_components + 1):
            eigenvalue = self.eigenvalues[-i]
            right_eigenvector = self.right_eigenvectors[:, -i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projection.append(right_eigenvector / np.sqrt(-eigenvalue))
        self.projection = np.vstack(projection)
        
    def filter_genotypes(self, selected_genotypes):
        self.selected_genotypes = selected_genotypes
        self.T = self.T[selected_genotypes, :]
        self.T = self.T[:, selected_genotypes]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.T.setdiag(0)
            A = self.A.tocsr()
            A = A[selected_genotypes, :]
            A = A[:, selected_genotypes]
            self.A = A.tocoo()
            
            d = -self.T.sum(1).A1
            self.T.setdiag(d)
        
        if hasattr(self, 'projection'):
            self.projection = self.projection[:, selected_genotypes]
            
        self.f = self.f[selected_genotypes]
        self.genotype_labels = self.genotype_labels[selected_genotypes]
        self.seqs = [seq for k, seq in zip(selected_genotypes, self.seqs) if k]
        self.n_genotypes = self.f.shape[0]
    
    def calc_visualization(self, meanf=None, n_components=100, tol=1e-9, recalculate=False,
                           cache_matrix=True, save=True):
        if meanf is not None:
            self.tune_ns(stationary_function=meanf)
        self.calc_stationary_frequencies()
        self.calc_reweighting_diag_matrices()
        self.get_sparse_reweighted_rate_matrix(recalculate=recalculate,
                                                     cache_matrix=cache_matrix)
        self.get_eigendecomposition(n_components, tol=tol,
                                    recalculate=recalculate,
                                    cache_matrix=cache_matrix,
                                    save=save)
        self.get_rescaled_projection()
    
    def calc_log_decay_rates(self):
        return(np.log(-1 / (self.eigenvalues[:-1])[::-1]))
    
    '''
    Plotting methods
    '''
    def get_nodes_coord(self, axis=[1, 2], force=False):
        if not hasattr(self, 'nodes') or force:
            if axis[-1] == 'f':
                self.nodes = np.vstack([self.projection[axis[:-1]],
                                        self.f]).transpose()
            else:
                self.nodes = self.projection[axis].transpose()
        return(self.nodes)
    
    def _calc_edges_coord(self, coords):
        i, j = self.get_neighbor_pairs()
        edges = np.stack([coords[i], coords[j]], axis=2).transpose((0, 2, 1))
        return(edges)
    
    def get_edges_coord(self, coords, force=False):
        if not hasattr(self, 'edges') or force:
            self.edges = self._calc_edges_coord(coords)
        return(self.edges)
    
    def plot_eigenvalues(self, axes):
        self.report('Plotting eigenvalues')
        k = np.arange(1 ,self.eigenvalues.shape[0])
        decay_rates = -1 / self.eigenvalues[:-1][::-1]
        axes.plot(k, decay_rates, linewidth=1, color='purple')
        axes.scatter(k, decay_rates, s=15, c='purple')
        axes.set_yscale('log')
        axes.set_xlabel(r'Eigenvalue order $k$', fontsize=14)
        axes.set_ylabel(r'$\frac{-1}{\lambda_{k}}$',
                        fontsize=14)

    def plot_edges(self, edges, axes):
        ndim = edges.shape[2]
        if ndim == 2:
            ln_coll = LineCollection(edges, color='grey', linewidths=0.5,
                                     alpha=0.2, zorder=1)
        elif ndim == 3:
            ln_coll = Line3DCollection(edges, color='grey', linewidths=0.5,
                                       alpha=0.2, zorder=1)
        else:
            msg = 'Only 2 or 3 dimensions allowed: {} found'.format(ndim)
            raise ValueError(msg)
        axes.add_collection(ln_coll)
    
    def get_nodes_colors(self, color_key=None, colors=None):
        if color_key is None:
            if colors is None:
                colors = self.f
        else:
            if colors is None:
                colors = np.array([color_key(seq) for seq in self.genotype_labels])
            else:
                msg = 'color_key and colors arguments are incompatible' 
                raise ValueError(msg)
        return(colors)
    
    def plot_local_maxima(self, axes, coords, colors, lw, size_factor, cmap):
        self.calc_local_maxima()
        i = self.local_maxima
        local_max_coords, max_value = coords[i], colors[i]
        self.plot_nodes_coords(axes, local_max_coords, max_value,
                               lw=lw, size_factor=size_factor, cmap=cmap)
    
    def sort_nodes(self, coords, colors, reverse=False):
        idx = np.argsort(colors)
        if reverse:
            idx = idx[::-1]
        coords, colors = coords[idx], colors[idx]
        return(coords, colors)
    
    def plot_nodes_coords(self, axes, coords, colors, lw, size_factor, cmap):
        ndim = coords.shape[1]
        if ndim == 2:
            sc = axes.scatter(coords[:, 0], coords[:, 1], c=colors,
                              linewidth=lw, s=size_factor, zorder=2,
                              edgecolor='black', cmap=cmap)
        elif ndim == 3:
            sc = axes.scatter(coords[:, 0], coords[:, 1],
                              zs=coords[:, 2], c=colors,
                              linewidth=lw, s=size_factor, zorder=2,
                              edgecolor='black', cmap=cmap, alpha=1)
        else:
            msg = 'Expected coords with dimension 2 or 3: {} found'.format(ndim)
            raise ValueError(msg)
        
        return(sc)
    
    def plot_nodes(self, coords, axes, cmap, label=None, size_factor=2.5,
                   colors=None, color_key=None, use_cmap=False,
                   sort=True, reverse=False, 
                   lw=0, highlight_local_maxima=False):
        colors = self.get_nodes_colors(color_key=color_key, colors=colors)
        if sort:
            coords, colors = self.sort_nodes(coords, colors, reverse=reverse)
        
        sc = self.plot_nodes_coords(axes, coords, colors,
                                    lw=lw, size_factor=size_factor, cmap=cmap)
        
        if color_key is None or use_cmap:
            plt.colorbar(sc, ax=axes).set_label(label=self.get_cmap_label(label), size=14)
        
        if highlight_local_maxima:
            self.plot_local_maxima(axes, coords, colors,
                                   lw=lw, size_factor=size_factor, cmap=cmap)
    
    def plot_labels(self, coords, axes, fontsize=6, labels_subset=None,
                    start=0, end=None):
        self.report('\tPlotting labels...')
        for gt, xt, yt in zip(self.seqs, coords[0], coords[1]):
            gt = ''.join([self.alphabet[a] for a in gt])
            if labels_subset is not None and gt not in labels_subset:
                continue
            if end is None:
                end = len(gt)
            else:
                end = min(end, len(gt))
            axes.text(xt, yt, gt[start:end], fontsize=fontsize)
    
    def plot_ns_mean_function(self, ns_min, ns_max, axes=None,
                              ylabel='Mean fitness', fname=None):
        ns_values = np.linspace(ns_min, ns_max, 51)
        fmean = []
        for ns in ns_values:
            self.set_ns(ns)
            self.calc_stationary_frequencies(silent=True)
            fmean.append(self.calc_stationary_function())
        
        if axes is None:
            if fname is None:
                raise ValueError('Either axes or fname must be provided')
            fig, axes = init_fig(1, 1)
        else:
            fig = None
            
        axes.plot(ns_values, fmean, lw=1, c='purple')
        arrange_plot(axes, xlims=(ns_min, ns_max), xlabel=r'$N_{e}s$',
                     ylabel=ylabel)
        
        if fig is not None:
            savefig(fig, fname)
    
    def minimize_distance(self, coords, prev_coords=None):
        
        if prev_coords is not None:
            test_coords = [np.vstack([v * s for v, s in zip(coords.T, scalars)]).T
                           for scalars in product([1, -1], repeat=coords.shape[1])]
            distances = [np.sqrt(np.sum((c - prev_coords) ** 2, 1)).mean(0)
                         for c in test_coords]
            coords = test_coords[np.argmin(distances)]
        return(coords)

    def _rotate_coords(self, coords, theta, axis):
        if axis == 'x':
            m = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            m = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 'z':
            m = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        else:
            raise ValueError('Axis can only be x,y,z')
        
        return(np.dot(coords, m))

    def rotate_coords(self, coords, thetas, axis):
        for theta, a in zip(thetas, axis):
            coords = self._rotate_coords(coords, theta, axis=a)
        return(coords)
    
    def get_logo_df(self, sel_idxs=None):
        seqs = self.genotype_labels
        if sel_idxs is not None:
            seqs = seqs[sel_idxs]
        instances = [Seq(seq) for seq in seqs]
        motif = motifs.create(instances, alphabet=''.join(self.alphabet))
        pwm = motif.counts.normalize()
        return(pd.DataFrame(pwm))
    
    def plot_logo(self, sel_idxs, axes, xticklabels=None):
        logo_df = self.get_logo_df(sel_idxs)
        logomaker.Logo(logo_df, ax=axes)
        axes.set_xticks(np.arange(self.length))
        
        if xticklabels is None:
            xticklabels = np.arange(self.length) + 1
        axes.set_xticklabels(xticklabels)
            
        sns.despine(ax=axes)

    def add_axis_labels(self, axes, x, y, z):
        axes.set_xlabel('Diffusion Axis {}'.format(x), fontsize=14)
        axes.set_ylabel('Diffusion Axis {}'.format(y), fontsize=14)
        if z is not None:
            axes.set_zlabel('Diffusion Axis {}'.format(z), fontsize=14)
    
    def plot(self, axes, x=1, y=2, z=None, show_edges=True, cmap=CMAP,
             label=None, show_labels=False, size=5, fontsize=6, colors=None,
             labels_subset=None, start=None, end=None, color_key=None,
             use_cmap=False, sort=True, reverse=False, lw=0, force_coords=True, 
             prev_coords=None, highlight_local_maxima=False, coords=None):
        
        axis = [x, y]
        if z is not None:
            axis.append(z)

        if coords is None:
            coords = self.get_nodes_coord(axis=axis, force=force_coords)
            coords = self.minimize_distance(coords, prev_coords)
        
        if show_edges:
            edges = self.get_edges_coord(coords, force=force_coords)
            self.plot_edges(edges, axes)
        
        self.plot_nodes(coords, axes, cmap, label, size_factor=size,
                        color_key=color_key, colors=colors,
                        use_cmap=use_cmap, sort=sort,
                        reverse=reverse, lw=lw,
                        highlight_local_maxima=highlight_local_maxima)
        
        if show_labels:
            self.plot_labels(coords, axes, fontsize=fontsize,
                             labels_subset=labels_subset, start=start, end=end)
        
        self.add_axis_labels(axes, x, y, z)
                
        return(coords)

    def plot_function_distrib(self, axes, label):    
        sns.distplot(self.f, bins=30, hist=True, kde=False, color='purple',
                     ax=axes)
        xlims = min(self.f), max(self.f)
        arrange_plot(axes, xlabel=label, ylabel='Number of genotypes',
                     xlims=xlims)
    
    def get_fname_plot(self, suffix, fname=None):
        if fname is None:
            if self.label is None:
                raise ValueError('Landscape label or fname must be provided')
            fname = '{}.{}'.format(self.label, suffix)
        return(fname)
    
    def get_cmap_label(self, label=None):
        if label is None:
            return(self.label_function)
        else:
            return(label)
    
    def get_eq_functions(self, fmin, fmax, n):
        if fmin is None:
            fmin = self.f.mean() + 0.2 * (self.f.max() - self.f.mean())
        if fmax is None:
            fmax = self.f.mean() + 0.8 * (self.f.max() - self.f.mean())
            
        eq_fs = np.linspace(fmin, fmax, n)
        return(eq_fs)
    
    '''
    Full figure methods
    '''
    
    def figure(self, fname=None, show_edges=True,
               cmap=CMAP, label=None, show_labels=False,
               size=5, fontsize=6, labels_subset=None,
               start=0, end=None, color_key=None, colors=None, lw=0,
               force_coords=True, highlight_local_maxima=False,
               x=1, y=2, z=None):
        
        fig, axes = init_single_fig(figsize=(10, 7.6), is_3d=z is not None)
        self.report('Plotting landscape visualization')
        self.plot(axes, x=x, y=y, z=z, show_edges=show_edges,
                  cmap=cmap, label=label, show_labels=show_labels,
                  size=size, fontsize=fontsize, force_coords=force_coords,
                  labels_subset=labels_subset, lw=lw,
                  start=start, end=end, color_key=color_key, colors=colors,
                  highlight_local_maxima=highlight_local_maxima)
        self.report('Saving plot')
        fname = self.get_fname_plot(suffix='visualization', fname=fname)
        savefig(fig, fname)
    
    def plot_grid_allele(self, fname=None, show_edges=True,
                         color='orange', size=5, lw=0):
        fname = self.get_fname_plot(suffix='alleles', fname=fname)
        
        fig, subplots = init_fig(self.n_alleles, self.length, colsize=3, rowsize=2.7)
        
        force_coords = True
        for i in range(self.n_alleles):
            for j in range(self.length):
                allele = self.alphabet[i]
                axes = subplots[i][j]
                
                self.plot(axes, x=2, y=3, show_edges=show_edges,
                          size=size, force_coords=force_coords, sort=True, lw=lw,
                          color_key=lambda x: color if x[j] == allele else 'lightgrey')
                if i < self.n_alleles - 1:
                    axes.set_xlabel('')
                    axes.set_xticks([])
                if j > 0:
                    axes.set_ylabel('')
                    axes.set_yticks([])
                    
                xlims, ylims = axes.get_xlim(), axes.get_ylim()
                x = xlims[0] + 0.05 * (xlims[1] - xlims[0])
                y = ylims[0] + 0.92 * (ylims[1] - ylims[0])
                axes.text(x, y, '{}{}'.format(allele, j+1), ha='center')
                force_coords = False
        
        self.report('Saving plot at {}'.format(fname))
        savefig(fig, fname)
    
    def plot_grid_shifting_motifs(self, motif, fname=None, show_edges=True,
                                  color='orange', size=5, lw=0):
        fname = self.get_fname_plot(suffix=motif, fname=fname)
        motif_length = len(motif)
        n_plots = self.length - motif_length + 1
        
        fig, subplots = init_fig(2, n_plots, colsize=3, rowsize=2.7)

        prev_j = 0
        force_coords = True   
        for j, x, y in [(0, 2, 3), (1, 4, 5)]:
            for i in range(n_plots):
                axes = subplots[j, i]
                
                self.plot(axes, x=x, y=y, show_edges=show_edges,
                          size=size, sort=True, lw=lw,
                          color_key=lambda x: color if x[i:i+motif_length] == motif else 'lightgrey',
                          force_coords=j != prev_j or force_coords)
                if i > 0:
                    axes.set_ylabel('')
                    axes.set_yticks([])
                    
                xlims, ylims = axes.get_xlim(), axes.get_ylim()
                x_pos = xlims[0] + 0.04 * (xlims[1] - xlims[0]) * motif_length
                y_pos = ylims[0] + 0.92 * (ylims[1] - ylims[0])
                axes.text(x_pos, y_pos, '{}{}'.format(motif, i+1), ha='center')
                force_coords = False
        
        self.report('Saving plot at {}'.format(fname))
        savefig(fig, fname)
    
    def plot_grid_eq_f(self, fname=None, fmin=None, fmax=None,
                       ncol=4, nrow=3, show_edges=True, size=5, cmap=CMAP,
                       label=None, lw=0, n_components=4):
        fname = self.get_fname_plot(suffix='ns', fname=fname)
        
        if fmin is None:
            fmin = self.f.mean() + 0.05 * (self.f.max() - self.f.mean())
        if fmax is None:
            fmax = self.f.mean() + 0.8 * (self.f.max() - self.f.mean())

        eq_fs = np.linspace(fmin, fmax, ncol*nrow)
        
        fig, subplots = init_fig(nrow, ncol, colsize=3, rowsize=2.7)
        subplots = subplots.flatten()
        
        fig2, subplots2 = init_fig(nrow, ncol, colsize=3, rowsize=2.7)
        subplots2 = subplots2.flatten()
        
        coords = None
        for i, (eq_f, axes, axes_eig) in enumerate(zip(eq_fs, subplots, subplots2)):
            self.calc_visualization(meanf=eq_f, n_components=n_components,
                                    recalculate=True, cache_matrix=False)
            self.plot_eigenvalues(axes_eig)
            axes_eig.set_title('Stationary f = {:.2f}'.format(eq_f))
            
            coords = self.plot(axes, x=2, y=3, show_edges=show_edges,
                               size=size, cmap=cmap, label=label,
                               force_coords=True, lw=lw,
                               prev_coords=coords)
            axes.set_title('Stationary f = {:.2f}'.format(eq_f))
            
            if i // ncol != nrow - 1:
                axes.set_xlabel('')
                axes.set_xticks([])
                axes_eig.set_xlabel('')
                axes_eig.set_xticks([])
            
            if i % ncol != 0:
                axes.set_ylabel('')
                axes.set_yticks([])
                axes_eig.set_xlabel('')
                axes_eig.set_xticks([])
        
        self.report('Saving plot at {}'.format(fname))
        savefig(fig, fname)
        savefig(fig2, fname + '.eig')
    
    def plot_complete(self, fname=None, show_edges=True,
                      cmap=CMAP, label=None, show_labels=False, size=6,
                      lw=0, fmt='png'):
        fname = self.get_fname_plot(suffix='complete', fname=fname)
        
        fig, subplots = init_fig(3, 2, colsize=5, rowsize=3.8)
        self.plot_eigenvalues(subplots[0][0])
        self.plot_function_distrib(subplots[1][0], label=label)
        self.plot(subplots[1][1], x=2, y=3, show_edges=show_edges,
                  cmap=cmap, label=label, show_labels=show_labels,
                  size=size, force_coords=True, lw=lw)
        self.plot(subplots[2][0], x=2, y=4, show_edges=show_edges,
                  cmap=cmap, label=label, show_labels=show_labels,
                  size=size, force_coords=True, lw=lw)
        self.plot(subplots[2][1], x=3, y=4, show_edges=show_edges,
                  cmap=cmap, label=label, show_labels=show_labels,
                  size=size, force_coords=True, lw=lw)
        self.report('Saving plot at {}'.format(fname))
        savefig(fig, fname, fmt=fmt)
    
    '''
    Interactive plotting methods
    '''
    
    def get_plotly_edges(self, coords, force_coords=True):
        ndim = coords.shape[1]
        edge_x = []
        edge_y = []
        edge_z = []
        edges = self.get_edges_coord(coords, force=force_coords)
        for node1, node2 in zip(edges[:, 0, :], edges[:, 1, :]):
            edge_x.extend([node1[0], node2[0], None])
            edge_y.extend([node1[1], node2[1], None])
            if ndim > 2:
                edge_z.extend([node1[2], node2[2], None])
                
        if ndim == 2:
            return(edge_x, edge_y)
        else:
            return(edge_x, edge_y, edge_z)
    
    def save_plotly(self, fig, fname=None):
        if fname is None:
            fig.show()
        else:
            fpath = join(PLOTS_DIR, '{}.html'.format(fname))
            fig.write_html(fpath)
    
    def plot_interactive_2d(self, colorlabel='Function', fname=None,
                            cmap=CMAP, show_edges=True, color=None,
                            force_coords=True):
        '''Inspired by https://plotly.com/python/network-graphs/'''
        
        self.report('Make interactive landscape visualization')
        fname = self.get_fname_plot(suffix='interactive', fname=fname)
        
        # Create nodes plot
        if color is None:
            color = self.f
        node_x, node_y = self.get_nodes_coord(force=force_coords)
        # cbar_length = (node_y.max() - node_y.min()) / 10
        cbar_length = 1
        colorbar = dict(thickness=25, title=colorlabel, xanchor='left',
                        titleside='right', len=cbar_length)
        marker = dict(showscale=True, colorscale=cmap, reversescale=False,
                      color=color, size=10, colorbar=colorbar, line_width=2)
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                                hoverinfo='text', marker=marker,
                                text=self.genotype_labels)
        traces = [node_trace]

        # Create edges        
        if show_edges:
            edge_x, edge_y = self.get_plotly_edges(node_x, node_y,
                                                   force_coords=force_coords)
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                    line=dict(width=0.5, color='#888'),
                                    hoverinfo='none', mode='lines')
            traces = [edge_trace, node_trace]
        
        # Create figure
        axis = dict(showgrid=False, zeroline=False, showticklabels=True)        
        layout = go.Layout(hovermode='closest',
                           xaxis=axis, yaxis=axis,
                           template='simple_white', title="Landscape visualization",
                           xaxis_title="Diffusion axis 1", yaxis_title="Diffusion axis 2")
        fig = go.Figure(data=traces, layout=layout)
        self.save_plotly(fig, fname=fname)
    
    def plot_interactive_3d(self, colorlabel=None, fname=None,
                            cmap=CMAP, show_edges=True, color=None,
                            force_coords=True, z=3):
        '''Inspired by https://plotly.com/python/v3/3d-network-graph/'''
        
        self.report('Make interactive landscape visualization')
        colorlabel = self.get_cmap_label(colorlabel)
        fname = self.get_fname_plot(suffix='interactive3d', fname=fname)
        
        # Create nodes plot
        if color is None:
            color = self.f
        coords = self.get_nodes_coord(axis=[1, 2, z], force=force_coords)
        node_x, node_y, node_z = coords.transpose()
        cbar_length = 1
        colorbar = dict(thickness=25, title=colorlabel, xanchor='left',
                        titleside='right', len=cbar_length)
        marker = dict(showscale=True, colorscale=cmap, reversescale=False,
                      color=color, size=4, colorbar=colorbar, line_width=2)
        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                                  hoverinfo='text', marker=marker,
                                  text=self.genotype_labels)
        traces = [node_trace]

        # Create edges        
        if show_edges:
            edge_x, edge_y, edge_z = self.get_plotly_edges(coords,
                                                           force_coords=force_coords)
            edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                                      line=dict(width=0.5, color='#888'),
                                      hoverinfo='none', mode='lines')
            traces = [edge_trace, node_trace]
        
        # Create figure        
        axis = dict(showgrid=False, zeroline=False, showticklabels=True)
        layout = go.Layout(hovermode='closest',
                           scene=dict(xaxis=axis.update(dict(title="Diffusion axis 1")),
                                      yaxis=axis.update(dict(title="Diffusion axis 2")),
                                      zaxis=axis.update(dict(title="Diffusion axis 3"))),
                           template='simple_white', title="Landscape visualization")
        fig = go.Figure(data=traces, layout=layout)
        self.save_plotly(fig, fname=fname)
    
    '''
    Movies methods
    '''
    
    def get_movie_dir(self, fdir, suffix):
        prefix = self.get_fname_plot(suffix=suffix, fname=fdir)
        dpath = join(PLOTS_DIR, prefix) 
        
        if not exists(dpath):
            os.makedirs(dpath)
        return(prefix, dpath)
    
    def create_ns_frames(self, prefix, dpath, fmin=None, fmax=None,
                         show_edges=True, size=5, cmap=CMAP,
                         label=None, lw=0, n_components=50,
                         nframes=120, force=False):
        eq_fs = self.get_eq_functions(fmin=fmin, fmax=fmax, n=nframes)
        
        fpaths = []
        coords = None
        for i, eq_f in tqdm(list(enumerate(eq_fs))):
            fname = '{}/{}'.format(prefix, i)
            fpath = join(dpath, '{}.png'.format(i))
            fpaths.append(fpath)
            if exists(fpath) and not force:
                continue
            
            self.tune_ns(eq_f)
            self.calc_visualization(n_components=n_components, recalculate=True,
                                    cache_matrix=False, save=False)
            
            fig, subplots = init_fig(1, 2, colsize=3.5, rowsize=3.2)
            
            # Eigenvalues plot
            self.plot_eigenvalues(subplots[0])
            subplots[0].set_title('Stationary f = {:.2f}'.format(eq_f))
            
            # Plot visualization
            coords = self.plot(subplots[1], x=1, y=2, show_edges=show_edges,
                               size=size, cmap=cmap, label=label,
                               force_coords=True, lw=lw, prev_coords=coords)
            subplots[1].set_title('Stationary f = {:.2f}'.format(eq_f))
            savefig(fig, fname)
            
        return(fpaths, coords)

    def create_rotation_frames(self, prefix, dpath, coords, axis=['x'],
                               counter=0, ylabel=None, xlabel=None, zlabel=None,
                               show_edges=True, size=5, force=False,
                               cmap=CMAP, label=None, lw=0, nframes=90,
                               lims=None, size_factor=2, is_3d=True, z=3,
                               colors=None):
        thetas = np.linspace(0, 2*np.pi, nframes)
        
        fpaths = []
        
        self.report('Creating frames for movie at {}'.format(dpath))
        for theta in tqdm(thetas):
            fpath = join(dpath, '{}.png'.format(counter))
            fpaths.append(fpath)
            c = self.rotate_coords(coords, thetas=[theta] * len(axis), axis=axis)
            fname = '{}/{}'.format(prefix, counter)
            counter += 1
            
            if exists(fpath) and not force:
                continue
        
            fig, axes = init_single_fig(colsize=3.5*size_factor,
                                 rowsize=3.2*size_factor, is_3d=is_3d)
            self.plot(axes, x=1, y=2, z=z if is_3d else None,
                      show_edges=show_edges, colors=colors,
                      size=size, cmap=cmap, label=label,
                      force_coords=True, lw=lw, coords=c)
            arrange_plot(axes, xlims=lims, ylims=lims, zlims=lims,
                         xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
            savefig(fig, fname)
            
        return(fpaths, c)
    
    def extend_last_frame(self, fpaths, tlast, fps):
        fpaths += [fpaths[-1]] * (tlast * fps)
        return(fpaths)
    
    def save_movie(self, fpaths, prefix, fps, tlast=0, boomerang=False):
        fpaths = self.extend_last_frame(fpaths, tlast, fps)
        if boomerang:
            fpaths = fpaths + fpaths[::-1]
        clip = ImageSequenceClip(fpaths, fps=fps)
        clip.write_videofile(join(PLOTS_DIR, '{}.mp4'.format(prefix)))
    
    def plot_ns_movie(self, fdir=None, fmin=None, fmax=None,
                      show_edges=True, size=5, cmap=CMAP,
                      label=None, lw=0, n_components=50,
                      nframes=120, fps=20, boomerang=False, force=False):
        prefix, dpath = self.get_movie_dir(fdir=fdir, suffix='ns_movie')
        fpaths, _ = self.create_ns_frames(prefix, dpath, fmin=fmin, fmax=fmax,
                                          show_edges=show_edges, size=size, cmap=cmap,
                                          label=label, lw=lw, n_components=n_components,
                                          nframes=nframes, force=force)
        self.save_movie(fpaths, prefix, fps, tlast=1, boomerang=boomerang)
        
    def plot_rotation_movie(self, fdir=None,
                            show_edges=True, size=5, cmap=CMAP,
                            label=None, lw=0,  nframes=120, fps=20, lims=None,
                            size_factor=2, force=False, z=3,
                            colors=None):
        self.report('Preparing frames for rotation movie')
        prefix, dpath = self.get_movie_dir(fdir=fdir, suffix='rotation_movie360')
        coords = self.get_nodes_coord(axis=[1, 2, 3])
        fpaths, coords = self.create_rotation_frames(prefix=prefix, dpath=dpath,
                                                     coords=coords, lims=lims,
                                                     show_edges=show_edges,
                                                     size=size, cmap=cmap, label=label,
                                                     lw=lw, nframes=nframes,
                                                     size_factor=size_factor,
                                                     force=force, z=z,
                                                     ylabel='Diffusion axis 2/3',
                                                     zlabel='Diffusion axis 2/3',
                                                     colors=colors)
        self.save_movie(fpaths, prefix, fps, tlast=1, boomerang=False)
        
    
class CodonFitnessLandscape(Visualization):
    def __init__(self, ns=1,
                 sel_codons=['UCU', 'UCA', 'UCC', 'UCG', 'AGU', 'AGC'],
                 add_variation=False, log=None):
        self.init(3, 4, log=log, alphabet_type='rna')
        self.cache_prefix = None
        self.cached_T = False
        self.cached_eigenvectors = False
        self.cached_eigenvalues = False
        self.ns = ns
        self.label = 'codon_landscape'
        self.stop_codons = ['UGA', 'UAA', 'UAG']
        self.alleles = ['A', 'C', 'G', 'U']
        self.genotypes = np.array([''.join(self.alleles[i] for i in seq)
                                   for seq in self.seqs])
        self.sel_codons = sel_codons
        self.add_variation = add_variation
        self.calc_fitness()
    
    def calc_fitness(self):
        fitness = []
        for gt in self.genotypes:
            if gt in self.sel_codons:
                f = 2
            elif gt in self.stop_codons:
                f = 0
            else:
                f = 1
            fitness.append(f)
        fitness = np.array(fitness)
        if self.add_variation:
            fitness = fitness + 1 / 10 * np.random.normal(size=len(fitness))
        self.load_function(fitness)
