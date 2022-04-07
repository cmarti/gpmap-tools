#!/usr/bin/env python
import os
import warnings
from os.path import exists, join

import logomaker
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

from tqdm import tqdm
from Bio import motifs
from Bio.Seq import Seq
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from scipy.sparse import identity
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh
from scipy.optimize._minimize import minimize
from scipy.special._logsumexp import logsumexp
from scipy.sparse.linalg.isolve.iterative import bicgstab

from gpmap.base import SequenceSpace, get_sparse_diag_matrix
from gpmap.plot import init_fig, savefig, arrange_plot, init_single_fig
from gpmap.settings import CMAP
from gpmap.utils import write_pickle, load_pickle
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg import lobpcg
from scipy.sparse.linalg.dsolve.linsolve import spilu


class Visualization(SequenceSpace):
    def __init__(self, length=None, n_alleles=None, alphabet_type='dna',
                 fpath=None, log=None):
        if fpath is not None:
            self.load(fpath, log=log)
        elif length is not None:
            self.init(length, n_alleles, log=log, alphabet_type=alphabet_type)
            self.calc_adjacency()
        else:
            msg = 'Either sequence length or fpath with visualization data'
            msg += ' should be provided'
            raise ValueError(msg)
    
    def get_protein_seq(self, table='Standard'):
        if self.length % 3 != 0:
            msg = 'Only sequences with length multiple of 3 can be translated'
            raise ValueError(msg)
        prot_genotypes = np.array([str(Seq(seq).translate(table=table))
                                   for seq in self.genotypes])
        return(prot_genotypes)
    
    def set_random_function(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.set_function(f=np.random.normal(size=self.n_genotypes))
    
    def set_function(self, f, label=None, codon_table=None, stop_f=-10):
        self.report('Loading function data')

        if codon_table is not None:
            self.prot = pd.Series(self.get_protein_seq(table=codon_table),
                                  index=self.genotypes)
            f = f.reindex(self.prot).fillna(stop_f)
            
        self.f = np.array(f)
        self.label = label
    
    def _calc_delta_f(self, rows, cols):
        return(self.f[cols] - self.f[rows])
    
    def calc_stationary_frequencies(self, Ns, store_sqrt_diag_matrices=True):
        log_freqs = Ns * self.f
        self.log_total = logsumexp(log_freqs)
        self.log_genotypes_stationary_frequencies = log_freqs - self.log_total
        freqs = np.exp(self.log_genotypes_stationary_frequencies)
        
        # Store equilibrium frequencies
        self.genotypes_stationary_frequencies = freqs
        
        if store_sqrt_diag_matrices:
            sqrt_freqs = np.sqrt(self.genotypes_stationary_frequencies)
            self.diag_freq = get_sparse_diag_matrix(sqrt_freqs) 
            self.diag_freq_inv = get_sparse_diag_matrix(1 / sqrt_freqs)
        
        return(freqs)
    
    def calc_stationary_function(self):
        self.fmean = np.sum(self.f * self.genotypes_stationary_frequencies)
        return(self.fmean)
    
    def calc_Ns(self, stationary_function=None, perc=None, tol=1e-8, maxiter=100, 
                max_attempts=10):
        if perc is not None:
            stationary_function = np.percentile(self.f, perc)
        elif stationary_function is None:
            msg = 'Either stationary_function or percentile must be provided'
            raise ValueError(msg)
            
        msg = 'Optimizing Ns to reach a stationary state with mean(f)={}'
        self.report(msg.format(stationary_function))
        
        def f(param):
            Ns = np.exp(param)
            self.calc_stationary_frequencies(Ns, store_sqrt_diag_matrices=False)
            x = self.calc_stationary_function()
            rel_error = (stationary_function - x) ** 2
            return(rel_error)
        
        for _ in range(max_attempts):
            result = minimize(f, x0=0, tol=tol,
                              options={'maxiter': maxiter})
            if f(result.x[0]) < tol:
                break
        else:
            msg = 'Could not find the Ns that yields the desired mean function'
            raise ValueError(msg)

        Ns = np.exp(result.x[0])
        return(Ns)
    
    def _calc_rate(self, delta_f, Ns):
        S = Ns * delta_f
        return(S / (1 - np.exp(-S)))

    def _calc_rate_vector(self, delta_f, Ns):
        rate = np.ones(delta_f.shape[0])
        idxs = np.isclose(delta_f, 0) == False
        rate[idxs] = self._calc_rate(delta_f[idxs], Ns)
        return(rate)
    
    def _check_symmetric(self, m, tol):
        if not (abs(m - m.T)> tol).nnz == 0:
            raise ValueError('Re-scaled rate matrix is not symmetric')
    
    def _ensure_time_reversibility(self, rate_matrix, tol=1e-8):
        '''D_pi^{1/2} Q D_pi^{-1/2} has to be symmetric'''
        self.report('Checking numerical time reversibility')
        sandwich_rate_m = self.diag_freq.dot(rate_matrix).dot(self.diag_freq_inv)
        sandwich_rate_m = (sandwich_rate_m + sandwich_rate_m.T) / 2
        self._check_symmetric(sandwich_rate_m, tol=tol)
        rate_matrix = self.diag_freq_inv.dot(sandwich_rate_m).dot(self.diag_freq)
        return(rate_matrix, sandwich_rate_m)
    
    def calc_rate_matrix(self, Ns, tol=1e-8):
        self.report('Calculating rate matrix with Ns={}'.format(Ns))
        i, j = self.get_neighbor_pairs()
        delta_f = self._calc_delta_f(i, j)
        size = (self.n_genotypes, self.n_genotypes)
        rate_ij = self._calc_rate_vector(delta_f, Ns)
        rate_matrix = csr_matrix((rate_ij, (i, j)), shape=size)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate_matrix.setdiag(-rate_matrix.sum(1).A1)
    
        rate_matrix, sandwich_rate_m = self._ensure_time_reversibility(rate_matrix, tol=tol)
        self.Ns = Ns
        self.rate_matrix = rate_matrix
        self.sandwich_rate_matrix = sandwich_rate_m
    
    def _calc_sandwich_aux_matrix(self):
        '''
        This method calculates an auxiliary matrix to ensure that the calculated
        eigenvalues are close to 0 to avoid numerical problems
        '''
        
        self.report('Calculating A = I - 1/c * T for eigen-decomposition')
        self.upper_bound_eigenvalue = np.abs(self.sandwich_rate_matrix).sum(1).max()
        I = identity(self.n_genotypes) 
        sandwich_aux_matrix = I + 1 / self.upper_bound_eigenvalue * self.sandwich_rate_matrix
        return(sandwich_aux_matrix)
    
    def _check_eigendecomposition(self, matrix, eigenvalues, right_eigenvectors, tol=1e-3):
        self.report('Testing eigendecomposition of T')
        for i in range(self.n_components):
            u = right_eigenvectors[:, i]
            v1 = matrix.dot(u)
            v2 = eigenvalues[i] * u
            abs_err = np.mean(np.abs(v1 - v2)) 
            if abs_err > tol:
                msg = 'Numeric error in eigendecomposition: abs error = {:.5f} > {:.5f}'
                raise ValueError(msg.format(abs_err, tol))
            
        self.report('Eigendecomposition is correct')
    
    def calc_eigendecomposition(self, n_components=10, tol=1e-14, eig_tol=1e-2,
                                maxiter=30):
        n_components = min(n_components, self.n_genotypes-1)
        self.n_components = n_components
        sandwich_aux_matrix = self._calc_sandwich_aux_matrix()
        
        self.report('Calculating {} eigenvalue-eigenvector pairs'.format(n_components))
        v0 = self.diag_freq.dot(self.genotypes_stationary_frequencies)
        lambdas, q = eigsh(sandwich_aux_matrix, n_components,
                           v0=v0, which='LM', tol=tol)
        
        # Reverse order
        lambdas = lambdas[::-1]
        q = np.fliplr(q)
        
        # Store results
        self.eigenvalues = self.upper_bound_eigenvalue * (lambdas - 1)
        self.right_eigenvectors = self.diag_freq_inv.dot(q)
        self._check_eigendecomposition(self.rate_matrix,
                                       self.eigenvalues,
                                       self.right_eigenvectors,
                                       tol=eig_tol)

    def _calc_decay_rates(self):
        return(-1 / self.eigenvalues[1:])
    
    def _calc_projection(self):
        self.report('Scaling projection axis')
        projection = []
        for i in range(self.n_components):
            eigenvalue = self.eigenvalues[i]
            right_eigenvector = self.right_eigenvectors[:, i]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projection.append(right_eigenvector / np.sqrt(-eigenvalue))
                
        colnames = [str(x) for x in np.arange(1, len(projection))]
        self.nodes_df = pd.DataFrame(np.vstack(projection[1:]).T,
                                     index=self.genotypes,
                                     columns=colnames)
        self.nodes_df['f'] = self.f
        self.nodes_df['stationary_freq'] = self.genotypes_stationary_frequencies
        
        i, j = self._get_edges()
        self.edges_df = pd.DataFrame({'i': i, 'j': j})
        
        if hasattr(self, 'prot'):
            self.nodes_df['protein'] = self.prot
        decay_rates = self._calc_decay_rates()
        k = np.arange(1, decay_rates.shape[0] + 1)
        self.decay_df = pd.DataFrame({'k': k, 'decay_rates': decay_rates})
        
    def calc_visualization(self, Ns=None, meanf=None, 
                           perc_function=None, n_components=10,
                           tol=1e-12, eig_tol=1e-2):
        if Ns is None and meanf is None and perc_function is None:
            msg = 'Either Ns or the expected mean function or its percentile '
            msg += 'in equilibrium is required to calculate the rate matrix'
            raise ValueError(msg)
        
        if meanf is not None or perc_function is not None:
            Ns = self.calc_Ns(stationary_function=meanf, perc=perc_function)
            
        self.calc_stationary_frequencies(Ns)
        self.calc_rate_matrix(Ns, tol=tol)
        self.calc_eigendecomposition(n_components, tol=tol, eig_tol=eig_tol)
        self._calc_projection()
    
    def save(self, fpath):
        attrs = ['Ns', 'nodes_df', 'edges_df', 'log_decay_rates', 'f',
                 'genotypes_stationary_frequencies', 'n_alleles', 'length',
                 'alphabet_type']
        data = {attr: getattr(self, attr) for attr in attrs}
        write_pickle(data, fpath)
    
    def write_tables(self, prefix):
        self.nodes_df.to_csv('{}.nodes.csv'.format(prefix))
        self.edges_df.to_csv('{}.edges.csv'.format(prefix), index=False)
        self.decay_df.to_csv('{}.decay_rates.csv'.format(prefix),
                             index=False)
        
    def load(self, fpath, log=None):
        self.log = log
        self.report('Loading visualization data from {}'.format(fpath))
        data = load_pickle(fpath)
        self.init(data['length'], data['n_alleles'],
                  alphabet_type=data['alphabet_type'], log=log)
        for attr, value in data.items():
            setattr(self, attr, value)
    
    def filter_genotypes(self, selected_genotypes):
        # TODO: Think whether to keep this method and re-write
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
        self.genotypes = self.genotypes[selected_genotypes]
        self.seqs = [seq for k, seq in zip(selected_genotypes, self.seqs) if k]
        self.n_genotypes = self.f.shape[0]
    
    
    '''
    Transition path theory methods
    '''
        
    def get_AB_genotypes_idxs(self, genotypes1, genotypes2):
        genotypes1 = self.extend_ambiguous_sequences(genotypes1)
        genotypes2 = self.extend_ambiguous_sequences(genotypes2)
        a = self.get_genotypes_idx(genotypes1)
        b = self.get_genotypes_idx(genotypes2)
        return(a, b)

    def get_noAB_genotypes_idxs(self, a, b):
        idx = np.full(self.n_genotypes, True)
        ab_gts = np.hstack([a, b])
        idx[ab_gts] = False
        no_ab_genotypes = np.where(idx)[0]
        return(no_ab_genotypes)
    
    def calc_committor_probability(self, a, b, tol=1e-16):
        if np.intersect1d(a, b).shape[0] > 0:
            raise ValueError('The two sets of genotypes cannot overlap')
        
        # Select (AUB)^c genotypes
        no_ab = self.get_noAB_genotypes_idxs(a, b)
        
        # Solve system Uq = v 
        partial_rate_matrix = self.rate_matrix[no_ab, :]
        U = partial_rate_matrix[:, no_ab]
        v = -partial_rate_matrix[:, b].sum(1)
        q_reduced, exitCode = bicgstab(U, v, atol=tol)
        
        if exitCode != 0 or np.any(q_reduced < 0) or np.any(q_reduced > 1):
            msg = 'Uq = v solution was not properly found'
            raise ValueError(msg)
        
        # Get complete vector of committor probabilities including A and B     
        q = np.zeros(self.n_genotypes)
        q[no_ab] = q_reduced
        q[b] = 1
        return(q)

    def calc_gt_p_time_reactive_path(self, q):
        log_stationary_freqs = self.log_genotypes_stationary_frequencies
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt_log_p_reactive = log_stationary_freqs + np.log(q) + np.log(1-q)

        log_p_reactive = logsumexp(gt_log_p_reactive)
        gt_log_p_reactive = gt_log_p_reactive - log_p_reactive
        
        p_reactive = np.exp(log_p_reactive)
        gt_p = np.exp(gt_log_p_reactive)
        return(p_reactive, gt_p)
    
    def _get_edges(self, edges=None):
        if edges is None:
            edges = self.get_neighbor_pairs()
        return(edges)

    def calc_edges_flow(self, q, edges=None):
        i, j = self._get_edges(edges)
        stationary_freqs = self.genotypes_stationary_frequencies
        rate_ij = self._calc_rate_vector(self._calc_delta_f(i, j), self.Ns)
        flow = stationary_freqs[i] * (1-q)[i] * rate_ij * q[j]
        return(flow)

    def calc_edges_effective_flow(self, q, edges=None, flows=None):
        edges = self._get_edges(edges)
        i, j = edges
           
        if flows is None:
            stationary_freqs = self.genotypes_stationary_frequencies
            rate_ij = self._calc_rate_vector(self._calc_delta_f(i, j), self.Ns)
            eff_flow = stationary_freqs[i] * rate_ij * (q[j] - q[i])
        else:
            eff_flow = flows[i] - flows[j]
            
        eff_flow[eff_flow < 0] = 0
        return(eff_flow)
    
    def calc_evolutionary_rate(self, a, b, q=None):
        flow = self.calc_edges_flow(a, b, q=q)
        i, j = self.get_neighbor_pairs()
        if a.shape[0] > b.shape[0]:
            sel_gts = set(b)
            sel_idx = np.array([x in sel_gts for x in j])
        else:
            sel_gts = set(a)
            sel_idx = np.array([x in sel_gts for x in i])
        rate = flow[sel_idx].sum()
        return(rate)
    
    def sort_flows(self, i, j, flow):
        if i.shape[0] > 1:
            sorted_idx = np.argsort(flow)
            flow = flow[sorted_idx]
            i = i[sorted_idx]
            j = j[sorted_idx]
        return(i, j, flow)
    
    def get_graph(self, i, j, a, b, m):
        if i.shape[0] > 1:
            edgelist = list(zip(i[m:], j[m:]))
        else:
            edgelist = []
        edgelist.extend([('a', y) for y in a])
        edgelist.extend([(x, 'b') for x in b])
        graph = nx.DiGraph(edgelist)
        return(graph)
    
    def _calc_dynamic_bottleneck(self, i, j, a, b, eff_flow, sort_edges=True):
        if sort_edges:
            i, j, eff_flow = self.sort_flows(i, j, eff_flow)
            
        left, right = 0, eff_flow.shape[0]
        
        if i[-1] in a and j[-1] in b:
            return((i[-1], j[-1]), eff_flow[-1], eff_flow.shape[0]+1)
        
        while right - left > 1:
            m = int((right + left) / 2)
            graph = self.get_graph(i, j, a, b, m)
            if nx.has_path(graph, 'a', 'b'):
                left = m
            else:
                right = m
        
        # Check in case of ties with adjacent edge
        if nx.has_path(graph, 'a', 'b'):
            graph.remove_edge(i[left], j[left]) 
            if nx.has_path(graph, 'a', 'b'):
                left += 1
                right += 1
                    
        bottleneck = (i[left], j[left]) 
        min_flow = eff_flow[left]
        
        return(bottleneck, min_flow, right)
    
    def calc_dynamic_bottleneck(self, a, b, q=None):
        if q is None:
            q = self.calc_committor_probability(a, b)
        eff_flow = self.calc_edges_effective_flow(q)
        i, j = self.get_neighbor_pairs()
        i, j, eff_flow = self.sort_flows(i, j, eff_flow)
        return(self._calc_dynamic_bottleneck(i, j, a, b, eff_flow=eff_flow,
                                             sort_edges=False)[:2])
    
    def get_subgraph_partition(self, graph, digraph, node):
        if nx.has_path(graph, 'a', 'b'):
            msg = 'Path between source and sink found. There may be numerical '
            msg += 'errors when calculating the committor probabilities'
            raise ValueError(msg)
        nodes = nx.node_connected_component(graph, node)
        subgraph = nx.DiGraph(digraph.subgraph(nodes))
        subgraph.remove_node(node)
        i, j = [], []
        for s, t in subgraph.edges:
            i.append(s)
            j.append(t)
        return(np.array(i), np.array(j))
    
    def get_left_and_right_graphs(self, i, j, a, b, m):
        digraph = self.get_graph(i, j, a, b, m)
        graph = nx.Graph(digraph)
        left = self.get_subgraph_partition(graph, digraph, node='a')
        right = self.get_subgraph_partition(graph, digraph, node='b')
        return(left, right)

    def get_flows_dict(self, i, j, a, b, flow):
        flows = dict(zip(zip(i, j), flow))
        for genotype in a:
            flows[('a', genotype)] = np.sum(flow[i == genotype])
        for genotype in b:
            flows[(genotype, 'b')] = np.sum(flow[j == genotype])
        return(flows)
    
    def _calc_representative_pathway(self, i, j, a, b, flows_dict):
        if i.shape[0] == 0 or j.shape[0] == 0:
            return([])
        
        eff_flow = np.array([flows_dict[(s, t)] for s, t in zip(i, j)])
        eff_flow[eff_flow < 0] = 0
        i, j, eff_flow = self.sort_flows(i, j, eff_flow)
        bottleneck = self._calc_dynamic_bottleneck(i, j, a, b, eff_flow=eff_flow)
        b1, b2 = bottleneck[0]
        m = bottleneck[-1]
        left, right = self.get_left_and_right_graphs(i, j, a, b, m)

        if b1 in a:
            left = [b1]
        else:
            left_i, left_j = left
            left = self._calc_representative_pathway(left_i, left_j, a, [b1],
                                                     flows_dict=flows_dict)
        if b2 in b:
            right = [b2]
        else:
            right_i, right_j = right
            right = self._calc_representative_pathway(right_i, right_j, [b2], b,
                                                      flows_dict=flows_dict)
        return(left + right)

    def calc_representative_pathway(self, a, b, flows_dict=None):
        i, j = self._get_edges()
        eff_flows = None
        
        if flows_dict is None:
            q = self.calc_committor_probability(a, b)
            eff_flows = self.calc_edges_effective_flow(q)
            flows_dict = self.get_flows_dict(i, j, a, b, eff_flows)
            
        if eff_flows is None:
            eff_flows = np.array([flows_dict[(s, t)] for s, t in zip(i, j)])
        
        min_flow = self._calc_dynamic_bottleneck(i, j, a, b, 
                                                 eff_flow=eff_flows)[1]
        path = self._calc_representative_pathway(i, j, a=a, b=b,
                                                 flows_dict=flows_dict)
        return(path, min_flow)
    
    def calc_representative_pathways(self, a, b, max_missing_flow_p=1e-4,
                                     max_paths=None):
        i, j = self.get_neighbor_pairs()
        q = self.calc_committor_probability(a, b)
        eff_flows = self.calc_edges_effective_flow(q)
        flows_dict = self.get_flows_dict(i, j, a, b, eff_flows)

        total_flow = np.sum([flows_dict[('a', t)] for t in a])
        flow_sum, flow_cum_p = 0, 0
        n_paths = 0
        min_total_flow_p = 1 - max_missing_flow_p
        
        while flow_cum_p < min_total_flow_p:
            eff_flows = np.array([flows_dict[(s, t)] for s, t in zip(i, j)])
            bottleneck, path_flow = self._calc_dynamic_bottleneck(i, j, a, b, 
                                                                  eff_flow=eff_flows)[:2]
            path = self._calc_representative_pathway(i, j, a=a, b=b,
                                                     flows_dict=flows_dict)
            for s, t in zip(path, path[1:]):
                flows_dict[(s, t)] -= path_flow
            flow_sum += path_flow
            flow_cum_p = flow_sum / total_flow
            flow_p = path_flow / total_flow
            
            yield(bottleneck, path, path_flow, flow_p)
            n_paths += 1
            
            if max_paths is not None and n_paths >= max_paths:
                break
            
    def calc_jump_transition_matrix(self, a, b):
        i, j = self.get_neighbor_pairs()
        delta_f = self._calc_delta_f(i, j)
        size = (self.n_genotypes, self.n_genotypes)
        rates = self._calc_rate_vector(delta_f, self.Ns)
        t = csr_matrix((rates, (i, j)), shape=size)
        rowsums = t.sum(1).A1
        
        q = self.calc_committor_probability(a, b)
        p_ij = rates / rowsums[i]
                
        not_start_at_a = np.array([x not in a for x in i])
        start_at_b = np.array([x in b for x in i])
        p_ij[start_at_b] = 0
        p_ij[not_start_at_a] = p_ij[not_start_at_a] * q[j][not_start_at_a] / q[i][not_start_at_a]
        
        # Add absorbing probabilities at b
        i = np.append(i, b)
        j = np.append(j, b)
        p_ij = np.append(p_ij, np.ones(b.shape[0]))
        
        # Make sparse jump matrix
        jump_matrix = csr_matrix((p_ij, (i, j)), shape=size)
        return(jump_matrix)
    
    def calc_p_return(self, a, b, tol=1e-12, inverse=False):
        self.report('Calculating jump matrix for reactive paths')
        jump_matrix = self.calc_jump_transition_matrix(a, b)
        M = (identity(self.n_genotypes) - jump_matrix).tocsc()
        
        # Select (AUB)^c genotypes
        no_ab_genotypes = self.get_noAB_genotypes_idxs(a, b)
        M = M[no_ab_genotypes, :][:, no_ab_genotypes]
        
        # Init return times
        exp_return_times = np.zeros(self.n_genotypes)
        z = np.zeros(M.shape[0], dtype=np.float32)
        m = []
        
        if inverse:
            self.report('Calculating matrix inverse')
            m = inv(M).diagonal() - 1
        else:
            self.report('Calculating preconditioner')
            ilu = spilu(M, drop_tol=1e-2, fill_factor=50)
            preconditioner = LinearOperator(M.shape, matvec=ilu.solve,
                                            matmat=ilu.solve,
                                            dtype=np.float32)
            
            self.report('Solving for return times')
            for i in tqdm(np.arange(M.shape[0])):
                b = z.copy()
                b[i] = 1
                
                hitting_times, exitCode = bicgstab(M, b, atol=tol,
                                                   M=preconditioner)
                if exitCode != 0 or np.any(hitting_times < 0):
                    msg = 'Uq = v had a non-zero exit code'
                    self.report(msg)
    
                m.append(hitting_times[i])
            m = np.array(m) - 1

        #exp_return_times[no_ab_genotypes] = inv(M).diagonal() - 1
        exp_return_times[no_ab_genotypes] = m
        
        p_return = exp_return_times / (1 + exp_return_times)
        return(p_return)

    def calc_genotypes_flow(self, flows):
        i, j = self._get_edges()
        flow_matrix = csr_matrix((flows, (i, j)),
                                 shape=(self.n_genotypes, self.n_genotypes))
        genotypes_flows = flow_matrix.sum(1).A.flatten()
        return(genotypes_flows)
    
    def calc_gt_p_reactive_path(self, genotypes_flows, p_return, log=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = np.log(genotypes_flows) - np.log(genotypes_flows.max()) + np.log(1 - p_return)

        if not log:
            res = np.exp(res)
            
        return(res)
    
    def calc_normalized_stationary_freq(self, a, b):
        ab = np.append(a, b)
        freqs = self.genotypes_stationary_frequencies.copy()
        freqs[ab] = 0
        freqs = freqs / freqs.sum()
        return(freqs)
    
    def get_dominant_paths_dfs(self, paths):
        bottlenecks = []
        edges = []
        
        for bottleneck, path, flow, p in tqdm(paths):
            bottlenecks.append({'i': bottleneck[0], 'j': bottleneck[1],
                                'flow': flow,  'flow_p': p})
            for i, j in zip(path, path[1:]):
                edges.append({'i': i, 'j': j, 'flow': flow, 'flow_p': p})
        
        bottlenecks_df = pd.DataFrame(bottlenecks)
        bottlenecks_df['from'] = self.genotypes[bottlenecks_df['i']]
        bottlenecks_df['to'] = self.genotypes[bottlenecks_df['j']]
        bottlenecks_df['mutation'] = [','.join(['{}{}{}'.format(x, p+1, y)
                                                for p, (x, y) in enumerate(zip(s1, s2))
                                                if x != y])
                                      for s1, s2 in zip(bottlenecks_df['from'],
                                                        bottlenecks_df['to'])]
        edges_df = pd.DataFrame(edges)
        return(bottlenecks_df, edges_df)
            
    def calc_transition_path_objects(self, genotypes1, genotypes2, tol=1e-12,
                                     max_missing_flow_p=1e-4, max_paths=None,
                                     skip_p_return=False):
        self.report('Finding indexes of starting and end genotypes')
        a, b = self.get_AB_genotypes_idxs(genotypes1, genotypes2)
        norm_freqs = self.calc_normalized_stationary_freq(a, b)
        
        self.report('Calculating committor probabilities')
        q = self.calc_committor_probability(a, b, tol)
        
        self.report('Calculating proportion of time spent in a reactive path')
        m_ab = self.calc_gt_p_time_reactive_path(q)[1]
        
        self.report('Calculating flows through the edges and nodes')
        edges = self.get_neighbor_pairs()
        flows = self.calc_edges_flow(q, edges=edges)
        eff_flows = self.calc_edges_effective_flow(q=q, edges=edges,
                                                   flows=flows)
        genotypes_flows = self.calc_genotypes_flow(flows)
        genotypes_eff_flows = self.calc_genotypes_flow(eff_flows)

        if skip_p_return:
            self.report('Skipping calculation of return probabilities')
        else:
            self.report('Calculating return probabilities before absortion')
            p_return = self.calc_p_return(a, b)
            
            self.report('Calculating proportion of paths including each genotype')
            p_gt_in_path = self.calc_gt_p_reactive_path(genotypes_flows, p_return)
        
        self.report('Calculating dominant evolving paths')
        paths = self.calc_representative_pathways(a, b,
                                                  max_missing_flow_p=max_missing_flow_p,
                                                  max_paths=max_paths)
        
        genotype_df = pd.DataFrame({'f': self.f,
                                    'norm_freqs': norm_freqs,
                                    'q': q, 'm_ab': m_ab,
                                    'flow': genotypes_flows,
                                    'eff_flow': genotypes_eff_flows},
                                   index=self.genotypes)
        
        if not skip_p_return:
            genotype_df['p_in_path'] = p_gt_in_path
            genotype_df['p_return'] = p_return
        
        
        edges_df = pd.DataFrame({'i': edges[0], 'j': edges[1],
                                 'flow': flows, 'eff_flow': eff_flows})
        
        bottlenecks_df, dom_paths_edges = self.get_dominant_paths_dfs(paths)
        
        return({'nodes': genotype_df, 'edges': edges_df,
                'bottleneck': bottlenecks_df,
                'dom_paths_edges': dom_paths_edges})
    
    def write_tpt_objects(self, objects, prefix):
        for label, df in objects.items():
            fpath = '{}.{}.csv'.format(prefix, label)
            self.report('Writting {} into {}'.format(label, fpath))
            df.to_csv(fpath)
        
    
    '''
    Plotting methods
    '''
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
        seqs = self.genotypes
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

    def plot_function_distrib(self, axes, label):    
        sns.distplot(self.f, bins=30, hist=True, kde=False, color='purple',
                     ax=axes)
        xlims = min(self.f), max(self.f)
        arrange_plot(axes, xlabel=label, ylabel='Number of genotypes',
                     xlims=xlims)
    
    def get_eq_functions(self, fmin, fmax, n):
        if fmin is None:
            fmin = self.f.mean() + 0.1 * (self.f.max() - self.f.mean())
        if fmax is None:
            fmax = self.f.mean() + 0.9 * (self.f.max() - self.f.mean())
            
        eq_fs = np.linspace(fmin, fmax, n)
        return(eq_fs)
    
    '''
    Movies methods
    '''
    
    def get_movie_dir(self, dpath):
        if not exists(dpath):
            os.makedirs(dpath)
        return(dpath)
    
    def create_ns_frames(self, dpath, fmin=None, fmax=None,
                         show_edges=True, size=5, cmap=CMAP,
                         label=None, lw=0, n_components=50,
                         nframes=120, force=False):
        eq_fs = self.get_eq_functions(fmin=fmin, fmax=fmax, n=nframes)
        
        fpaths = []
        coords = None
        for i, eq_f in tqdm(list(enumerate(eq_fs))):
            fpath = join(dpath, '{}.png'.format(i))
            fpaths.append(fpath)
            if exists(fpath) and not force:
                continue
            
            self.calc_visualization(meanf=eq_f, n_components=n_components)
            
            fig, subplots = init_fig(1, 2, colsize=3.5, rowsize=3.2)
            
            # Eigenvalues plot
            self.plot_eigenvalues(subplots[0])
            subplots[0].set_title('Stationary f = {:.2f}'.format(eq_f))
            
            # Plot visualization
            coords = self.plot(subplots[1], x=1, y=2, show_edges=show_edges,
                               size=size, cmap=cmap, label=label,
                               force_coords=True, lw=lw, prev_coords=coords)
            subplots[1].set_title('Stationary f = {:.2f}'.format(eq_f))
            savefig(fig, join(dpath, '{}'.format(i)))
            
        return(fpaths, coords)

    def create_rotation_frames(self, dpath, coords, axis=['x'],
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
            savefig(fig, join(dpath, '{}'.format(counter)))
            counter += 1
            
        return(fpaths, c)
    
    def extend_last_frame(self, fpaths, tlast, fps):
        fpaths += [fpaths[-1]] * (tlast * fps)
        return(fpaths)
    
    def save_movie(self, fpaths, dpath, fps, tlast=0, boomerang=False):
        fpaths = self.extend_last_frame(fpaths, tlast, fps)
        if boomerang:
            fpaths = fpaths + fpaths[::-1]
        clip = ImageSequenceClip(fpaths, fps=fps)
        clip.write_videofile('{}.mp4'.format(dpath))
    
    def plot_ns_movie(self, dpath, fmin=None, fmax=None,
                      show_edges=True, size=5, cmap=CMAP,
                      label=None, lw=0, n_components=50,
                      nframes=120, fps=20, boomerang=False, force=False):
        dpath = self.get_movie_dir(dpath)
        fpaths, _ = self.create_ns_frames(dpath, fmin=fmin, fmax=fmax,
                                          show_edges=show_edges, size=size, cmap=cmap,
                                          label=label, lw=lw, n_components=n_components,
                                          nframes=nframes, force=force)
        self.save_movie(fpaths, dpath, fps, tlast=1, boomerang=boomerang)
        
    def plot_rotation_movie(self, dpath,
                            show_edges=True, size=5, cmap=CMAP,
                            label=None, lw=0,  nframes=120, fps=20, lims=None,
                            size_factor=2, force=False, z=3,
                            colors=None):
        self.report('Preparing frames for rotation movie')
        dpath = self.get_movie_dir(dpath=dpath)
        coords = self._get_nodes_coord(axis=[1, 2, 3])
        fpaths, coords = self.create_rotation_frames(dpath=dpath,
                                                     coords=coords, lims=lims,
                                                     show_edges=show_edges,
                                                     size=size, cmap=cmap, label=label,
                                                     lw=lw, nframes=nframes,
                                                     size_factor=size_factor,
                                                     force=force, z=z,
                                                     ylabel='Diffusion axis 2/3',
                                                     zlabel='Diffusion axis 2/3',
                                                     colors=colors)
        self.save_movie(fpaths, dpath, fps, tlast=1, boomerang=False)
        
    
class CodonFitnessLandscape(Visualization):
    def __init__(self,
                 sel_codons=['UCU', 'UCA', 'UCC', 'UCG', 'AGU', 'AGC'],
                 add_variation=False, log=None, seed=None):
        self.init(3, alphabet_type='rna', log=log)
        self.cache_prefix = None
        self.cached_T = False
        self.cached_eigenvectors = False
        self.cached_eigenvalues = False
        self.label = 'codon_landscape'
        self.stop_codons = ['UGA', 'UAA', 'UAG']
        self.sel_codons = sel_codons
        self.add_variation = add_variation
        self.seed = seed
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
            if self.seed is not None:
                np.random.seed(self.seed)
            fitness = fitness + 1 / 10 * np.random.normal(size=len(fitness))
        self.set_function(fitness)


def filter_genotypes(nodes_df, genotypes, edges_df=None):
    size = nodes_df.shape[0]
    nodes_df['index'] = np.arange(size)
    nodes_df = nodes_df.loc[genotypes, :]
    
    if edges_df is not None:
        m = csr_matrix((edges_df.index, (edges_df['i'], edges_df['j'])),
                       shape=(size, size))
        m = m[nodes_df['index'], :][:, nodes_df['index']].tocoo()
        edges_df = edges_df.iloc[m.data, :].copy()
        edges_df['i'] = m.row
        edges_df['j'] = m.col
        return(nodes_df, edges_df)
    else:
        return(nodes_df)
