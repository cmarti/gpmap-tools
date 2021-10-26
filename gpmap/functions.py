import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
import time
import warnings

from itertools import combinations, product
from numpy.linalg import eigh, solve
from numpy.random import choice, normal, randint, random, seed, shuffle, uniform
from scipy.linalg import orth
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, dia_matrix, load_npz, save_npz
from scipy.sparse.linalg import eigsh, gmres
from scipy.special import comb, factorial

U_MAX = 500
PHI_UB, PHI_LB = 100, 0


#
# Preliminary preparation
#


def preliminary_preparation(alpha, l, P, parameters_only=False, with_kernel_basis=True, time_it=False):

    if 1 <= P <= l:
        pass
    elif P == (l+1):
        print('"P" = l+1, the optimal density is equal to the empirical frequency.')
        sys.exit()
    else:
        print('"P" not in the right range.')
        sys.exit()

    # Set start time
    start_time = time.perf_counter()

    # Set global parameters
    set_global_parameters(alpha, l, P, time_it)

    # Prepare L
    if not parameters_only:
        prepare_L()

    # Prepare D kernel basis
    if not parameters_only:
        if with_kernel_basis:
            prepare_D_kernel_basis()

    # Construct D spectrum
    if not parameters_only:
        construct_D_spectrum()

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))


def set_global_parameters(alpha0, l0, P0, time_it0):

    # Set global parameters for later use
    global alpha, l, P, time_it, G, s, bases, sequences, seq_to_pos_converter

    print('Setting global parameters ...')
    start_time = time.perf_counter()

    alpha = alpha0
    l = l0
    P = P0
    time_it = time_it0

    G = alpha**l
    s = comb(l,P) * comb(alpha,2)**P * alpha**(l-P)
    bases = list(range(alpha))
    sequences = list(product(bases, repeat=l))
    seq_to_pos_converter = np.flip(alpha**np.array(range(l)), axis=0)

    if time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))


def prepare_L(path='sparse_matrix/L/'):

    # Set global parameters for later use
    global L_sparse

    # Get list of current sparse matrices
    spm_list = os.listdir(path)

    # If the matrix desired has been made already, load it. Otherwise, construct and save it
    file_name = 'L_alpha'+str(alpha)+'_l'+str(l)+'.npz'

    if file_name in spm_list:

        print('Loading L ...')
        start_time = time.perf_counter()
        L_sparse = load_npz(path+file_name)
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

    else:

        print('Constructing L ...')
        start_time = time.perf_counter()
        L_sparse = construct_L()
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        save_npz(path+file_name, L_sparse)


def construct_L():

    # Generate bases and sequences
    bases = list(range(alpha))
    seqs = list(product(bases, repeat=l))

    # Find indices of L at which L = -1
    row_ids, col_ids, values = [], [], []
    for i in range(G):
        row_ids.append(i)
        col_ids.append(i)
        values.append(l*(alpha-1))
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    row_ids.append(i)
                    col_ids.append(j)
                    values.append(-1)

    # Save L as a sparse matrix
    L_sparse = csr_matrix((values, (row_ids, col_ids)), shape=(G,G))

    # Return
    return L_sparse


def prepare_D_kernel_basis(path='sparse_matrix/D_kernel_basis/'):

    # Set global parameters for later use
    global D_kernel_dim, D_kernel_basis_sparse, D_kernel_basis_orth_sparse

    # Get list of current sparse matrices
    spm_list = os.listdir(path)

    # If the matrix desired has been made already, load it. Otherwise, construct and save it
    file_name1 = 'D_kernel_basis_alpha'+str(alpha)+'_l'+str(l)+'_P'+str(P)+'.npz'
    file_name2 = 'D_kernel_basis_orth_alpha'+str(alpha)+'_l'+str(l)+'_P'+str(P)+'.npz'

    if (file_name1 in spm_list) and (file_name2 in spm_list):

        print('Loading D kernel basis ...')
        start_time = time.perf_counter()
        D_kernel_basis_sparse = load_npz(path+file_name1)
        D_kernel_basis_orth_sparse = load_npz(path+file_name2)
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        D_kernel_dim = 0
        for p in range(P):
            D_kernel_dim += int(comb(l,p) * (alpha-1)**p)

    else:

        print('Constructing D kernel basis ...')
        start_time = time.perf_counter()
        D_kernel_dim, D_kernel_basis_sparse, D_kernel_basis_orth_sparse = construct_D_kernel_basis()
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        save_npz(path+file_name1, D_kernel_basis_sparse)
        save_npz(path+file_name2, D_kernel_basis_orth_sparse)


def construct_D_kernel_basis():

    # Generate bases and sequences
    bases = np.array(list(range(alpha)))
    seqs = np.array(list(product(bases, repeat=l)))

    # Construct D kernel basis
    for p in range(P):

        # Basis of kernel W(0)
        if p == 0:
            W0_dim = 1
            W0_basis = np.ones([G,W0_dim])
            D_kernel_basis = W0_basis

        # Basis of kernel W(1)
        if p == 1:
            W1_dim = l*(alpha-1)
            W1_basis = np.zeros([G,W1_dim])
            for site in range(l):
                W1_basis[:,site*(alpha-1):(site+1)*(alpha-1)] = pd.get_dummies(seqs[:,site], drop_first=True).values
            D_kernel_basis = np.hstack((D_kernel_basis, W1_basis))

        # Basis of kernel W(>=2)
        if p >= 2:
            W2_dim = int(comb(l,p) * (alpha-1)**p)
            W2_basis = np.ones([G,W2_dim])
            site_groups = list(combinations(range(l), p))
            base_groups = list(product(range(1,alpha), repeat=p))  # because we have dropped first base
            col = 0
            for site_group in site_groups:
                for base_group in base_groups:
                    for i in range(p):
                        site, base_idx = site_group[i], base_group[i]-1  # change 'base' to its 'idx'
                        W2_basis[:,col] *= W1_basis[:,site*(alpha-1)+base_idx]
                    col += 1
            D_kernel_basis = np.hstack((D_kernel_basis, W2_basis))

    # Get kernel dimension
    D_kernel_dim = D_kernel_basis.shape[1]

    # Make D kernel basis orthonormal
    D_kernel_basis_orth = orth(D_kernel_basis)

    # Save D_kernel_basis and D_kernel_basis_orth as a sparse matrix
    D_kernel_basis_sparse = csr_matrix(D_kernel_basis)
    D_kernel_basis_orth_sparse = csr_matrix(D_kernel_basis_orth)

    # Return
    return D_kernel_dim, D_kernel_basis_sparse, D_kernel_basis_orth_sparse


def construct_D_spectrum():

    # Set global parameters for later use
    global D_eig_vals, D_multis

    print('Constructing D spectrum ...')
    start_time = time.perf_counter()

    # Compute D eigenvalues and their multiplicity
    D_eig_vals, D_multis = np.zeros(l+1), np.zeros(l+1)
    for k in range(l+1):
        lambda_k = k * alpha
        Lambda_k = 1
        for p in range(P):
            Lambda_k *= lambda_k - p * alpha
        m_k = comb(l,k) * (alpha-1)**k
        D_eig_vals[k], D_multis[k] = Lambda_k/factorial(P), m_k

    if time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))


#
# Data importation
#


def import_data(path, coding_dict, ignore_sites=None):

    # Read in processed data
    df = pd.read_csv(path, sep='\t', names=['sequence', 'count'], dtype=str)

    # Get flags for the sites of interest
    if ignore_sites is not None:
        flags = np.full(l+len(ignore_sites), True)
        flags[ignore_sites] = False

    # Obtain count data
    Ns = np.zeros(G)
    for i in range(len(df)):
        sequence, count = df.loc[i,'sequence'], int(df.loc[i,'count'])
        try:  # sequences with letters not included in coding_dict will be ignored
            seq = [coding_dict[letter] for letter in sequence]
            if ignore_sites is not None:
                seq = np.array(seq)[flags]
            pos = sequence_to_position(seq)
            Ns[pos] = count
        except:
            pass

    # Normalize count data
    N = np.sum(Ns)
    R = Ns / N

    # Save N and R
    data_dict = {'N': int(N), 'R': R}

    # Return
    return data_dict


#
# Data simulation
#


def simulate_data_from_prior(N, a_true, random_seed=None):

    # Set random seed
    seed(random_seed)

    # Simulate phi from prior distribution
    v = normal(loc=0, scale=1, size=G)
    construct_MAT()
    phi_true = np.zeros(G)
    for k in range(P, l+1):
        # eta_k = ? for k < P
        eta_k = np.sqrt(s) / np.sqrt(a_true * D_eig_vals[k])
        solve_b_k(k)
        phi_true += eta_k * W_k_opt(v)

    # Construct Q_true from the simulated phi
    Q_true = np.exp(-phi_true) / np.sum(np.exp(-phi_true))

    # Simulate N data points from Q_true
    data = choice(G, size=N, replace=True, p=Q_true)

    # Obtain count data
    values, counts = np.unique(data, return_counts=True)
    Ns = np.zeros(G)
    Ns[values] = counts

    # Normalize count data
    R = Ns / N

    # Save N and R
    data_dict = {'N': int(N), 'R': R, 'Q_true': Q_true}

    # Return
    return data_dict


def construct_MAT():

    # Set global parameters for later use
    global MAT

    # Construct C
    C = np.zeros([l+1,l+1])
    for i in range(l+1):
        for j in range(l+1):
            if i == j:
                C[i,j] = i * (alpha-2)
            if i == j+1:
                C[i,j] = i
            if i == j-1:
                C[i,j] = (l-j+1) * (alpha-1)

    # Construct D
    D = np.array(np.diag(l*(alpha-1)*np.ones(l+1), 0))

    # Construct B
    B = D - C

    # Construct u
    u = np.zeros(l+1)
    u[0], u[1] = l*(alpha-1), -1

    # Construct MAT column by column
    MAT = np.zeros([l+1,l+1])
    MAT[0,0] = 1
    for j in range(1, l+1):
        MAT[:,j] = np.array(np.mat(B)**(j-1) * np.mat(u).T).ravel()


def solve_b_k(k):

    # Set global parameters for later use
    global b_k

    # Tabulate w_k(d)
    w_k = np.zeros(l+1)
    for d in range(l+1):
        w_k[d] = w(k, d)

    # Solve for b_k
    b_k = solve(MAT, w_k)


def w(k, d):
    ss = 0
    for q in range(l+1):
        ss += (-1)**q * (alpha-1)**(k-q) * comb(d,q) * comb(l-d,k-q)
    return 1/alpha**l * ss


def W_k_opt(v):
    max_power = len(b_k) - 1
    Lsv = np.zeros([G,len(b_k)])
    Lsv[:,0] = b_k[0] * v
    power = 1
    while power <= max_power:
        v = L_opt(v)
        Lsv[:,power] = b_k[power] * v
        power += 1
    Wkv = Lsv.sum(axis=1)
    return Wkv


#
# MAP estimation
#


def estimate_MAP_solution(a, data_dict, phi_initial=None, method='L-BFGS-B', options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Do scaling
    a /= scale_by
    N /= scale_by

    # Set initial guess of phi if it is not provided
    if phi_initial is None:
        Q_initial = np.ones(G) / G
        phi_initial = -np.log(Q_initial)

    # Find the MAP estimate of phi
    if a == 0:

        with np.errstate(divide='ignore'):
            phi_a = -np.log(R)

    elif 0 < a < np.inf:

        res = minimize(fun=S, jac=grad_S, args=(a,N,R), x0=phi_initial, method=method, options=options)
        if not res.success:
            print(res.message)
            print()
        phi_a = res.x

    elif a == np.inf:

        b_initial = D_kernel_basis_orth_sparse.T.dot(phi_initial)
        res = minimize(fun=S_inf, jac=grad_S_inf, args=(N,R), x0=b_initial, method=method, options=options)
        if not res.success:
            print(res.message)
            print()
        b_a = res.x
        phi_a = D_kernel_basis_orth_sparse.dot(b_a)

    else:

        print('"a" not in the right range.')
        sys.exit()

    # Undo scaling
    a *= scale_by
    N *= scale_by

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return phi_a


def trace_MAP_curve(data_dict, resolution=0.1, num_a=20, fac_max=1, fac_min=1e-6, options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Create a container dataframe
    df_map = pd.DataFrame(columns=['a', 'phi'])

    # Compute a = inf end
    print('Computing a = inf ...')
    a_inf = np.inf
    phi_inf = estimate_MAP_solution(a_inf, data_dict, phi_initial=None, options=options, scale_by=scale_by)
    df_map = df_map.append({'a': a_inf, 'phi': phi_inf}, ignore_index=True)

    # Find a_max that is finite and close enough to a = inf
    a_max = s * fac_max
    print('Computing a_max = %f ...' % a_max)
    phi_max = estimate_MAP_solution(a_max, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
    print('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, phi_inf))
    while D_geo(phi_max, phi_inf) > resolution:
        a_max *= 10
        print('Computing a_max = %f ...' % a_max)
        phi_max = estimate_MAP_solution(a_max, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
        print('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, phi_inf))
    df_map = df_map.append({'a': a_max, 'phi': phi_max}, ignore_index=True)

    # Compute a = 0 end
    print()
    print('Computing a = 0 ...')
    a_0 = 0
    phi_0 = estimate_MAP_solution(a_0, data_dict, phi_initial=None, options=options, scale_by=scale_by)
    df_map = df_map.append({'a': a_0, 'phi': phi_0}, ignore_index=True)

    # Find a_min that is finite and close enough to a = 0
    a_min = s * fac_min
    print('Computing a_min = %f ...' % a_min)
    phi_min = estimate_MAP_solution(a_min, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
    print('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, phi_0))
    while D_geo(phi_min, phi_0) > resolution:
        a_min /= 10
        print('Computing a_min = %f ...' % a_min)
        phi_min = estimate_MAP_solution(a_min, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
        print('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, phi_0))
    df_map = df_map.append({'a': a_min, 'phi': phi_min}, ignore_index=True)

    # Compute 0 < a < inf
    if num_a is None:

        # Gross-partition the MAP curve
        print()
        print('Gross-partitioning the MAP curve ...')
        aa = np.geomspace(a_min, a_max, 10)
        phi_initial = phi_inf
        for i in range(len(aa)-2, 0, -1):
            a = aa[i]
            print('Computing a = %f ...' % a)
            phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_initial, options=options, scale_by=scale_by)
            df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)
            phi_initial = phi_inf

        # Fine-partition the MAP curve to achieve desired resolution
        print()
        print('Fine-partitioning the MAP curve ...')
        flag = True
        while flag:
            df_map = df_map.sort_values(by='a')
            aa, phis = df_map['a'].values, df_map['phi'].values
            flag = False
            for i in range(len(df_map)-1):
                a_i, a_j = aa[i], aa[i+1]
                phi_i, phi_j = phis[i], phis[i+1]
                if D_geo(phi_i, phi_j) > resolution:
                    a = np.geomspace(a_i, a_j, 3)[1]
                    print('Computing a = %f ...' % a)
                    phi_initial = phi_inf
                    phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_initial, options=options, scale_by=scale_by)
                    df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)
                    flag = True

    else:

        # Partition the MAP curve into num_a points
        print()
        print('Partitioning the MAP curve into %d points ...' % num_a)
        aa = np.geomspace(a_min, a_max, num_a)
        phi_initial = phi_inf
        for i in range(len(aa)-2, 0, -1):
            a = aa[i]
            print('Computing a_%d = %f ...' % (i, a))
            phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_initial, options=options, scale_by=scale_by)
            df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)
            phi_initial = phi_inf
        df_map = df_map.sort_values(by='a')

    df_map = df_map.sort_values(by='a')
    df_map = df_map.reset_index(drop=True)

    # Report total execution time
    if time_it:
        print('Total execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def compute_log_Es(data_dict, df_map):

    # Set global parameters for later use
    global Delta

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Construct D matrix
    Delta = D_mat()

    # Compute terms (~ "log Z_ratio")
    terms = np.zeros(len(df_map))

    for i in range(len(df_map)):

        a, phi_a = df_map['a'].values[i], df_map['phi'].values[i]

        if a == 0:

            terms[i] = -np.inf

        elif 0 < a < np.inf:

            S_a = S(phi_a, a, N, R)
            H_a = hess_S(phi_a, a, N, R)
            H_a_eig_vals = eigh(H_a)[0]
            terms[i] = - S_a + (G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * np.sum(np.log(H_a_eig_vals))

        elif a == np.inf:

            b_a = D_kernel_basis_orth_sparse.T.dot(phi_a)
            S_a = S_inf(b_a, N, R)
            Ne_sparse = csr_matrix(N*np.exp(-phi_a))
            Ne_ker = ((D_kernel_basis_orth_sparse.T.multiply(Ne_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()
            Ne_ker_eig_vals = eigh(Ne_ker)[0]
            D_row_eig_vals, D_row_multis = D_eig_vals[P:], D_multis[P:]
            terms[i] = - S_a - 1/2 * (np.sum(np.log(Ne_ker_eig_vals)) + np.sum(D_row_multis * np.log(D_row_eig_vals)))

        else:

            print('"a" not in the right range.')
            sys.exit()

    # Compute log_Es
    term_inf = terms[(df_map['a'] == np.inf)]
    log_Es = terms - term_inf

    # Save log_Es
    df_map['log_E'] = log_Es

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def compute_log_Es_bounds(data_dict, df_map):

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Compute the diagonal element of D
    u_0 = np.zeros(G)
    u_0[0] = 1
    D_ii = np.sum(u_0 * D_opt(u_0))

    # Compute terms (~ "log Z_ratio")
    terms_lb, terms_ub = np.zeros(len(df_map)), np.zeros(len(df_map))

    for i in range(len(df_map)):

        a, phi_a = df_map['a'].values[i], df_map['phi'].values[i]

        if a == 0:

            terms_lb[i] = -np.inf
            terms_ub[i] = terms_lb[i]

        elif 0 < a < np.inf:

            S_a = S(phi_a, a, N, R)
            log_det_lb = np.sum(np.log(N * np.exp(-phi_a)))
            log_det_ub = np.sum(np.log(a/s * D_ii + N * np.exp(-phi_a)))
            terms_lb[i] = - S_a + (G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * log_det_ub
            terms_ub[i] = - S_a + (G-D_kernel_dim)/2 * np.log(a/s) - 1/2 * log_det_lb

        elif a == np.inf:

            b_a = D_kernel_basis_orth_sparse.T.dot(phi_a)
            S_a = S_inf(b_a, N, R)
            Ne_sparse = csr_matrix(N*np.exp(-phi_a))
            Ne_ker = ((D_kernel_basis_orth_sparse.T.multiply(Ne_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()
            Ne_ker_eig_vals = eigh(Ne_ker)[0]
            D_row_eig_vals, D_row_multis = D_eig_vals[P:], D_multis[P:]
            terms_lb[i] = - S_a - 1/2 * (np.sum(np.log(Ne_ker_eig_vals)) + np.sum(D_row_multis * np.log(D_row_eig_vals)))
            terms_ub[i] = terms_lb[i]

        else:

            print('"a" not in the right range.')
            sys.exit()

    # Compute log_Es bounds
    term_inf = terms_lb[(df_map['a'] == np.inf)]
    log_Es_lb, log_Es_ub = terms_lb - term_inf, terms_ub - term_inf

    # Save log_Es bounds
    df_map['log_E_lb'], df_map['log_E_ub'] = log_Es_lb, log_Es_ub

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def compute_log_Ls(data_dict, df_map, cv_fold=5, random_seed=None, options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Generate training sets and validation sets
    df_train_data, df_valid_data = split_data(data_dict, cv_fold, random_seed)

    # Compute log_Ls averaged over k folds
    log_Lss = np.zeros([cv_fold,len(df_map)])

    for k in range(cv_fold):

        print('Doing cross validation fold # %d ...' % k)

        N_train, R_train = df_train_data['N'].values[k], df_train_data['R'].values[k]
        N_valid, R_valid = df_valid_data['N'].values[k], df_valid_data['R'].values[k]

        data_dict_train = {'N': N_train, 'R': R_train}
        Ns_valid = N_valid * R_valid

        # For each a, compute Q with training set and compute log_L with validation set
        for i in range(len(df_map)):
            a, phi_a = df_map['a'].values[i], df_map['phi'].values[i]
            phi = estimate_MAP_solution(a, data_dict_train, phi_initial=phi_a, options=options, scale_by=scale_by)
            Q = np.exp(-phi) / np.sum(np.exp(-phi))
            if a == 0:
                N_logQ = np.zeros(G)
                N_flags, Q_flags = (Ns_valid == 0), (Q == 0)
                flags = ~N_flags * Q_flags
                N_logQ[flags] = -np.inf
                flags = ~N_flags * ~Q_flags
                N_logQ[flags] = Ns_valid[flags] * np.log(Q[flags])
                if any(N_logQ == -np.inf):
                    log_L = -np.inf
                else:
                    log_L = np.sum(N_logQ)
            else:
                log_L = np.sum(Ns_valid * np.log(Q))
            log_Lss[k,i] = log_L

    log_Ls = log_Lss.mean(axis=0)

    # Save log_Ls
    df_map['log_L'] = log_Ls

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def split_data(data_dict, cv_fold, random_seed=None):

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Generate raw data
    raw_data = generate_raw_data(data_dict, random_seed)

    # Reshape raw data into an array of k=cv_fold rows
    remainder = N % cv_fold
    row_len = int((N-remainder) / cv_fold)
    raw_data_array = np.reshape(raw_data[:N-remainder], (cv_fold, row_len))

    # If some raw data are left, create a dictionary to map each raw datum left to one k
    if remainder != 0:
        raw_data_left = raw_data[-remainder:]
        left_dict = {}
        for k, raw_datum_left in enumerate(raw_data_left):
            left_dict[k] = raw_datum_left
        left_dict_keys = list(left_dict.keys())

    # Split raw data into training sets and validation sets
    df_train_data, df_valid_data = pd.DataFrame(columns=['N', 'R']), pd.DataFrame(columns=['N', 'R'])

    for k in range(cv_fold):

        # Get training data
        ids = list(range(cv_fold))
        ids.remove(k)
        train_data = raw_data_array[ids,:].reshape(-1)
        if remainder != 0:
            for id in ids:
                if id in left_dict_keys:
                    train_data = np.append(train_data, left_dict[id])
        values, counts = np.unique(train_data, return_counts=True)
        Ns_train = np.zeros(G)
        Ns_train[values] = counts
        N_train = np.sum(counts)
        R_train = Ns_train / N_train
        df_train_data = df_train_data.append({'N': N_train, 'R': R_train}, ignore_index=True)

        # Get validation data
        valid_data = raw_data_array[k,:]
        if remainder != 0:
            if k in left_dict_keys:
                valid_data = np.append(valid_data, left_dict[k])
        values, counts = np.unique(valid_data, return_counts=True)
        Ns_valid = np.zeros(G)
        Ns_valid[values] = counts
        N_valid = np.sum(counts)
        R_valid = Ns_valid / N_valid
        df_valid_data = df_valid_data.append({'N': N_valid, 'R': R_valid}, ignore_index=True)

    # Return
    return df_train_data, df_valid_data


def compute_rms_log_p_association(phi, p):
    if any(phi == np.inf):
        rms_log_p_association = np.inf
    else:
        Dphi = phi.copy()
        for i in range(p):
            Dphi = L_opt(Dphi, i)
        Dphi /= factorial(p)
        s_p = comb(l,p) * comb(alpha,2)**p * alpha**(l-p)
        rms_log_p_association = np.sqrt(abs(1/s_p * np.sum(phi * Dphi)))
    return rms_log_p_association


def compute_marginal_probability(phi):
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    Q_ker = D_kernel_basis_sparse.T.dot(Q)
    df_marginal_probs = pd.DataFrame(columns=['sites', 'bases', 'probability'])
    c = 0
    for p in range(P):
        site_groups = list(combinations(range(l), p))
        base_groups = list(product(range(1,alpha), repeat=p))  # because we have dropped first base
        for site_group in site_groups:
            for base_group in base_groups:
                df_marginal_probs = df_marginal_probs.append({'sites': site_group, 'bases': base_group,
                                                              'probability': Q_ker[c]}, ignore_index=True)
                c += 1
    return df_marginal_probs


#
# Posterior sampling
#


def posterior_sampling(phi, a, data_dict, num_samples, method, args, random_seed=None):

    # Set start time
    start_time = time.perf_counter()

    # Set random seed
    seed(random_seed)

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Perform posterior sampling
    if method == 'hmc':
        phi_initial, phi_samples, acceptance_rates = hamiltonian_monte_carlo(phi, a, N, R, num_samples, args)

    else:
        print('"method" not recognized.')
        sys.exit()

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return phi_initial, phi_samples, acceptance_rates


def hamiltonian_monte_carlo(phi_star, a_star, N, R, num_samples, args):

    # Get HMC parameters
    e = args['e']
    L = args['L']
    Le = args['Le']
    L_max = args['L_max']
    m = args['m']
    f = args['f']
    window = args['window']
    gamma_old = args['gamma_old']
    gamma_new = args['gamma_new']
    perturbation = args['perturbation']
    num_warmup = args['num_warmup']
    num_thinning = args['num_thinning']

    num_draws = num_warmup + num_samples * num_thinning

    # Compute scales
    u_0 = np.zeros(G)
    u_0[0] = 1
    D_ii = np.sum(u_0 * D_opt(u_0))
    H_ii = a_star/s * D_ii + N * np.exp(-phi_star)
    scales = 1 / np.sqrt(H_ii)

    # Other settings
    phi_initial = phi_star + 2*(random(G)-0.5) * perturbation * scales
    a = a_star

    warnings.filterwarnings('error')

    if a == 0:

        phi_initial, phi_samples, acceptance_rates = None, None, None

    elif 0 < a < np.inf:

        # Initiate iteration
        phi_old = phi_initial.copy()
        S_phi_old = S(phi_old, a, N, R)
        grad_S_phi_old = grad_S(phi_old, a, N, R)
        psi_old = normal(loc=0, scale=np.sqrt(m), size=G)

        # HMC iterations
        phi_samples, acceptance_rates = np.zeros([G,num_samples]), []
        num_acceptance = 0
        k, c = 1, 0
        while k <= num_draws:

            try:

                # Update psi
                psi = normal(loc=0, scale=np.sqrt(m), size=G)
                psi_old = f * psi_old + np.sqrt(1-f**2) * psi

                # Set multiple stepsizes
                es = e * scales

                # Leapfrog steps
                phi, psi = phi_old.copy(), psi_old.copy()
                psi -= 1/2 * es * grad_S_phi_old
                for leapfrog_step in range(L-1):
                    phi += es / m * psi
                    grad_S_phi = grad_S(phi, a, N, R)
                    psi -= es * grad_S_phi
                phi += es / m * psi
                grad_S_phi = grad_S(phi, a, N, R)
                psi -= 1/2 * es * grad_S_phi
                psi *= -1

                # Compute probability ratio
                S_phi = S(phi, a, N, R)
                log_P = - S_phi - 1/2 * np.sum(psi**2) / m
                log_P_old = - S_phi_old - 1/2 * np.sum(psi_old**2) / m
                log_r = log_P - log_P_old

                # Accept/Reject proposed phi
                if log_r > np.log(random()):
                    phi_old = phi.copy()
                    S_phi_old = S_phi.copy()
                    grad_S_phi_old = grad_S_phi.copy()
                    psi_old = psi.copy()
                    num_acceptance += 1
                else:
                    phi_old = phi_old.copy()
                    S_phi_old = S_phi_old.copy()
                    grad_S_phi_old = grad_S_phi_old.copy()
                    psi_old = psi_old.copy()

                # Save phi and negate psi
                if (k > num_warmup) and (k % num_thinning == 0):
                    phi_samples[:,c] = phi_old
                    c += 1
                psi_old *= -1

                # Adapt e and L
                if k % window == 0:
                    acceptance_rate = num_acceptance / window
                    e_new = tune_hmc_stepsize(e, acceptance_rate)
                    e = (e**gamma_old * e_new*gamma_new)**(1/(gamma_old+gamma_new))
                    L = min(int(Le/e), L_max)
                    acceptance_rates.append(acceptance_rate)
                    num_acceptance = 0

                k += 1

            except Warning:

                phi_old = phi_old.copy()
                S_phi_old = S_phi_old.copy()
                grad_S_phi_old = grad_S_phi_old.copy()
                psi_old = psi_old.copy()
                e *= 0.95
                L = min(int(Le/e), L_max)

    elif a == np.inf:

        phi_initial, phi_samples, acceptance_rates = None, None, None

    else:

        print('"a" not in the right range.')
        sys.exit()

    # Return
    return phi_initial, phi_samples, acceptance_rates


def tune_hmc_stepsize(e, acceptance_rate):
    if acceptance_rate < 0.001:
        e *= 0.1
    elif 0.001 <= acceptance_rate < 0.05:
        e *= 0.5
    elif 0.05 <= acceptance_rate < 0.2:
        e *= 0.7
    elif 0.2 <= acceptance_rate < 0.5:
        e *= 0.8
    elif 0.5 <= acceptance_rate < 0.6:
        e *= 0.9
    elif 0.6 <= acceptance_rate <= 0.7:
        e *= 1
    elif 0.7 < acceptance_rate <= 0.8:
        e *= 1.1
    elif 0.8 < acceptance_rate <= 0.9:
        e *= 1.5
    elif 0.9 < acceptance_rate <= 0.95:
        e *= 2
    elif 0.95 < acceptance_rate:
        e *= 3
    return e


def compute_R_hat(multi_phi_samples0):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    num_subchains, len_subchain = 2*num_chains, int(num_samples_per_chain/2)

    # Re-shape multi_phi_samples into a shape of (num_subchains, G, len_subchain)
    a = []
    for k in range(num_chains):
        a.append(multi_phi_samples[k,:,:len_subchain])
        a.append(multi_phi_samples[k,:,len_subchain:])
    multi_phi_samples_reshaped = np.array(a)

    # Compute R_hat for each component of phi
    R_hats = []
    for i in range(G):

        # Collect the (sub)chains of samples of phi_i
        i_collector = np.zeros([len_subchain,num_subchains])
        for j in range(num_subchains):
            i_collector[:,j] = multi_phi_samples_reshaped[j,i,:]

        # Compute the between-(sub)chain variance
        mean_0 = i_collector.mean(axis=0)
        mean_01 = mean_0.mean()
        B = len_subchain/(num_subchains-1) * np.sum((mean_0 - mean_01)**2)

        # Compute the within-(sub)chain variance
        s2 = np.zeros(num_subchains)
        for j in range(num_subchains):
            s2[j] = 1/(len_subchain-1) * np.sum((i_collector[:,j] - mean_0[j])**2)
        W = s2.mean()

        # Estimate the marginal posterior variance
        var = (len_subchain-1)/len_subchain * W + 1/len_subchain * B

        # Compute R_hat
        R_hat = np.sqrt(var/W)

        # Save
        R_hats.append(R_hat)

    # Return
    return np.array(R_hats)


def plot_trajectory(i, multi_phi_samples0, phi_map, colors, save_fig=False):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    # Plot trajectory of the i-th component of phi
    plt.figure(figsize=(6,5))
    for k in range(num_chains):
        plt.plot(range(num_samples_per_chain), multi_phi_samples[k,i,:], color=colors[k], alpha=0.4, zorder=1)

    if phi_map is not None:
        plt.hlines(y=phi_map[i], xmin=0, xmax=num_samples_per_chain, color='black', zorder=2)

    plt.xlabel('Sample #', fontsize=14)
    plt.ylabel(r'$\phi_{%d}$'%i, fontsize=16)
    plt.xlim(0, num_samples_per_chain)
    if save_fig:
        plt.savefig('trajectory_%d'%i, dpi=200)
    plt.show()


def combine_samples(multi_phi_samples0):

    # Copy the multi_phi_samples
    multi_phi_samples = multi_phi_samples0.copy()

    num_chains, G, num_samples_per_chain = \
        multi_phi_samples.shape[0], multi_phi_samples.shape[1], multi_phi_samples.shape[2]

    # Combine phi samples
    phi_samples = multi_phi_samples[0,:,:]
    for k in range(1, num_chains):
        phi_samples = np.hstack((phi_samples, multi_phi_samples[k,:,:]))

    # Return
    return phi_samples


def plot_distribution(i, phi_samples_list, phi_map, num_bins, colors, save_fig=False):

    # Plot distribution of the i-th component of phi
    plt.figure(figsize=(6,5))
    hist_max = 0
    for k in range(len(phi_samples_list)):
        hist, bin_edges = np.histogram(phi_samples_list[k][i,:], bins=num_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = np.linspace(bin_edges[0]+bin_width/2, bin_edges[-1]-bin_width/2, len(bin_edges)-1)
        plt.bar(bin_centers, hist, width=bin_width, color=colors[k], alpha=0.5, edgecolor=colors[k], zorder=1)
        hist_max = max(hist_max, max(hist))

    if phi_map is not None:
        plt.vlines(x=phi_map[i], ymin=0, ymax=1.2*hist_max, color='black', zorder=2)

    plt.xlabel(r'$\phi_{%d}$'%i, fontsize=16)
    plt.ylim(0, 1.2*hist_max)
    if save_fig:
        plt.savefig('distribution_%d'%i, dpi=200)
    plt.show()


#
# Analysis tools: computing pairwise association
#


def compute_log_ORs(phi, site_i, site_j, site_i_mut=None, site_j_mut=None, condition={}, coding_dict=None):

    # If coding dictionary is provided, convert letters to codes
    if coding_dict is not None:
        if (site_i_mut is not None) and (site_j_mut is not None):
            site_i_mut = [coding_dict[letter] for letter in site_i_mut]
            site_j_mut = [coding_dict[letter] for letter in site_j_mut]
        for key in condition.keys():
            value = [coding_dict[letter] for letter in condition[key]]
            condition[key] = value

    # Generate bases
    bases = list(range(alpha))

    # Get background sites
    bg_sites = list(set(range(l)) - {site_i,site_j})

    # Get allowable bases for each background site
    bg_sites_bases = []
    for bg_site in bg_sites:
        if bg_site in condition.keys():
            bg_sites_bases.append(condition[bg_site])
        else:
            bg_sites_bases.append(bases)

    # Generate background sequences
    bg_seqs = product(*bg_sites_bases)

    # Generate all possible 2x2 faces that can be formed by site i (mut) and site j (mut)
    if (site_i_mut is not None) and (site_j_mut is not None):
        faces = [list(product(site_i_mut, site_j_mut))]
    else:
        base_pairs = list(combinations(bases, 2))
        base_pair_products = list(product(base_pairs, base_pairs))
        faces = []
        for base_pair_product in base_pair_products:
            faces.append(list(product(*base_pair_product)))

    # For each background sequence, compute log_OR on all faces formed by site i (mut) and site j (mut)
    log_ORs, associated_seqs = [], []
    for bg_seq in bg_seqs:
        for face in faces:
            face_phis, face_seqs = [], []
            for k in range(4):
                face_vertex_k_seq = np.full(l, -1, dtype=int)
                face_vertex_k_seq[bg_sites] = bg_seq
                face_vertex_k_seq[[site_i,site_j]] = face[k]
                face_vertex_k_pos = sequence_to_position(face_vertex_k_seq)
                face_phis.append(phi[face_vertex_k_pos])
                face_seqs.append(face_vertex_k_seq)
            log_ORs.append(-((face_phis[3]-face_phis[1])-(face_phis[2]-face_phis[0])))
            associated_seqs.append(face_seqs)

    # If coding dictionary is provided, convert codes to letters
    if coding_dict is not None:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        TMP = []
        for seqs in associated_seqs:
            tmp = []
            for seq in seqs:
                tmp.append(''.join([rev_coding_dict[code] for code in seq]))
            TMP.append(tmp)
        associated_seqs = TMP

    # Save log_ORs and associated sequences in a dataframe
    df_log_ORs = pd.DataFrame()
    df_log_ORs['log_OR'], df_log_ORs['associated_seqs'] = log_ORs, associated_seqs
    df_log_ORs = df_log_ORs.sort_values(by='log_OR', ascending=False).reset_index(drop=True)

    # Return
    return df_log_ORs


#
# Analysis tools: making visualization
#


def make_visualization(Q, markov_chain, K=20, tol=1e-9, reuse_Ac=False, path='sparse_matrix/Ac/'):

    # Set start time
    Start_time = time.perf_counter()

    # If reuse existing A and c, load them. Otherwise, construct A and c from scratch and save them
    if reuse_Ac:

        print('Loading A and c ...')
        start_time = time.perf_counter()
        A_sparse = load_npz(path+'A.npz')
        c = joblib.load(path+'c.pkl')
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

    else:

        print('Constructing A and c ...')
        start_time = time.perf_counter()
        A_sparse, c = construct_Ac(Q, markov_chain)
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        save_npz(path+'A.npz', A_sparse)
        joblib.dump(c, path+'c.pkl')

    # Compute the dominant eigenvalues and eigenvectors of A
    print('Computing dominant eigenvalues and eigenvectors of A ...')
    start_time = time.perf_counter()
    eig_vals_tilt, eig_vecs_tilt = eigsh(A_sparse, K, which='LM', tol=tol)
    if time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))

    # Check accuracy of the eigenvalues and eigenvectors of A
    df_check = pd.DataFrame(columns=['eigenvalue', 'colinearity', 'max_difference'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        Au = A_sparse.dot(u)
        max_diff = abs(Au-lda*u).max()
        Au /= np.sqrt(np.sum(Au**2))
        colin = np.sum(Au*u)
        df_check = df_check.append({'eigenvalue': lda, 'colinearity': colin, 'max_difference': max_diff}, ignore_index=True)
    df_check = df_check.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Obtain the eigenvalues and eigenvectors of T, and use them to construct visualization coordinates
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(G,G))
    df_visual = pd.DataFrame(columns=['eigenvalue', 'coordinate'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        if lda < 1:
            eig_val = c * (lda - 1)
            eig_vec = Diag_Q_inv_sparse.dot(u)
            coordinate = eig_vec / np.sqrt(-eig_val)
            df_visual = df_visual.append({'eigenvalue': eig_val, 'coordinate': coordinate}, ignore_index=True)
        else:
            df_visual = df_visual.append({'eigenvalue': 0, 'coordinate': np.full(G,np.nan)}, ignore_index=True)
    df_visual = df_visual.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - Start_time))

    # Return
    return df_visual, df_check


def construct_Ac(Q, markov_chain):

    # Choose a model for the reversible Markov chain
    if markov_chain == 'evolutionary':
        T_ij = T_evolutionary
    elif markov_chain == 'Metropolis':
        T_ij = T_Metropolis
    elif markov_chain == 'power_law':
        T_ij = T_power_law
    else:
        print('markov_chain "model" not recognized.')
        sys.exit()

    # Generate bases and sequences
    bases = list(range(alpha))
    seqs = list(product(bases, repeat=l))

    # Construct transition matrix (or rate matrix) T
    row_ids, col_ids, values = [], [], []
    for i in range(G):
        tmp = []
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    # Blocking transitions between +1 & -1 state for 'aneuploidy data' subsets
                    # k = np.where(np.array(seqs[i]) != np.array(seqs[j]))[0][0]
                    # if (seqs[i][k]==1 and seqs[j][k]==2) or (seqs[i][k]==2 and seqs[j][k]==1):
                    #     value = 0
                    # else:
                    value = T_ij(Q[i], Q[j])
                    row_ids.append(i)
                    col_ids.append(j)
                    values.append(value)
                    tmp.append(value)
        row_ids.append(i)
        col_ids.append(i)
        values.append(-np.sum(tmp))

    # Save T as a sparse matrix
    T_sparse = csr_matrix((values, (row_ids, col_ids)), shape=(G,G))

    # Construct a symmetric matrix T_tilt from T
    Diag_Q_sparse = dia_matrix((np.sqrt(Q), np.array([0])), shape=(G,G))
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(G,G))
    T_tilt_sparse = Diag_Q_sparse.dot(T_sparse * Diag_Q_inv_sparse)

    # Choose the value of c
    c = 0
    for i in range(G):
        sum_i = abs(T_tilt_sparse[i,i])
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    sum_i += abs(T_tilt_sparse[i,j])
        c = max(c, sum_i)

    # Construct A and save it as a sparse matrix
    I_sparse = dia_matrix((np.ones(G), np.array([0])), shape=(G,G))
    A_sparse = I_sparse + 1/c * T_tilt_sparse

    # Return
    return A_sparse, c


def T_evolutionary(Q_i, Q_j, par=1):
    if Q_i == Q_j:
        return 1
    else:
        return par * (np.log(Q_j)-np.log(Q_i)) / (1 - np.exp(-par * (np.log(Q_j)-np.log(Q_i))))


def T_Metropolis(Q_i, Q_j):
    if Q_j > Q_i:
        return 1
    else:
        return Q_j/Q_i


def T_power_law(Q_i, Q_j, par=1/2):
    return Q_j**par / Q_i**(1-par)


def get_nodes(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual['coordinate'].values[kx]*xflip, df_visual['coordinate'].values[ky]*yflip

    # Save the coordinates
    df_nodes = pd.DataFrame()
    df_nodes['node'], df_nodes['x'], df_nodes['y'] = range(G), x, y

    # Return
    return df_nodes


def get_edges(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual['coordinate'].values[kx]*xflip, df_visual['coordinate'].values[ky]*yflip

    # Generate bases and sequences
    bases = list(range(alpha))
    seqs = list(product(bases, repeat=l))

    # Get coordinates of all edges (i > j)
    nodes_i, nodes_j, edges = [], [], []
    for i in range(G):
        for site in range(l):
            for base in bases:
                seq_i = np.array(seqs[i])
                if base != seq_i[site]:
                    seq_i[site] = base
                    j = sequence_to_position(seq_i)
                    if i > j:
                        nodes_i.append(i)
                        nodes_j.append(j)
                        edges.append([(x[i],y[i]), (x[j],y[j])])

    # Save the coordinates
    df_edges = pd.DataFrame()
    df_edges['node_i'], df_edges['node_j'], df_edges['edge'] = nodes_i, nodes_j, edges

    # Return
    return df_edges


#
# Analysis tools: others
#


def find_local_max(phi, data_dict=None, coding_dict=None, threshold=0):

    # Get counts if data dictionary is provided
    if data_dict is not None:
        N, R = data_dict['N'], data_dict['R']
        Ns = N * R

    # Generate bases and sequences
    bases = list(range(alpha))
    seqs = list(product(bases, repeat=l))

    # Find local maxima
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    local_max_seqs, local_max_probs, local_max_cnts = [], [], []
    for i in range(G):
        if Q[i] > threshold:
            js = []
            for site in range(l):
                for base in bases:
                    seq_i = np.array(seqs[i])
                    if base != seq_i[site]:
                        seq_i[site] = base
                        j = sequence_to_position(seq_i)
                        js.append(j)
            if all(np.greater(np.ones(l*(alpha-1))*Q[i], np.take(Q,js))):
                local_max_seqs.append(seqs[i])
                local_max_probs.append(Q[i])
                if data_dict is not None:
                    local_max_cnts.append(int(Ns[i]))

    # If coding dictionary is provided, convert codes to letters
    if coding_dict is not None:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        tmp = []
        for seq in local_max_seqs:
            tmp.append(''.join([rev_coding_dict[code] for code in seq]))
        local_max_seqs = tmp

    # Save local maxima in a dataframe
    df_local_max = pd.DataFrame()
    df_local_max['sequence'], df_local_max['probability'] = local_max_seqs, local_max_probs
    if data_dict is not None:
        df_local_max['count'] = local_max_cnts
    df_local_max = df_local_max.sort_values(by='probability', ascending=False).reset_index(drop=True)

    # Return
    return df_local_max


def compute_entropy(phi):
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    if any(Q == 0):
        flags = (Q != 0)
        entropy = -np.sum(Q[flags] * np.log2(Q[flags]))
    else:
        entropy = -np.sum(Q * np.log2(Q))
    return entropy


#
# Utility functions
#


def sequence_to_position(seq, coding_dict=None):
    if coding_dict is None:
        return int(np.sum(seq * seq_to_pos_converter))
    else:
        tmp = [coding_dict[letter] for letter in seq]
        return int(np.sum(tmp * seq_to_pos_converter))


def position_to_sequence(pos, coding_dict=None):
    if coding_dict is None:
        return sequences[pos]
    else:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        tmp = sequences[pos]
        return ''.join([rev_coding_dict[code] for code in tmp])


def D_geo(phi1, phi2):
    Q1 = np.exp(-phi1) / np.sum(np.exp(-phi1))
    Q2 = np.exp(-phi2) / np.sum(np.exp(-phi2))
    x = min(np.sum(np.sqrt(Q1 * Q2)), 1)
    return 2 * np.arccos(x)


def generate_raw_data(data_dict, random_seed=None):

    # Set random seed
    seed(random_seed)

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Generate raw data
    Ns = N * R
    raw_data = []
    for i in range(G):
        raw_data.extend([i]*int(round(Ns[i])))
    raw_data = np.array(raw_data)

    # Make sure the amount of raw data is correct
    if len(raw_data) != N:
        print('"raw_data" not correctly generated.')
        sys.exit()

    # Shuffle raw data
    shuffle(raw_data)

    # Return
    return raw_data


def sample_from_data(N, data_dict, random_seed=None):

    # Set random seed
    seed(random_seed)

    # Generate raw data
    raw_data = generate_raw_data(data_dict, random_seed)

    # Sample N points from raw data
    sample = choice(raw_data, size=N, replace=False)

    # Turn sample into count data
    values, counts = np.unique(sample, return_counts=True)
    Ns = np.zeros(G)
    Ns[values] = counts

    # Make sure the amount of sample is correct
    if np.sum(Ns) != N:
        print('"sample" not correctly drawn from data.')

    # Save N and R
    R = Ns / N
    sample_dict = {'N': int(N), 'R': R}

    # Return
    return sample_dict


#
# Basic functions
#


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)


def S(phi, a, N, R):
    S1 = a/(2*s) * np.sum(phi * D_opt(phi))
    S2 = N * np.sum(R * phi)
    S3 = N * np.sum(safe_exp(-phi))
    regularizer = 0
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_UB)[flags]**2)
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_LB)[flags]**2)
    return S1 + S2 + S3 + regularizer


def grad_S(phi, a, N, R):
    grad_S1 = a/s * D_opt(phi)
    grad_S2 = N * R
    grad_S3 = N * safe_exp(-phi)
    regularizer = np.zeros(G)
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_UB)[flags]
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_LB)[flags]
    return grad_S1 + grad_S2 - grad_S3 + regularizer


def hess_S(phi, a, N, R):
    hess_S1 = a/s * Delta
    hess_S2 = N * np.diag(safe_exp(-phi), 0)
    return np.array(hess_S1 + hess_S2)


def S_inf(b, N, R):
    phi = D_kernel_basis_orth_sparse.dot(b)
    S_inf1 = N * np.sum(R * phi)
    S_inf2 = N * np.sum(safe_exp(-phi))
    regularizer = 0
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_UB)[flags]**2)
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_LB)[flags]**2)
    return S_inf1 + S_inf2 + regularizer


def grad_S_inf(b, N, R):
    phi = D_kernel_basis_orth_sparse.dot(b)
    grad_S_inf1 = N * R
    grad_S_inf2 = N * safe_exp(-phi)
    regularizer = np.zeros(G)
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_UB)[flags]
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_LB)[flags]
    return D_kernel_basis_orth_sparse.T.dot(grad_S_inf1 - grad_S_inf2 + regularizer)


def hess_S_inf(b, N, R):
    phi = D_kernel_basis_orth_sparse.dot(b)
    hess_S_inf_sparse = csr_matrix(N*np.exp(-phi))
    return ((D_kernel_basis_orth_sparse.T.multiply(hess_S_inf_sparse)).dot(D_kernel_basis_orth_sparse)).toarray()


def L_opt(phi, p=0):
    return L_sparse.dot(phi) - p*alpha * phi


def D_opt(phi):
    Dphi = phi.copy()
    for p in range(P):
        Dphi = L_opt(Dphi, p)
    return Dphi/factorial(P)


def L_mat():
    L = np.zeros([G,G])
    for i in range(G):
        for j in range(i+1):
            u_i, u_j = np.zeros(G), np.zeros(G)
            u_i[i], u_j[j] = 1, 1
            L[i,j] = np.sum(u_i * L_opt(u_j))
            L[j,i] = L[i,j]
    return L


def D_mat():
    D = np.zeros([G,G])
    for i in range(G):
        for j in range(i+1):
            u_i, u_j = np.zeros(G), np.zeros(G)
            u_i[i], u_j[j] = 1, 1
            D[i,j] = np.sum(u_i * D_opt(u_j))
            D[j,i] = D[i,j]
    return D
