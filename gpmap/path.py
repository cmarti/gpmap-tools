import numpy as np
from _collections import defaultdict
from tqdm import tqdm
import seaborn as sns

from gpmap.visualization import Visualization
from gpmap.inference import VCregression
from scipy.linalg.matfuncs import expm
from scipy.stats.stats import pearsonr
from gpmap.plot_utils import init_fig, savefig


def switch_proposal(path, A, logp):
    if len(path) <= 2:
        return(None, -np.inf)
    
    i = np.random.choice(np.arange(1, len(path)-1))
    start, node, end = path[i-1:i+2]
    common_neighbors = np.logical_and(A[start] == 1, A[end] == 1)
    possible_nodes = np.where(common_neighbors)[0]
    possible_nodes = possible_nodes[possible_nodes != node]
    if possible_nodes.shape[0] == 0:
        return(None, -np.inf)
    
    new_node = np.random.choice(possible_nodes)
    new_path = np.hstack([path[:i], [new_node], path[i+1:]])
    logp_ratio = logp[start, new_node] + logp[new_node, end] - logp[start, node] - logp[node, end]
    return(new_path, logp_ratio)


def node_extension_proposal(path, A, logp):
    if np.random.uniform() < 1/2:
        n_nodes = len(path)
        possible_nodes = np.arange(1, n_nodes-1)
        i = np.random.choice(possible_nodes)
        new_node = path[i]
        new_path = np.hstack([path[:i], [new_node], path[i:]])
        repeated_nodes = sum([x1 == x2 for x1, x2 in zip(new_path, new_path[1:])])
        logp_ratio = logp[new_node, new_node] + np.log(n_nodes-2) - np.log(repeated_nodes)
        
    else:
        repeated_nodes = [i for i, (x1, x2) in enumerate(zip(path, path[1:]))
                          if x1 == x2]
        n_repeated_nodes = len(repeated_nodes)
        
        if n_repeated_nodes > 0:
            selected_node_idx = np.random.choice(repeated_nodes)
            selected_node = path[selected_node_idx]
            new_path = np.hstack([path[:selected_node_idx], path[selected_node_idx+1:]])
            n_nodes = len(new_path)
            logp_ratio = -logp[selected_node, selected_node] + np.log(n_repeated_nodes) - np.log(n_nodes-2)
        else:
            new_path, logp_ratio = None, -np.inf
            
    return(new_path, logp_ratio)


def detour_proposal(path, A, logp):
    if np.random.uniform() < 1/2:
        # Introduce detour
        n_nodes = len(path)
        possible_nodes = np.arange(0, n_nodes-1)
        i = np.random.choice(possible_nodes)
        start, end = path[i:i+2]
        
        n1 = A[start] == 1
        n2 = A[end] == 1
        selected = np.where(A[n1, n2] == 1)[0]
        nodes1 = np.where(n1)[0][selected]
        nodes2 = np.where(n2)[0][selected]
        sel = np.logical_and(np.logical_and(nodes1 != path[0], nodes1 != path[-1]),
                             np.logical_and(nodes2 != path[0], nodes2 != path[-1]))
        nodes1 = nodes1[sel]
        nodes2 = nodes2[sel]
        
        if nodes1.shape[0] == 0:
            return(None, -np.inf)
        
        idx = np.random.choice(np.arange(nodes1.shape[0]))
        node1, node2 = nodes1[idx], nodes2[idx]
        new_path = np.hstack([path[:i+1], [node1, node2], path[i+1:]])
        
        detour_nodes = np.where([A[x1, x4] == 1 and x1 != x2 and x2 != x3 and x3 != x4
                                 for x1, x2, x3, x4 in zip(new_path, new_path[1:], new_path[2:], new_path[3:])])[0]
        n_detours = len(detour_nodes)
        
        logp_ratio = logp[start, node1] + logp[node1, node2] + logp[node2, end] -logp[start, end]
        logp_ratio += np.log(n_nodes-1) - np.log(n_detours)
        # print(new_path, logp_ratio, detour_nodes)
    else:
        # remove detour
        detour_nodes = np.where([A[x1, x4] == 1 and x1 != x2 and x2 != x3 and x3 != x4
                                 for x1, x2, x3, x4 in zip(path, path[1:], path[2:], path[3:])])[0]
        n_detours = len(detour_nodes)
        
        if n_detours > 0:
            # print('detour removal', path, detour_nodes)
            idx = np.random.choice(detour_nodes)
            removed_edges = path[idx:idx+4]
            
            new_path = np.hstack([path[:idx+1], path[idx+3:]])
            n_nodes = len(new_path)
            
            logp_ratio = logp[removed_edges[0], removed_edges[-1]] - np.sum([logp[s, e] for s, e in zip(removed_edges, removed_edges[1:])])
            logp_ratio += np.log(n_detours) - np.log(n_nodes-1)
            # print(path, removed_edges, new_path)
        else:
            new_path, logp_ratio = None, -np.inf
    
    
    return(new_path, logp_ratio)



def initialize_path(start, end, alpha, length):
    different_pos = start != end
    distances = np.sum(different_pos)
    order = np.where(different_pos)[0]
    np.random.shuffle(order)
    
    path = [start]
    for i in order:
        new_node = path[-1].copy()
        new_node[i] = end[i]
        path.append(new_node)
    path = np.vstack(path)
    basis = np.array([alpha ** x for x in range(length)[::-1]])
    path = np.dot(path, basis)
    return(path)


if __name__ == '__main__':
    alpha = 4
    length = 6
    np.random.seed(1)
    space = Visualization(length, n_alleles=alpha, alphabet_type='custom')
    vc = VCregression(length, n_alleles=alpha, alphabet_type='custom')
    f = vc.simulate([1000, 100, 10, 1, 0.1, 0, 0])
    space.load_function(f)
    
    A = space.get_adjacency_matrix().todense().A
    Adiag = A + np.eye(A.shape[0])
    Q = space.calc_transition_matrix()
    P = expm(Q * 0.5).todense()

    # A = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
    #               [1, 0, 0, 1, 0, 1, 0, 0],
    #               [1, 0, 0, 1, 0, 0, 1, 0],
    #               [0, 1, 1, 0, 0, 0, 0, 1],
    #               [1, 0, 0, 0, 0, 1, 1, 0],
    #               [0, 1, 0, 0, 1, 0, 0, 1],
    #               [0, 0, 1, 0, 1, 0, 0, 1],
    #               [0, 0, 0, 1, 0, 1, 1, 0]])
    #
    # P = np.array([[0.9, 0.05, 0.025, 0, 0.025, 0, 0, 0],
    #               [0.05, 0.8, 0, 0.1, 0, 0.05, 0, 0],
    #               [0.025, 0, 0.9, 0.025, 0, 0, 0.05, 0],
    #               [0, 0.1, 0.025, 0.85, 0, 0, 0, 0.025],
    #               [0.025, 0, 0, 0, 0.7, 0.2, 0.075, 0],
    #               [0, 0.05, 0, 0, 0.2, 0.7, 0, 0.05],
    #               [0, 0, 0.05, 0, 0.075, 0, 0.8, 0.075],
    #               [0, 0, 0, 0.025, 0, 0.05, 0.075, 0.85]])
    logp = np.log(P)
    
    
    # path = [0, 1, 3, 7]
    start, end = np.array([0, 0, 0, 0, 0, 0]), np.array([2, 0, 1, 2, 1, 0])
    
    counts = defaultdict(lambda : 0)
    n_iter = 100000
    print(A.shape)
    
    lengths = []
    acceptance_ps = []


    for _ in range(4):
        path = initialize_path(start, end, alpha, length)
        for i in tqdm(range(n_iter)):
            
            # Propose new path
            new_path = path.copy()
            u = np.random.uniform()
            
            if u < 3/4:
                new_path, logp_ratio = switch_proposal(path, Adiag, logp)
            # elif u < 7/8:
            #     new_path, logp_ratio = detour_proposal(path, A, logp)
            else:
                new_path, logp_ratio = node_extension_proposal(path, A, logp)
                
            # Calculate acceptance probability
            acceptance_p = min(np.exp(logp_ratio), 1)
            acceptance_ps.append(acceptance_p)
            # print(new_path, acceptance_p, logp_ratio)
            if np.random.uniform() < acceptance_p:
                path = new_path
        
            if i % 1 == 0:
                counts[tuple(path)] += 1
                lengths.append(len(path))
            # print(start, node, end, possible_nodes, acceptance_p, path)
    
    ps = {path: np.exp(np.sum([logp[s, e] for s, e in zip(path, path[1:])]))
          for path in counts.keys()}
    total_p = np.sum([x for x in ps.values()])
    
    print(len(counts), total_p)
    x, y = [], []
    for path, c in counts.items():
        x.append(c/(4*n_iter))
        y.append(ps[path] / total_p)
    x = np.log10(x)
    print('most common path probability', 10**(x.max()))
    y = np.log10(y)
    # x, y = x[x > -5], y[x > -5]
        
    print(pearsonr(x, y), np.mean(acceptance_ps))
    
    fig, subplots = init_fig(1, 3)
    subplots[0].scatter(x, y)
    subplots[1].plot(lengths)
    sns.histplot(lengths, ax=subplots[2], bins=max(lengths))
    savefig(fig, 'path_sampling')
    