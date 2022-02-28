import numpy as np
from _collections import defaultdict
from gpmap.visualization import Visualization
from gpmap.inference import VCregression
from scipy.linalg.matfuncs import expm


if __name__ == '__main__':
    space = Visualization(3, n_alleles=3, alphabet_type='custom')
    vc = VCregression(3, n_alleles=3, alphabet_type='custom')
    f = vc.simulate([100, 10, 1, 0.1])
    space.load_function(f)
    
    A = space.get_adjacency_matrix().todense()
    Q = space.calc_transition_matrix()
    P = expm(Q * 0.5).todense()

#     A = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
#                   [1, 0, 0, 1, 0, 1, 0, 0],
#                   [1, 0, 0, 1, 0, 0, 1, 0],
#                   [0, 1, 1, 0, 0, 0, 0, 1],
#                   [1, 0, 0, 0, 0, 1, 1, 0],
#                   [0, 1, 0, 0, 1, 0, 0, 1],
#                   [0, 0, 1, 0, 1, 0, 0, 1],
#                   [0, 0, 0, 1, 0, 1, 1, 0]])
# 
#     P = np.array([[0.9, 0.05, 0.025, 0, 0.025, 0, 0, 0],
#                   [0.05, 0.8, 0, 0.1, 0, 0.05, 0, 0],
#                   [0.025, 0, 0.9, 0.025, 0, 0, 0.05, 0],
#                   [0, 0.1, 0.025, 0.85, 0, 0, 0, 0.025],
#                   [0.025, 0, 0, 0, 0.7, 0.2, 0.075, 0],
#                   [0, 0.05, 0, 0, 0.2, 0.7, 0, 0.05],
#                   [0, 0, 0.05, 0, 0.075, 0, 0.8, 0.075],
#                   [0, 0, 0, 0.025, 0, 0.05, 0.075, 0.85]])
    logp = np.log(P)
    
    
    path = [0, 1, 4, 13]
    
    counts = defaultdict(lambda : 0)
    n_iter = 100
    
    for i in range(n_iter):
        
        # Propose new path
        new_path = path
        u = np.random.uniform()
        
        if u < 1/3:
            # node change
            i = np.random.choice(np.arange(1, len(path)-1))
            start, node, end = path[i-1:i+2]
            possible_nodes = np.where(np.logical_and(A[start] == 1, A[end] == 1))[0]
            possible_nodes = possible_nodes[possible_nodes != node]
            new_node = np.random.choice(possible_nodes)
            new_path = path[:i] + [new_node] + path[i+1:]
        elif u < 3/3:
            # node inclusion
            i = np.random.choice(np.arange(1, len(path)))
            start, end = path[i-1:i+1]
            forbiden_nodes = set([start, end, path[0], path[-1]])
            print(i, start, end, forbiden_nodes)
            possible_nodes = set(np.where(np.logical_and(A[start] == 1, A[end] == 1))[0])
            print(i, start, end, possible_nodes)
            possible_nodes = possible_nodes - forbiden_nodes
            print(i, start, end, possible_nodes)
            if not possible_nodes:
                continue
            new_node = np.random.choice(possible_nodes)
            new_path = path[:i] + [new_node] + path[i:]
            print(new_path)
        
        else:
            # node removal: shorten path
            i = np.random.choice(np.arange(1, len(path)))
        continue
            
        
        # Calculate acceptance probability
        p_ratio = np.exp(logp[start, new_node] + logp[new_node, end] - logp[start, node] - logp[node, end])
        acceptance_p = min(p_ratio, 1)
        if np.random.uniform() < acceptance_p:
            path = new_path
    
        counts[tuple(path)] += 1
        # print(start, node, end, possible_nodes, acceptance_p, path)
    
    ps = {path: np.exp(np.sum([logp[s, e] for s, e in zip(path, path[1:])]))
          for path in counts.keys()}
    total_p = np.sum([x for x in ps.values()])
    
    for path, c in counts.items():
        print(path, c/n_iter, ps[path] / total_p)
        
        
        
    