import numpy as np
from _collections import defaultdict


if __name__ == '__main__':
    # space = Visualization(3, n_alleles=2, alphabet_type='custom')
    # space.load_function([])
    #
    # A = space.get_adjacency_matrix().todense()
    # Q = space.calc_transition_matrix()

    A = np.array([[0, 1, 1, 0, 1, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 0, 1, 1, 0]])

    P = np.array([[0.9, 0.05, 0.025, 0, 0.025, 0, 0, 0],
                  [0.05, 0.8, 0, 0.1, 0, 0.05, 0, 0],
                  [0.025, 0, 0.9, 0.025, 0, 0, 0.05, 0],
                  [0, 0.1, 0.025, 0.85, 0, 0, 0, 0.025],
                  [0.025, 0, 0, 0, 0.7, 0.2, 0.075, 0],
                  [0, 0.05, 0, 0, 0.2, 0.7, 0, 0.05],
                  [0, 0, 0.05, 0, 0.075, 0, 0.8, 0.075],
                  [0, 0, 0, 0.025, 0, 0.05, 0.075, 0.85]])
    logp = np.log(P)
    
    
    path = [0, 1, 3, 7]
    
    counts = defaultdict(lambda : 0)
    n_iter = 100000
    
    for i in range(n_iter):
        
        # Propose new path
        new_path = path
        i = np.random.choice(np.arange(1, len(path)-1))
        start, node, end = path[i-1:i+2]
        possible_nodes = np.where(np.logical_and(A[start] == 1, A[end] == 1))[0]
        possible_nodes = possible_nodes[possible_nodes != node]
        new_node = np.random.choice(possible_nodes)
        new_path = path[:i] + [new_node] + path[i+1:]
        
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
        
        
        
    