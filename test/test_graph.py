#!/usr/bin/env python
import unittest
import sys

import numpy as np
import networkx as nx

from gpmap.graph import calc_max_min_path, is_better_path


class GraphTests(unittest.TestCase):
    def test_is_better_path(self):
        w1, w2 = [2, 2, 2], [2, 1, 2]
        assert(is_better_path(w1, w2))
        
        w1, w2 = [2, 1, 1], [2, 1, 2]
        assert(not is_better_path(w1, w2))
        
        w1, w2 = [2, 1, 2], [2, 1, 2]
        assert(not is_better_path(w1, w2))
        
        w1, w2 = [2, 1], [2, 1, 1]
        assert(is_better_path(w1, w2))
        
        w1, w2 = [1, 1], [2, 2, 2]
        assert(not is_better_path(w1, w2))
    
    def test_max_min_path(self):
        nodes = list(range(6))
        edges = [(0, 1), (1, 2), (2, 3),
                 (0, 4), (4, 5), (5, 3)]
        graph = nx.DiGraph(edges)

        ws = [3, 2, 2, 5, 1, 0]        
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(nodes, ws)})
        path, w = calc_max_min_path(graph, [0], [3])
        assert(np.all(path == [0, 1, 2, 3]))
        assert(np.allclose(min(w), 2))
        
        ws = [3, 1, 1, 5, 3, 3]
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(nodes, ws)})
        path, w = calc_max_min_path(graph, [0], [3])
        assert(np.all(path == [0, 4, 5, 3]))
        assert(np.allclose(min(w), 3))
        
        ws = [3, 2, 2, 3, 2, 2]
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(nodes, ws)})
        path, w = calc_max_min_path(graph, [0], [5])
        assert(np.all(path == [0, 4, 5]))
        assert(np.allclose(min(w), 2))
        
        # Choose longer better path
        edges = [(0, 1), (1, 2), (2, 3),
                 (0, 4), (4, 5), (3, 5)]
        graph = nx.DiGraph(edges)
        ws = [3, 3, 3, 3, 2, 3]
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(nodes, ws)})
        path, w = calc_max_min_path(graph, [0], [5])
        assert(np.all(path == [0, 1, 2, 3, 5]))
        assert(np.allclose(min(w), 3))
        
        # Ensure it still works if we only store the minimum value
        path, w = calc_max_min_path(graph, [0], [5], only_min=True)
        assert(np.all(path == [0, 1, 2, 3, 5]))
        assert(np.allclose(w, 3))
        
        
if __name__ == '__main__':
    sys.argv = ['', 'GraphTests']
    unittest.main()
