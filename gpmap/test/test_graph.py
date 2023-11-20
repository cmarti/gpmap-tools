#!/usr/bin/env python
import unittest
import sys

import numpy as np
import networkx as nx

from gpmap.src.graph import calc_max_min_path


class GraphTests(unittest.TestCase):
    def test_max_min_path(self):
        nodes = list(range(6))
        ws = [3, 2, 2, 5, 1, 0]
        edges = [(0, 1), (1, 2), (2, 3),
                 (0, 4), (4, 5), (5, 3)]
        
        graph = nx.DiGraph(edges)
        nx.set_node_attributes(graph, {node: {'weight': w}
                                       for node, w in zip(nodes, ws)})
        path, w = calc_max_min_path(graph, [0], [3])
        assert(np.all(path == [0, 1, 2, 3]))
        assert(np.allclose(w, 2))
        
        
if __name__ == '__main__':
    sys.argv = ['', 'GraphTests']
    unittest.main()
