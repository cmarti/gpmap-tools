import numpy as np
import networkx as nx


def add_source(graph, source_nodes, source_id):
    edge_list = [(source_id, s) for s in source_nodes]
    graph.add_edges_from(edge_list)


def add_target(graph, target_nodes, target_id):
    edge_list = [(t, target_id) for t in target_nodes]
    graph.add_edges_from(edge_list)


def has_path(graph, start, end, start_id="start", end_id="end", rm_aux=True):
    add_source(graph, start, start_id)
    add_target(graph, end, end_id)
    has_path = nx.has_path(graph, start_id, end_id)
    if rm_aux:
        graph.remove_nodes_from([start_id, end_id])
    return has_path


def bipartition_edge_cut_search(graph, start, end, edges):
    """
    This function assumes that edges in the graph are sorted and finds the
    edge with the largest value that disconnects start from end nodes
    """
    left, right = 0, graph.number_of_edges()

    while right - left > 1:
        m = int((right + left) / 2)
        subgraph = graph.edge_subgraph(edges[m:]).copy()

        if has_path(subgraph, start, end):
            left = m
        else:
            right = m

    # Check in case of ties with adjacent edge
    if has_path(subgraph, start, end):
        subgraph.remove_edge(*edges[left])
        if has_path(subgraph, start, end):
            left += 1
    return (left, subgraph)


def calc_bottleneck(graph, start, end):
    edges = sorted(graph.edges, key=lambda x: graph.edges[x]["weight"])
    best_edge_i, best_edge_j = edges[-1]

    if best_edge_i in start and best_edge_j in end:
        bottleneck = (best_edge_i, best_edge_j)
        subgraph = nx.DiGraph()
    else:
        left, subgraph = bipartition_edge_cut_search(graph, start, end, edges)
        bottleneck = edges[left]

    min_flow = graph.edges[bottleneck]["weight"]
    return (bottleneck, min_flow, subgraph)


def get_left_and_right_graphs(
    graph, start, end, start_id="start", end_id="end"
):
    add_source(graph, start, start_id)
    add_target(graph, end, end_id)
    undirected = graph.to_undirected()

    if has_path(undirected, start, end, rm_aux=False):
        msg = "Path between source and sink found when it should not"
        raise ValueError(msg)

    partitions = []
    for node in [start_id, end_id]:
        sel_nodes = nx.node_connected_component(undirected, node)
        partition = graph.subgraph(sel_nodes).copy()
        graph.remove_node(node)
        partition.remove_node(node)
        partitions.append(partition)
    return partitions


def calc_pathway(graph, start, end):
    bottleneck, eff_flow, subgraph = calc_bottleneck(graph, start, end)
    left, right = get_left_and_right_graphs(subgraph, start, end)
    b1, b2 = bottleneck

    if b1 in start:
        left = [b1]
    else:
        left, _ = calc_pathway(left, start, [b1])

    if b2 in end:
        right = [b2]
    else:
        right, _ = calc_pathway(right, [b2], end)

    return (left + right, eff_flow)


def is_better_path(w1, w2, is_sorted=False, only_min=False):
    if not w1:
        if not w2:
            is_better = False
        else:
            is_better = True
    elif not w2:
        is_better = False
    else:
        if only_min:
            return w1 > w2

        else:
            if not is_sorted:
                w1, w2 = sorted(w1), sorted(w2)
            if w1[0] > w2[0]:
                is_better = True
            elif w1[0] < w2[0]:
                is_better = False
            else:
                is_better = is_better_path(w1[1:], w2[1:], is_sorted=True)
    return is_better


def _calc_max_min_path(graph, start, end, attribute, cache={}, only_min=False):
    if start == end:
        return ([], np.inf if only_min else [np.inf])
    else:
        best_w = -np.inf if only_min else [-np.inf]
        for node in graph.predecessors(end):
            path, w = cache.get(node, (None, None))

            if path is None:
                path, w = _calc_max_min_path(
                    graph,
                    start,
                    node,
                    attribute,
                    cache=cache,
                    only_min=only_min,
                )
                cache[node] = path, w

            prev_w = graph.nodes[end].get(attribute, np.inf)
            if only_min:
                w = min(w, prev_w)
            else:
                w = w + [prev_w]

            if is_better_path(w, best_w, only_min=only_min):
                best_path = path + [node]
                best_w = w
        return (best_path, best_w)


def calc_max_min_path(graph, start, end, attribute="weight", only_min=False):
    if not has_path(
        graph, start, end, start_id="start", end_id="end", rm_aux=False
    ):
        msg = "There is no path"
        raise ValueError(msg)
    cache = {}
    path, w = _calc_max_min_path(
        graph,
        "start",
        "end",
        attribute=attribute,
        cache=cache,
        only_min=only_min,
    )
    return (path[1:], w)
