from collections import Counter, defaultdict
from itertools import combinations


def build_shared_pretext_graph(corpus_ids, context_length=1, *, max_successors=None):
    return _build_graph(corpus_ids, context_length, 0, max_successors)


def build_shared_context_graph(corpus_ids, context_length=1, *, max_successors=None):
    return _build_graph(corpus_ids, context_length, context_length, max_successors)


def _build_graph(corpus_ids, left_length, right_length, max_successors):
    if left_length < 1:
        raise ValueError("context_length must be at least 1")

    targets_by_context = defaultdict(set)
    for i in range(left_length, len(corpus_ids) - right_length):
        left = tuple(map(int, corpus_ids[i - left_length : i]))
        right = tuple(map(int, corpus_ids[i + 1 : i + 1 + right_length]))
        target = int(corpus_ids[i])
        targets_by_context[left + right].add(target)

    edge_counts = Counter()
    for targets in targets_by_context.values():
        if max_successors is not None and len(targets) > max_successors:
            continue
        for pair in combinations(sorted(targets), 2):
            edge_counts[pair] += 1

    return edge_counts
