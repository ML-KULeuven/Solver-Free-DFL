import random
import time
from collections import deque

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_unique_solutions(solutions):
    def round_list(lst, precision):
        return tuple(round(x, precision) for x in lst)

    list_of_tuples = [round_list(lst, 2) for lst in solutions]
    unique_lists = set(list_of_tuples)
    return len(unique_lists)


def find_partial_lexipos(A, required_rows):
    """
    Find an m-×-m selection and ordering of columns of A so that
    only the rows in `required_rows` are lexicopositive.

    Required for initial application of the TNP-rule, as this rule requires a lexicopositive basis to start with.

    Geue, F. (1993). An improved N-tree algorithm for the enumeration of all neighbors of a degenerate vertex.
    Annals of Operations Research, 46(2), 361-391.

    Returns:
        - A list of m column indices (ordered), or None if no solution or timeout after 3 seconds.
    """
    start_time = time.time()
    m, n = A.shape
    R = list(required_rows)

    # Columns with positive and negative entries for each required row
    pos_cols = {i: list(np.where(A[i] > 0)[0]) for i in R}
    neg_cols = {i: set(np.where(A[i] < 0)[0]) for i in R}

    # Sort required rows by fewest positive options, to prune sooner
    R_order = sorted(R, key=lambda i: len(pos_cols[i]))

    pivots = {}
    graph = {}  # ordering constraints between columns

    def has_cycle():
        visited = {}

        def dfs(u):
            visited[u] = 1
            for v in graph.get(u, ()):
                if visited.get(v, 0) == 1:
                    return True
                if visited.get(v, 0) == 0 and dfs(v):
                    return True
            visited[u] = 2
            return False

        for u in graph:
            if visited.get(u, 0) == 0 and dfs(u):
                return True
        return False

    def backtrack_req(k):
        # Check for timeout
        if time.time() - start_time > 3:
            return None

        # All required rows assigned
        if k == len(R_order):
            return True
        row = R_order[k]
        for c in pos_cols[row]:
            # Tentatively assign column c as pivot for this row
            old_graph = {u: set(v) for u, v in graph.items()}
            pivots[row] = c
            graph.setdefault(c, set())

            # Add constraints among required pivots
            for prev_row in R_order[:k]:
                prev_c = pivots[prev_row]
                # If c is negative in prev_row, prev_c -> c
                if c in neg_cols[prev_row]:
                    graph[prev_c].add(c)
                # If prev_c is negative in this row, c -> prev_c
                if prev_c in neg_cols[row]:
                    graph[c].add(prev_c)

            if not has_cycle():
                result = backtrack_req(k + 1)
                if result is None:  # Timeout occurred
                    return None
                if result:
                    return True

            # Backtrack
            graph.clear()
            graph.update(old_graph)
            pivots.pop(row, None)
        return False

    # Assign required-row pivots
    if backtrack_req(0) is None:  # Check for timeout
        return None
    if not backtrack_req(0):
        return None

    # Build set of distinct pivot columns
    P = set(pivots.values())

    # Add extra columns to reach m total distinct columns
    extras_needed = m - len(P)
    extras = [c for c in range(n) if c not in P][:extras_needed]
    if len(extras) < extras_needed:
        return None

    # Combine into full column set S
    S = list(P) + extras

    # Add graph nodes for extras and edges to prevent negatives before pivots
    for c in extras:
        graph.setdefault(c, set())
        for i in R:
            if c in neg_cols[i]:
                graph[pivots[i]].add(c)

    # Topological sort on S
    in_deg = {c: 0 for c in S}
    for u in graph:
        for v in graph[u]:
            if v in in_deg:
                in_deg[v] += 1

    q = deque([c for c in S if in_deg[c] == 0])
    order = []
    while q:
        # Check for timeout
        if time.time() - start_time > 3:
            return None

        u = q.popleft()
        order.append(u)
        for v in graph.get(u, ()):
            if v in in_deg:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)

    return order if len(order) == m else None


def search_for_transition_column(A, basic_indices, non_basic_indices, sol):
    """
    Find an initial transition column to start application of the TNP-rule.

    For more information on transition columns and the TNP-rule, see:

    Geue, F. (1993). An improved N-tree algorithm for the enumeration of all neighbors of a degenerate vertex.
    Annals of Operations Research, 46(2), 361-391.
    """
    while True:
        A_basic = A[:, basic_indices]
        dirs = np.linalg.solve(-A_basic, A[:, non_basic_indices])
        basic_var_values = sol[basic_indices]

        # Add numerical stability threshold
        EPSILON = 1e-10

        i = np.random.choice(len(non_basic_indices))
        entering_var_index = non_basic_indices[i]

        dir = dirs[:, i]

        indices_dir_neg = np.where(dir < -EPSILON)[0]
        basic_var_values_dir_neg = basic_var_values[indices_dir_neg]
        negative_dir_values = dir[indices_dir_neg]
        ratios = -basic_var_values_dir_neg / negative_dir_values

        if ratios.size != 0:
            min_ratio = np.min(ratios)
            min_ratio_indices = indices_dir_neg[np.where(ratios == min_ratio)[0]]
            if min_ratio > 0:
                break
            else:
                min_ratio_index = np.random.choice(min_ratio_indices)
                leaving_var_index = basic_indices[min_ratio_index]

                new_basic_indices = np.copy(basic_indices)
                new_basic_indices[min_ratio_index] = entering_var_index
                new_non_basic_indices = np.copy(non_basic_indices)
                new_non_basic_indices[i] = leaving_var_index

                basic_indices = new_basic_indices
                non_basic_indices = new_non_basic_indices

    return basic_indices, non_basic_indices


def is_lexicofeasible(A):
    """
    Check whether each row of `A` satisfies lexicographic feasibility.

    A matrix is lexicographically feasible if the first non-zero entry
    in every row is strictly positive. This helper routine is used as
    a check before applying the TNP-rule.
    """
    for row in A:
        # Find the first non-zero element
        non_zero_elements = row[row != 0]
        if (len(non_zero_elements) > 0 and non_zero_elements[0] <= 0):
            return False
    return True
