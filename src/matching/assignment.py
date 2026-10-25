"""
Exact linear assignment on a sparse candidate set.

The naive approach builds a dense n x n cost matrix and pads every non-candidate
pair with a huge cost. At 42,000 nodes that is a 13 GB allocation and an O(n^3)
solve, which is what made full-network runs impractical.

Because candidates are limited to a spatial radius, the candidate bipartite graph
falls apart into many small connected components, and an optimal assignment can
NEVER match across components: those pairs carry no finite cost. So the problem
decomposes exactly - solving each component separately gives the identical answer
on matrices a few hundred wide instead of one 42,000 wide.

Measured on Dataset1/Dataset5 at r = 20 m: 26,837 components, largest 223 nodes,
median 2. Peak memory drops from ~13 GB to a few hundred KB.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import (connected_components, maximum_bipartite_matching,
                                  min_weight_full_bipartite_matching)

# A dense solve costs O(n^3) time and n^2 memory, but scipy's implementation is
# fast in practice: a 7,800-node component takes seconds and ~490 MB. The sparse
# solver is exact too, yet on a component that spans the whole zone it can run
# for hours, so dense is preferred up to DENSE_MAX (n^2 x 8 bytes = 1.2 GB at
# 12,000) and sparse is kept only for what does not fit.
DENSE_MAX = 12000


def block_assignment(rows, cols, costs, verbose=False):
    """Minimum-cost one-to-one assignment over the given candidate pairs.

    Args:
        rows, cols: candidate pair identifiers (any hashable/sortable values)
        costs:      cost of each candidate pair (lower = better)

    Returns:
        list of (row_id, col_id) for the matched pairs.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    costs = np.asarray(costs, dtype=float)
    if len(costs) == 0:
        return []

    ur, rinv = np.unique(rows, return_inverse=True)
    uc, cinv = np.unique(cols, return_inverse=True)
    nr, nc = len(ur), len(uc)

    adj = coo_matrix((np.ones(len(rinv)), (rinv, cinv + nr)), shape=(nr + nc, nr + nc))
    ncomp, lab = connected_components(adj + adj.T, directed=False)

    entries = {}
    for k in range(len(rinv)):
        entries.setdefault(lab[rinv[k]], []).append(k)

    if verbose:
        sizes = [len(v) for v in entries.values()]
        print(f"    block assignment: {len(entries):,} components, "
              f"largest {max(sizes):,} pairs, median {int(np.median(sizes))}")

    BIG = float(costs.max()) + 1.0
    matches = []
    for ks in entries.values():
        rs = sorted({rinv[k] for k in ks})
        cs = sorted({cinv[k] for k in ks})
        ri = {r: i for i, r in enumerate(rs)}
        ci = {c: i for i, c in enumerate(cs)}

        if max(len(rs), len(cs)) <= DENSE_MAX:
            n = max(len(rs), len(cs))
            M = np.full((n, n), BIG)
            for k in ks:
                M[ri[rinv[k]], ci[cinv[k]]] = costs[k]
            a, b = linear_sum_assignment(M)
            for i, j in zip(a, b):
                if i < len(rs) and j < len(cs) and M[i, j] < BIG:
                    matches.append((ur[rs[i]], uc[cs[j]]))
            continue

        # Sparse path. min_weight_full_bipartite_matching needs every row to be
        # matchable, which a candidate set does not guarantee, so the rows are
        # first restricted to a maximum-cardinality matchable subset. Costs are
        # shifted to stay strictly positive: a stored zero means "no edge".
        r_idx = np.array([ri[rinv[k]] for k in ks])
        c_idx = np.array([ci[cinv[k]] for k in ks])
        w = costs[np.asarray(ks)] + 1.0
        A = csr_matrix((w, (r_idx, c_idx)), shape=(len(rs), len(cs)))
        m = maximum_bipartite_matching(csr_matrix((np.ones(len(ks)), (r_idx, c_idx)),
                                                  shape=(len(rs), len(cs))),
                                       perm_type='column')
        keep = np.nonzero(m >= 0)[0]
        if len(keep) == 0:
            continue
        sub = A[keep]
        used = np.unique(sub.nonzero()[1])
        sub = sub[:, used]
        a, b = min_weight_full_bipartite_matching(sub)
        for i, j in zip(a, b):
            matches.append((ur[rs[keep[i]]], uc[cs[used[j]]]))
    return matches
