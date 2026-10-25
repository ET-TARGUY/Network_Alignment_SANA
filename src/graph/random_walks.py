"""
Random walk generation for compound graphs.
Uses Numba JIT for the hot inner loop (~10-50x faster than pure Python).

Strategy:
- Node IDs are tuples like ('G1', 123) — not Numba-compatible.
- We encode all nodes as integers, run the Numba kernel on pure int/float arrays,
  then decode back to original tuples.
- Adjacency stored as CSR flat arrays for Numba.
- Graceful fallback to optimized Python if Numba is not installed.
"""
import numpy as np
from tqdm import tqdm

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================
# NUMBA KERNELS
# ============================================================

if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _walks_numba(starts, walk_length, offsets, neighbors, probs_flat, rand_vals):
        """
        Parallel Numba kernel for weighted random walks.

        Args:
            starts      : int64[N]            — start node (int ID) for each walk
            walk_length : int
            offsets     : int64[n_nodes+1]    — CSR row pointers
            neighbors   : int64[n_edges]      — CSR column indices
            probs_flat  : float64[n_edges]    — normalized weights per edge
            rand_vals   : float64[N, L-1]     — pre-sampled uniform values

        Returns:
            out : int64[N, walk_length]  — walks, padded with -1 at dead ends
        """
        n_walks = starts.shape[0]
        out = np.full((n_walks, walk_length), -1, dtype=np.int64)

        for i in prange(n_walks):
            current = starts[i]
            out[i, 0] = current
            for step in range(walk_length - 1):
                s = offsets[current]
                e = offsets[current + 1]
                deg = e - s
                if deg == 0:
                    break
                r = rand_vals[i, step]
                cumsum = 0.0
                chosen = s
                for k in range(deg):
                    cumsum += probs_flat[s + k]
                    if r <= cumsum:
                        chosen = s + k
                        break
                current = neighbors[chosen]
                out[i, step + 1] = current

        return out

    @njit(parallel=True, cache=True)
    def _walks_cena_numba(starts, walk_length,
                          same_offsets, same_neighbors,
                          cross_offsets, cross_neighbors, cross_probs_flat,
                          rand_switch, rand_choice, q):
        """
        Parallel Numba kernel for CENA biased random walks.

        rand_switch : float64[N, L-1]  — compared to q for stay/switch decision
        rand_choice : float64[N, L-1]  — used for neighbor sampling
        """
        n_walks = starts.shape[0]
        out = np.full((n_walks, walk_length), -1, dtype=np.int64)

        for i in prange(n_walks):
            current = starts[i]
            out[i, 0] = current
            for step in range(walk_length - 1):
                s0 = same_offsets[current];  s1 = same_offsets[current + 1]
                c0 = cross_offsets[current]; c1 = cross_offsets[current + 1]
                n_same  = s1 - s0
                n_cross = c1 - c0
                stay = rand_switch[i, step] < q

                if stay:
                    if n_same > 0:
                        idx = int(rand_choice[i, step] * n_same)
                        if idx >= n_same:
                            idx = n_same - 1
                        current = same_neighbors[s0 + idx]
                    elif n_cross > 0:
                        r = rand_choice[i, step]
                        cumsum = 0.0; chosen = c0
                        for k in range(n_cross):
                            cumsum += cross_probs_flat[c0 + k]
                            if r <= cumsum:
                                chosen = c0 + k
                                break
                        current = cross_neighbors[chosen]
                    else:
                        break
                else:
                    if n_cross > 0:
                        r = rand_choice[i, step]
                        cumsum = 0.0; chosen = c0
                        for k in range(n_cross):
                            cumsum += cross_probs_flat[c0 + k]
                            if r <= cumsum:
                                chosen = c0 + k
                                break
                        current = cross_neighbors[chosen]
                    elif n_same > 0:
                        idx = int(rand_choice[i, step] * n_same)
                        if idx >= n_same:
                            idx = n_same - 1
                        current = same_neighbors[s0 + idx]
                    else:
                        break

                out[i, step + 1] = current

        return out


# ============================================================
# GRAPH ENCODING: NetworkX -> integer CSR arrays
# ============================================================

def encode_graph(G):
    """
    Encode a NetworkX graph into integer node IDs and CSR adjacency arrays.

    Returns:
        node_list    : list  — original node IDs, index = integer ID
        node_to_int  : dict  — {original_node: int_id}
        offsets      : int64[n+1]
        nbr_arr      : int64[n_edges]
        prob_arr     : float64[n_edges]
        node_graph   : int64[n]  — 0=G1, 1=G2
    """
    node_list   = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    offsets  = np.zeros(n + 1, dtype=np.int64)
    nbr_buf, prob_buf = [], []

    for i, node in enumerate(node_list):
        nbs = list(G.neighbors(node))
        if not nbs:
            offsets[i + 1] = offsets[i]
            continue
        w = np.array([G[node][nb].get('weight', 1.0) for nb in nbs], dtype=np.float64)
        w /= w.sum()
        offsets[i + 1] = offsets[i] + len(nbs)
        nbr_buf.extend(node_to_int[nb] for nb in nbs)
        prob_buf.extend(w.tolist())

    nbr_arr  = np.array(nbr_buf,  dtype=np.int64)
    prob_arr = np.array(prob_buf, dtype=np.float64)
    node_graph = np.array(
        [0 if G.nodes[node_list[i]]['graph'] == 'G1' else 1 for i in range(n)],
        dtype=np.int64
    )
    return node_list, node_to_int, offsets, nbr_arr, prob_arr, node_graph


def encode_graph_cena(G):
    """
    Encode graph for CENA: separate CSR arrays for same-graph and cross-graph neighbors.
    """
    node_list   = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    same_offsets  = np.zeros(n + 1, dtype=np.int64)
    cross_offsets = np.zeros(n + 1, dtype=np.int64)
    same_buf, cross_buf, cross_prob_buf = [], [], []

    for i, node in enumerate(node_list):
        s_nb, c_nb, c_w = [], [], []
        for nb in G.neighbors(node):
            et = G[node][nb].get('edge_type', 'unknown')
            if et.startswith('within_'):
                s_nb.append(node_to_int[nb])
            elif et == 'cross_graph':
                c_nb.append(node_to_int[nb])
                c_w.append(G[node][nb].get('weight', 1.0))

        same_offsets[i + 1]  = same_offsets[i]  + len(s_nb)
        cross_offsets[i + 1] = cross_offsets[i] + len(c_nb)
        same_buf.extend(s_nb)

        if c_w:
            cw = np.array(c_w, dtype=np.float64)
            cw /= cw.sum()
            cross_buf.extend(c_nb)
            cross_prob_buf.extend(cw.tolist())

    node_graph = np.array(
        [0 if G.nodes[node_list[i]]['graph'] == 'G1' else 1 for i in range(n)],
        dtype=np.int64
    )
    return (node_list, node_to_int,
            same_offsets,  np.array(same_buf,       dtype=np.int64),
            cross_offsets, np.array(cross_buf,       dtype=np.int64),
            np.array(cross_prob_buf, dtype=np.float64),
            node_graph)


# ============================================================
# DECODE: int array -> list of tuple walks
# ============================================================

def decode_walks(walks_int, node_list):
    """Convert Numba int output back to list of original-node walks."""
    result = []
    for row in walks_int:
        walk = []
        for v in row:
            if v == -1:
                break
            walk.append(node_list[v])
        result.append(walk)
    return result


# ============================================================
# STATISTICS
# ============================================================

def _compute_stats(all_walks, node_graph_dict):
    walk_lengths = np.array([len(w) for w in all_walks])
    cross_trans  = np.array([
        sum(1 for i in range(len(w) - 1)
            if node_graph_dict[w[i]] != node_graph_dict[w[i + 1]])
        for w in all_walks
    ])
    return {
        'total_walks':             len(all_walks),
        'mean_length':             float(np.mean(walk_lengths)),
        'median_length':           float(np.median(walk_lengths)),
        'min_length':              int(np.min(walk_lengths)),
        'max_length':              int(np.max(walk_lengths)),
        'mean_cross_transitions':  float(np.mean(cross_trans)),
        'median_cross_transitions':float(np.median(cross_trans)),
        'walks_with_cross':        int(np.sum(cross_trans > 0)),
        'percentage_with_cross':   float(100 * np.sum(cross_trans > 0) / len(all_walks)),
    }


def _print_stats(stats, label="WALK GENERATION STATISTICS"):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"\nGenerated walks: {stats['total_walks']:,}")
    print(f"\nWalk Lengths:")
    print(f"  Mean:   {stats['mean_length']:.2f}")
    print(f"  Median: {stats['median_length']:.0f}")
    print(f"  Min:    {stats['min_length']}")
    print(f"  Max:    {stats['max_length']}")
    print(f"\nCross-Graph Transitions:")
    print(f"  Mean per walk:                 {stats['mean_cross_transitions']:.2f}")
    print(f"  Median per walk:               {stats['median_cross_transitions']:.0f}")
    print(f"  Walks with >= 1 cross-transition: {stats['walks_with_cross']:,} ({stats['percentage_with_cross']:.1f}%)")


# ============================================================
# FALLBACK (no Numba)
# ============================================================

def _build_adj_cache(G):
    adj = {}
    for node in G.nodes():
        nbs = list(G.neighbors(node))
        if not nbs:
            adj[node] = {'neighbors': [], 'probs': np.array([])}
            continue
        w = np.array([G[node][nb].get('weight', 1.0) for nb in nbs], dtype=np.float64)
        w /= w.sum()
        adj[node] = {'neighbors': nbs, 'probs': w}
    return adj


def _fallback_gaussian(G, num_walks, walk_length, verbose):
    adj = _build_adj_cache(G)
    node_graph_dict = {n: G.nodes[n]['graph'] for n in G.nodes()}
    all_walks = []
    for node in tqdm(G.nodes(), desc="Generating walks (fallback)", disable=not verbose):
        for _ in range(num_walks):
            walk = [node]; current = node
            for _ in range(walk_length - 1):
                e = adj[current]
                if not e['neighbors']: break
                current = e['neighbors'][int(np.random.choice(len(e['neighbors']), p=e['probs']))]
                walk.append(current)
            all_walks.append(walk)
    return all_walks, node_graph_dict


def _fallback_cena(G, num_walks, walk_length, q, verbose):
    node_graph_dict = {n: G.nodes[n]['graph'] for n in G.nodes()}
    adj = {}
    for node in G.nodes():
        same, cross, cw = [], [], []
        for nb in G.neighbors(node):
            et = G[node][nb].get('edge_type', 'unknown')
            if et.startswith('within_'):   same.append(nb)
            elif et == 'cross_graph':      cross.append(nb); cw.append(G[node][nb].get('weight', 1.0))
        cp = np.array(cw, dtype=np.float64)
        if len(cp): cp /= cp.sum()
        adj[node] = {'same': same, 'cross': cross, 'cross_probs': cp}

    all_walks = []
    for node in tqdm(G.nodes(), desc="Generating CENA walks (fallback)", disable=not verbose):
        for _ in range(num_walks):
            walk = [node]; current = node
            for _ in range(walk_length - 1):
                e = adj[current]
                s, c, cp = e['same'], e['cross'], e['cross_probs']
                if np.random.random() < q:
                    if s:    current = s[int(np.random.randint(len(s)))]
                    elif c:  current = c[int(np.random.choice(len(c), p=cp))]
                    else:    break
                else:
                    if c:    current = c[int(np.random.choice(len(c), p=cp))]
                    elif s:  current = s[int(np.random.randint(len(s)))]
                    else:    break
                walk.append(current)
            all_walks.append(walk)
    return all_walks, node_graph_dict


# ============================================================
# PUBLIC API
# ============================================================

def generate_random_walks(G, num_walks=10, walk_length=80, verbose=True, n_jobs=-1):
    """
    Generate weighted random walks (Gaussian method).
    Uses Numba JIT if installed, else optimized Python fallback.
    Install Numba: pip install numba
    """
    backend = "Numba JIT + parallel" if NUMBA_AVAILABLE else "Python (pip install numba for ~20x speedup)"
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING RANDOM WALKS  [{backend}]")
        print(f"{'='*60}")
        print(f"  num_walks per node : {num_walks}")
        print(f"  walk_length        : {walk_length}")
        print(f"  Total nodes        : {G.number_of_nodes():,}")
        print(f"  Expected walks     : ~{G.number_of_nodes() * num_walks:,}")

    if not NUMBA_AVAILABLE:
        all_walks, node_graph_dict = _fallback_gaussian(G, num_walks, walk_length, verbose)
        stats = _compute_stats(all_walks, node_graph_dict)
        if verbose: _print_stats(stats)
        return all_walks, stats

    # ── Numba path ───────────────────────────────────────────
    if verbose: print("\n  Encoding graph...")
    node_list, _, offsets, nbr_arr, prob_arr, node_graph_arr = encode_graph(G)
    n_nodes = len(node_list)

    starts    = np.repeat(np.arange(n_nodes, dtype=np.int64), num_walks)
    N         = len(starts)
    rand_vals = np.random.random((N, walk_length - 1))

    if verbose: print(f"  ✓ {n_nodes:,} nodes encoded. Warming up JIT (compiles once)...")

    # Warm-up compile on tiny input
    _walks_numba(starts[:2], 3, offsets, nbr_arr, prob_arr, np.random.random((2, 2)))

    if verbose: print(f"  ✓ JIT ready. Running {N:,} walks in parallel...")

    walks_int = _walks_numba(starts, walk_length, offsets, nbr_arr, prob_arr, rand_vals)

    if verbose: print("  Decoding...")
    all_walks = decode_walks(walks_int, node_list)

    node_graph_dict = {node_list[i]: ('G1' if node_graph_arr[i] == 0 else 'G2') for i in range(n_nodes)}
    stats = _compute_stats(all_walks, node_graph_dict)
    if verbose: _print_stats(stats)
    return all_walks, stats


def generate_random_walks_cena(G, num_walks=10, walk_length=80, q=0.5, verbose=True, n_jobs=-1):
    """
    Generate CENA biased random walks.
    Uses Numba JIT if installed, else optimized Python fallback.
    Install Numba: pip install numba
    """
    backend = "Numba JIT + parallel" if NUMBA_AVAILABLE else "Python (pip install numba for ~20x speedup)"
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING CENA WALKS  [{backend}]")
        print(f"{'='*60}")
        print(f"  num_walks per node : {num_walks}")
        print(f"  walk_length        : {walk_length}")
        print(f"  q (stay prob)      : {q}")
        print(f"  Total nodes        : {G.number_of_nodes():,}")
        print(f"  Expected walks     : ~{G.number_of_nodes() * num_walks:,}")

    if not NUMBA_AVAILABLE:
        all_walks, node_graph_dict = _fallback_cena(G, num_walks, walk_length, q, verbose)
        stats = _compute_stats(all_walks, node_graph_dict)
        if verbose: _print_stats(stats, "CENA WALK GENERATION STATISTICS")
        return all_walks, stats

    # ── Numba path ───────────────────────────────────────────
    if verbose: print("\n  Encoding graph (CENA)...")
    (node_list, _, same_off, same_nbr, cross_off,
     cross_nbr, cross_prob, node_graph_arr) = encode_graph_cena(G)
    n_nodes = len(node_list)

    starts       = np.repeat(np.arange(n_nodes, dtype=np.int64), num_walks)
    N            = len(starts)
    rand_switch  = np.random.random((N, walk_length - 1))
    rand_choice  = np.random.random((N, walk_length - 1))

    if verbose: print(f"  ✓ {n_nodes:,} nodes encoded. Warming up JIT...")

    _walks_cena_numba(
        starts[:2], 3,
        same_off, same_nbr, cross_off, cross_nbr, cross_prob,
        np.random.random((2, 2)), np.random.random((2, 2)), q
    )

    if verbose: print(f"  ✓ JIT ready. Running {N:,} walks in parallel...")

    walks_int = _walks_cena_numba(
        starts, walk_length,
        same_off, same_nbr, cross_off, cross_nbr, cross_prob,
        rand_switch, rand_choice, q
    )

    if verbose: print("  Decoding...")
    all_walks = decode_walks(walks_int, node_list)

    node_graph_dict = {node_list[i]: ('G1' if node_graph_arr[i] == 0 else 'G2') for i in range(n_nodes)}
    stats = _compute_stats(all_walks, node_graph_dict)
    if verbose: _print_stats(stats, "CENA WALK GENERATION STATISTICS")
    return all_walks, stats


# ============================================================
# UNCHANGED UTILITY FUNCTIONS
# ============================================================

def walks_to_sentences(walks):
    """Convert walks (tuple node IDs) to string sentences for Word2Vec."""
    return [[f"{graph}_{node_id}" for graph, node_id in walk] for walk in walks]


def analyze_walk_statistics(walks_str):
    """Analyze detailed step statistics from string walks."""
    total = within_G1 = within_G2 = cross = 0
    for walk in walks_str:
        for i in range(len(walk) - 1):
            total += 1
            gc = walk[i].split('_')[0]
            gn = walk[i + 1].split('_')[0]
            if gc == gn:
                if gc == 'G1': within_G1 += 1
                else:          within_G2 += 1
            else:
                cross += 1
    return {
        'total_steps':     total,
        'within_G1_steps': within_G1,
        'within_G2_steps': within_G2,
        'cross_steps':     cross,
        'within_G1_pct':   100 * within_G1 / total if total else 0,
        'within_G2_pct':   100 * within_G2 / total if total else 0,
        'cross_pct':       100 * cross      / total if total else 0,
    }


# Legacy compatibility
def weighted_random_walk(G, start_node, walk_length):
    adj = _build_adj_cache(G)
    walk = [start_node]; current = start_node
    for _ in range(walk_length - 1):
        e = adj[current]
        if not e['neighbors']: break
        current = e['neighbors'][int(np.random.choice(len(e['neighbors']), p=e['probs']))]
        walk.append(current)
    return walk


def biased_random_walk_cena(G, start_node, walk_length, q=0.5):
    walks, _ = generate_random_walks_cena(G, num_walks=1, walk_length=walk_length, q=q, verbose=False)
    return walks[0] if walks else [start_node]