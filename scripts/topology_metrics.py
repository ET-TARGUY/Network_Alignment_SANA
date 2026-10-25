#!/usr/bin/env python3
"""
Topological homogeneity of the real sources (Notion #13 / #14).

No alignment method is run: this only describes the graphs, over the WHOLE of
each network (Z1 includes its halo).

Per network
  deg1..deg4p   share of nodes of degree 1, 2, 3 and >= 4 (%)
  H_deg         Shannon entropy of the degree distribution (bits); lower means
                more homogeneous connectivity
  clustering    average local clustering coefficient
  wl_shared_h   share of nodes whose Weisfeiler-Leman colour after h rounds is
                shared with at least one other node (%): nodes a structural
                method limited by 1-WL cannot tell apart from some other node
  wl_ratio_h    number of distinct WL colours / |V|
  bc_gini       Gini coefficient of (sampled) betweenness centrality; bc_cv its
                coefficient of variation. Hop-count betweenness, estimated from
                k random sources (fixed seed)

Usage (repo root, main .venv)
  python scripts/topology_metrics.py                    # all networks
  python scripts/topology_metrics.py --only Dataset1 IGN
  python scripts/topology_metrics.py --k 200 --wl-iters 3
  python scripts/topology_metrics.py --no-betweenness   # fast pass

Outputs
  results/topology_metrics.csv
  results/topology_table.tex   (table rows, ready to paste)
"""
import argparse
import csv
import math
import pickle
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results"

NETWORKS = {
    "Dataset1": PROCESSED / "Dataset1" / "Graph.pkl",
    "Dataset2": PROCESSED / "Dataset2" / "Graph.pkl",
    "Dataset3": PROCESSED / "Dataset3" / "Graph.pkl",
    "Dataset4": PROCESSED / "Dataset4" / "Graph.pkl",
    "Dataset5": PROCESSED / "Dataset5" / "Graph.pkl",
    "Z1": PROCESSED / "Z1__identity_L0_s01" / "Dataset1" / "Graph.pkl",
    "IGN": PROCESSED / "IGN" / "Graph.pkl",
    "OSM": PROCESSED / "OSM" / "Graph.pkl",
}


def load(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    G = nx.Graph(G.to_undirected() if G.is_directed() else G)   # simple, undirected
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def degree_stats(G):
    deg = np.array([d for _, d in G.degree()])
    n = len(deg)
    share = lambda m: 100.0 * m.sum() / n
    counts = np.array(list(Counter(deg.tolist()).values()), dtype=float) / n
    H = float(-(counts * np.log2(counts)).sum())
    return {"deg1": share(deg == 1), "deg2": share(deg == 2), "deg3": share(deg == 3),
            "deg4p": share(deg >= 4), "mean_degree": float(deg.mean()), "H_deg": H}


def wl_colours(G, iters):
    """1-WL colour refinement; colour 0 = degree. Returns one Counter per round."""
    col = {v: G.degree(v) for v in G}
    out = []
    for _ in range(iters):
        sig = {v: (col[v], tuple(sorted(col[u] for u in G[v]))) for v in G}
        ids = {}
        col = {v: ids.setdefault(s, len(ids)) for v, s in sig.items()}
        out.append(Counter(col.values()))
    return out


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(NETWORKS))
    ap.add_argument("--wl-iters", type=int, default=3)
    ap.add_argument("--k", type=int, default=200, help="betweenness sample size (sources)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-betweenness", action="store_true")
    args = ap.parse_args()

    rows = []
    for name in args.only or NETWORKS:
        path = NETWORKS[name]
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        t0 = time.time()
        G = load(path)
        n = G.number_of_nodes()
        print(f"\n{name}: |V| {n:,}  |E| {G.number_of_edges():,}")
        r = {"network": name, "nodes": n, "edges": G.number_of_edges()}
        r.update(degree_stats(G))
        print(f"  degree 1/2/3/>=4: {r['deg1']:.1f} / {r['deg2']:.1f} / {r['deg3']:.1f} / "
              f"{r['deg4p']:.1f} %   H = {r['H_deg']:.3f} bits")
        r["clustering"] = nx.average_clustering(G)
        print(f"  clustering {r['clustering']:.4f}")
        for h, cnt in enumerate(wl_colours(G, args.wl_iters), 1):
            shared = sum(c for c in cnt.values() if c > 1)
            r[f"wl_shared_{h}"] = 100.0 * shared / n
            r[f"wl_ratio_{h}"] = len(cnt) / n
            print(f"  WL h={h}: {len(cnt):,} colours, {r[f'wl_shared_{h}']:.1f}% of nodes share theirs")
        if not args.no_betweenness:
            k = min(args.k, n)
            tb = time.time()
            bc = np.array(list(nx.betweenness_centrality(G, k=k, seed=args.seed,
                                                         normalized=True).values()))
            r["bc_gini"] = gini(bc)
            r["bc_cv"] = float(bc.std() / bc.mean()) if bc.mean() > 0 else 0.0
            r["bc_k"] = k
            print(f"  betweenness (k={k}, {time.time() - tb:.0f}s): Gini {r['bc_gini']:.3f}  CV {r['bc_cv']:.2f}")
        r["seconds"] = round(time.time() - t0, 1)
        rows.append(r)
        del G

    if not rows:
        return
    OUT.mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 999)
    with open(OUT / "topology_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows)

    h = args.wl_iters
    lines = [f"% generated by scripts/topology_metrics.py (WL h={h}, betweenness k={args.k})",
             "% Network & deg1 & deg2 & deg3 & deg>=4 (%) & H_deg & clustering & WL-shared (%) & BC Gini"]
    for r in rows:
        bc = f"{r['bc_gini']:.2f}" if "bc_gini" in r else "--"
        lines.append(f"{r['network']} & {r['deg1']:.1f} & {r['deg2']:.1f} & {r['deg3']:.1f} & "
                     f"{r['deg4p']:.1f} & {r['H_deg']:.2f} & {r['clustering']:.3f} & "
                     f"{r[f'wl_shared_{h}']:.1f} & {bc} \\\\")
    (OUT / "topology_table.tex").write_text("\n".join(lines) + "\n")
    print(f"\n-> {OUT / 'topology_metrics.csv'}\n-> {OUT / 'topology_table.tex'}")


if __name__ == "__main__":
    main()
