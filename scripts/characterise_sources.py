#!/usr/bin/env python3
"""
Step 1 - characterise the real sources (Tables 5.1 and 6.1).

No alignment method is run: this only describes the data, over the WHOLE of
each network.

Per network
  |V|, |E|     from data/processed/<name>/Graph.pkl
  s            median distance from a node to its nearest neighbour in the same network
  extent       bounding box, km
Per pair (V, V')
  ratio        |V'| / |V|
  co-located   share of V nodes with exactly one V' node within 0.1 m
               (the proximity ground-truth rule of step 09)
  displacement mean distance between MUTUAL nearest-neighbour pairs within the
               cutoff (default half the pair's mean spacing, 0.5 x s̄); plain nearest-neighbour distance would mix positional
               error with nodes that have no counterpart at all
  displ / s̄    displacement divided by the mean spacing of the two networks
  shift        length of the mean offset vector of those pairs: a systematic
               (global) shift between the sources

Usage (repo root)
  python scripts/characterise_sources.py                  # cutoff 0.5 x s̄, roads included
  python scripts/characterise_sources.py --cutoff-frac 0.25
  python scripts/characterise_sources.py --cutoff 10       # fixed cutoff in metres instead
  python scripts/characterise_sources.py --no-roads

Outputs
  results/characterisation_networks.csv
  results/characterisation_pairs.csv
  results/characterisation_tables.tex   (table rows, ready to paste)
"""
import argparse
import csv
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results"

WASTEWATER = ["Dataset1", "Dataset2", "Dataset3", "Dataset4", "Dataset5"]
ROADS = [("IGN", "OSM")]
COLOCATION_TOL = 0.1   # m, same as the proximity ground truth


def load(name):
    with open(PROCESSED / name / "Graph.pkl", "rb") as f:
        G = pickle.load(f)
    X = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)], dtype=float)
    return G, X


def network_stats(name, G, X):
    d, _ = cKDTree(X).query(X, k=2)          # k=2: the first hit is the node itself
    s = float(np.median(d[:, 1]))
    span = (X.max(axis=0) - X.min(axis=0)) / 1000.0
    return {"network": name, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "s_m": round(s, 2), "extent_x_km": round(float(span[0]), 1),
            "extent_y_km": round(float(span[1]), 1)}


def pair_stats(a, b, XA, XB, sA, sB, cutoff_m, cutoff_frac):
    s_bar = (sA + sB) / 2.0
    cutoff = cutoff_m if cutoff_m is not None else cutoff_frac * s_bar
    tA, tB = cKDTree(XA), cKDTree(XB)

    # co-location: exactly one partner within the tolerance (step 09's rule)
    counts = np.array([len(l) for l in tB.query_ball_point(XA, r=COLOCATION_TOL)])
    coloc = float(np.mean(counts == 1))

    # mutual nearest neighbours within the cutoff
    dAB, jAB = tB.query(XA, distance_upper_bound=cutoff)
    _, iBA = tA.query(XB, distance_upper_bound=cutoff)
    ok = np.isfinite(dAB)
    i = np.nonzero(ok)[0]
    j = jAB[ok]
    mutual = iBA[j] == i
    i, j = i[mutual], j[mutual]
    off = XB[j] - XA[i]
    dist = np.linalg.norm(off, axis=1)
    mean_d = float(dist.mean()) if len(dist) else float("nan")
    shift = float(np.linalg.norm(off.mean(axis=0))) if len(dist) else float("nan")
    return {"pair": f"{a}-{b}", "ratio": round(len(XB) / len(XA), 2),
            "colocated_pct": round(100 * coloc, 1),
            "mutual_pairs": int(len(dist)),
            "mean_displacement_m": round(mean_d, 2),
            "median_displacement_m": round(float(np.median(dist)), 2) if len(dist) else "",
            "displacement_over_s": round(mean_d / s_bar, 3),
            "shift_m": round(shift, 2), "cutoff_m": round(cutoff, 2)}


def short(name):
    return name.replace("Dataset", "D")


def fmt_int(v):
    return f"{v:,}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff-frac", type=float, default=0.5,
                    help="mutual-NN cutoff as a fraction of the pair's mean spacing s̄ (default 0.5)")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="fixed mutual-NN cutoff in metres (overrides --cutoff-frac)")
    ap.add_argument("--no-roads", action="store_true", help="skip IGN / OSM")
    args = ap.parse_args()

    names = WASTEWATER + ([] if args.no_roads else [n for p in ROADS for n in p])
    pairs = list(combinations(WASTEWATER, 2)) + ([] if args.no_roads else ROADS)

    data, nets = {}, []
    for n in names:
        print(f"loading {n} ...")
        G, X = load(n)
        st = network_stats(n, G, X)
        data[n] = (X, st["s_m"])
        nets.append(st)
        print(f"  |V| {st['nodes']:,}  |E| {st['edges']:,}  s {st['s_m']} m  "
              f"extent {st['extent_x_km']} x {st['extent_y_km']} km")
        del G

    rows = []
    for a, b in pairs:
        (XA, sA), (XB, sB) = data[a], data[b]
        r = pair_stats(a, b, XA, XB, sA, sB, args.cutoff, args.cutoff_frac)
        rows.append(r)
        print(f"{r['pair']:18s} ratio {r['ratio']:.2f}  co-located {r['colocated_pct']:5.1f}%  "
              f"mutual {r['mutual_pairs']:,}  displ {r['mean_displacement_m']} m  "
              f"displ/s {r['displacement_over_s']}  shift {r['shift_m']} m  (cutoff {r['cutoff_m']} m)")

    OUT.mkdir(exist_ok=True)
    for path, rs in ((OUT / "characterisation_networks.csv", nets),
                     (OUT / "characterisation_pairs.csv", rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)

    rule = f"{args.cutoff:g} m" if args.cutoff is not None else f"{args.cutoff_frac:g} x mean spacing"
    tex = [f"% generated by scripts/characterise_sources.py (mutual-NN cutoff {rule})",
           "% --- tbl:datasets ---"]
    for st in nets:
        if st["network"] == "IGN":
            tex.append("\\midrule")
        tex.append(f"{st['network']} & {fmt_int(st['nodes'])} & {fmt_int(st['edges'])} & "
                   f"{st['s_m']:.2f} & ${st['extent_x_km']:.1f}\\times{st['extent_y_km']:.1f}$ \\\\")
    tex.append("% --- tbl:pairs ---")
    for r in rows:
        a, b = r["pair"].split("-")
        if a == "IGN":
            tex.append("\\midrule")
        tex.append(f"{short(a)}--{short(b)} & {r['ratio']:.2f} & {r['colocated_pct']:.1f} & "
                   f"{r['mean_displacement_m']:.2f} & {r['displacement_over_s']:.3f} & "
                   f"{r['shift_m']:.2f} \\\\")
    (OUT / "characterisation_tables.tex").write_text("\n".join(tex) + "\n")
    print("\n✓ results/characterisation_networks.csv, characterisation_pairs.csv, "
          "characterisation_tables.tex")


if __name__ == "__main__":
    main()
