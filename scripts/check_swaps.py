#!/usr/bin/env python3
"""
Classify the errors of a Protocol A alignment.

A wrong match a -> copy(b) is EXPLAINED when
  1. it belongs to a permutation cycle: following a -> b -> ... returns to a
     within 5 steps (a swap is a cycle of length 2), and
  2. a and b cannot be told apart by the method's signal:
       co-located   same position (< 1 cm) - distance cannot separate them
       adjacent     a and b are neighbours - they share every random walk
       equivalent   same degree and same multiset of neighbour degrees
       same degree  same degree only (struc2vec compares degree sequences
                    with DTW, so near-equivalence is enough)
Anything else is listed as CHECK and needs a look.

Usage (repo root)
  python scripts/check_swaps.py data/processed/Z1__identity_L0_s01 SANA_a1 Struc2vec_L6
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import yaml

MAX_CYCLE = 5
CLASSES = ["co-located", "adjacent", "equivalent", "same degree", "other"]


def signature(G, n):
    return (G.degree(n), tuple(sorted(G.degree(m) for m in G.neighbors(n))))


def check(variant_dir, name, verbose=True, max_list=10):
    vdir = Path(variant_dir)
    cfg = yaml.safe_load(open(vdir / "config.yaml"))
    ds1, ds2 = cfg["dataset_pairs"][0]
    G1 = pickle.load(open(vdir / ds1 / "Graph.pkl", "rb"))
    inverse = pickle.load(open(Path(cfg["provenance"]), "rb"))["inverse"]   # origin -> copy
    forward = {v: k for k, v in inverse.items()}                            # copy -> origin
    ev = pickle.load(open(vdir / ds1 / f"alignment_evaluation_{name}_{ds2}.pkl", "rb"))
    al = pickle.load(open(vdir / ds1 / f"alignment_hungarian_{name}_{ds2}.pkl", "rb"))
    pred = {m["g1_id"]: m["g2_id"] for m in al["matches"]}

    def origin_of_match(n):
        return forward.get(pred.get(n))

    def cycle_len(a):
        n, steps = a, 0
        while steps < MAX_CYCLE:
            n = origin_of_match(n)
            steps += 1
            if n is None:
                return None
            if n == a:
                return steps
        return None

    wrong = [m["g1_id"] for m in ev.get("incorrect_matches", [])]
    classes, cycles, unexplained = Counter(), Counter(), []
    for a in wrong:
        b = origin_of_match(a)
        if b is None:
            classes["other"] += 1
            unexplained.append((a, pred.get(a), "matched to a node with no counterpart"))
            continue
        dx = G1.nodes[a]["x"] - G1.nodes[b]["x"]
        dy = G1.nodes[a]["y"] - G1.nodes[b]["y"]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.01:
            cls = "co-located"
        elif G1.has_edge(a, b):
            cls = "adjacent"
        elif signature(G1, a) == signature(G1, b):
            cls = "equivalent"
        elif G1.degree(a) == G1.degree(b):
            cls = "same degree"
        else:
            cls = "other"
        classes[cls] += 1
        L = cycle_len(a)
        cycles[L if L else "open"] += 1
        if L is None or cls == "other":
            why = []
            if L is None:
                why.append("not part of a cycle")
            if cls == "other":
                why.append(f"degrees {G1.degree(a)} vs {G1.degree(b)}, not adjacent")
            unexplained.append((a, b, f"{'; '.join(why)}; {dist:.1f} m apart"))

    out = dict(name=name, errors=len(wrong), classes=dict(classes),
               cycles=dict(cycles), other=unexplained)
    if verbose:
        cls_txt = ", ".join(f"{c} {classes[c]}" for c in CLASSES if classes[c])
        cyc_txt = ", ".join(f"length {k}: {v}" for k, v in sorted(cycles.items(), key=str))
        print(f"\n{name}: {len(wrong)} errors")
        if wrong:
            print(f"  kind   : {cls_txt}")
            print(f"  cycles : {cyc_txt}")
            print("  PASS: every error is explained" if not unexplained
                  else f"  {len(unexplained)} unexplained:")
        for a, b, why in unexplained[:max_list]:
            print(f"  CHECK  {a} -> origin {b}: {why}")
        if len(unexplained) > max_list:
            print(f"  ... {len(unexplained) - max_list} more")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for n in sys.argv[2:]:
        check(sys.argv[1], n)
