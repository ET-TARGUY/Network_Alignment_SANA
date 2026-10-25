#!/usr/bin/env python3
"""
Step 2 - the real source pairs, with the values tuned under Protocol A.

Every method runs on the same pairs, the same candidate radius and the same
evaluator (proximity ground truth: a unique partner within ANCHOR_TOLERANCE).
Tuned on the composite L0.125 variant, seed 1:

  NN            no parameter
  SANA a=0      alpha = 0 (the embedding is ignored)
  SANA          sigma = 6, alpha = 0.9
  CENA          K = 6, structural candidates (top_m), no radius
  DeepWalk      no parameter          | run separately, see --posthoc below
  struc2vec     until_layer = 4       |

Outputs, per pair, in data/processed/<ds1>/:
  alignment_{hungarian,evaluation}_<Name>_<ds2>.pkl   Name = NearestNeighbor |
  SANA_a09 | SANA_a0 | CENA
and a summary table written to results/real_pairs.csv.

Usage (repo root, main .venv)
  python scripts/step2_real_pairs.py --dry-run
  python scripts/step2_real_pairs.py                     # all methods, all pairs
  python scripts/step2_real_pairs.py --only NN SANA      # NN SANA SANA0 CENA
  python scripts/step2_real_pairs.py --pairs Dataset1:Dataset2
  python scripts/step2_real_pairs.py --summary           # rebuild the table only
  python scripts/step2_real_pairs.py --posthoc           # print the DeepWalk /
                                                         # struc2vec commands
"""
import argparse
import csv
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "config" / "all_pairs.yaml"
OUT_CSV = "results/real_pairs.csv"
PROCESSED = ROOT / "data" / "processed"
WORK = ROOT / "data" / "processed" / "_step2"

TUNED = {"sigma": 6, "alpha": 0.9, "K": 6, "top_m": 50, "struc2vec_layer": 4}
METHODS = ["NN", "SANA", "SANA0", "CENA"]
NAMES = [("NN", "NearestNeighbor"), ("SANA (s=6, a=0.9)", "SANA_a09"),
         ("SANA (a=0)", "SANA_a0"), ("CENA (K=6)", "CENA")]


def run(script, cfg, quiet=True):
    env = dict(os.environ, SANA_CONFIG=str(cfg))
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=str(ROOT), env=env,
                         stdout=subprocess.PIPE if quiet else None,
                         stderr=subprocess.STDOUT if quiet else None)
    if res.returncode != 0:
        tail = (res.stdout or b"").decode(errors="replace").strip().splitlines()[-12:]
        print(f"    FAILED {script}\n      " + "\n      ".join(tail))
        return False
    return True


def write_cfg(base, name, pair, **changes):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["dataset_pairs"] = [list(pair)]
    for dotted, value in changes.items():
        d, keys = cfg, dotted.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    for k in ("export_gpkg", "export_figures", "save_model", "save_walk_stats"):
        cfg[k] = False
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"{name}_{pair[0]}_{pair[1]}.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def rename(ds1, ds2, src, dst):
    for kind in ("hungarian", "evaluation"):
        a = PROCESSED / ds1 / f"alignment_{kind}_{src}_{ds2}.pkl"
        if a.exists():
            shutil.copy2(a, PROCESSED / ds1 / f"alignment_{kind}_{dst}_{ds2}.pkl")


def summary(pairs):
    rows = []
    for ds1, ds2 in pairs:
        for label, name in NAMES:
            p = PROCESSED / ds1 / f"alignment_evaluation_{name}_{ds2}.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:
                ev = pickle.load(f)
            tp, fp, fn = ev["true_positives"], ev["false_positives"], ev["false_negatives"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            rows.append({"pair": f"{ds1}-{ds2}", "method": label,
                         "anchors": ev["ground_truth_anchors"], "TP": tp, "FP": fp, "FN": fn,
                         "precision": round(prec, 6), "recall": round(rec, 6),
                         "f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 6)})
    if rows:
        out = ROOT / OUT_CSV
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{'pair':20s}{'method':20s}{'anchors':>9s}{'F1':>8s}{'FP':>8s}{'FN':>7s}")
        for r in rows:
            print(f"{r['pair']:20s}{r['method']:20s}{r['anchors']:>9,d}"
                  f"{r['f1']:>8.4f}{r['FP']:>8,d}{r['FN']:>7,d}")
        print(f"\n-> {out.relative_to(ROOT)}")
    else:
        print("no evaluation files yet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=METHODS)
    ap.add_argument("--pairs", nargs="+", help="Dataset1:Dataset2 ...")
    ap.add_argument("--base", default="config/all_pairs.yaml",
                    help="base config naming the dataset pairs and the GT tolerance")
    ap.add_argument("--out", default="results/real_pairs.csv",
                    help="summary CSV to write")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--posthoc", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    global BASE, OUT_CSV
    BASE = ROOT / args.base
    OUT_CSV = args.out
    base = yaml.safe_load(open(BASE))
    pairs = [tuple(p.split(":")) for p in args.pairs] if args.pairs else [tuple(p) for p in base["dataset_pairs"]]

    if args.posthoc:
        print("DeepWalk and struc2vec run through main_posthoc.py:\n")
        print(f"  python scripts/main_posthoc.py --config config/posthoc_real.yaml "
              f"--method deepwalk --base config/all_pairs.yaml --jobs 2")
        print(f"  python scripts/main_posthoc.py --config config/posthoc_real.yaml "
              f"--method struc2vec --base config/all_pairs.yaml --layers {TUNED['struc2vec_layer']} --jobs 2")
        return
    if args.summary:
        summary(pairs)
        return

    todo = args.only or METHODS
    print(f"Step 2: {len(pairs)} pair(s), methods {todo}")
    print(f"  tuned on composite L0.125: sigma={TUNED['sigma']} alpha={TUNED['alpha']} "
          f"K={TUNED['K']} top_m={TUNED['top_m']}")
    if args.dry_run:
        for ds1, ds2 in pairs:
            print(f"  {ds1} -> {ds2}")
        return

    quiet = not args.verbose
    for i, (ds1, ds2) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {ds1} -> {ds2}")
        # candidates (radius) are shared by NN, SANA and SANA a=0
        cfg = write_cfg(base, "spatial", (ds1, ds2))
        if any(m in todo for m in ("NN", "SANA", "SANA0")):
            if not run("02_spatial_candidates.py", cfg, quiet):
                continue

        if "NN" in todo:
            print("  NN ...", flush=True)
            run("main_nearest_neighbor.py", cfg, quiet)

        if "SANA" in todo or "SANA0" in todo:
            c = write_cfg(base, "sana", (ds1, ds2),
                          **{"compound_graph.method": "SANA",
                             "compound_graph.gaussian.sigma": TUNED["sigma"],
                             "alignment.alpha": TUNED["alpha"],
                             "alignment.beta": round(1 - TUNED["alpha"], 6)})
            if "SANA" in todo:
                print(f"  SANA (sigma={TUNED['sigma']}, alpha={TUNED['alpha']}) ...", flush=True)
                ok = all(run(s, c, quiet) for s in ("04_build_compound_graph.py",
                                                    "05_generate_random_walks.py",
                                                    "06_generate_embeddings.py",
                                                    "08_hungarian_alignment.py",
                                                    "09_evaluate_alignment.py"))
                if ok:
                    rename(ds1, ds2, "SANA", "SANA_a09")
            if "SANA0" in todo:
                print("  SANA (alpha=0) ...", flush=True)
                c0 = write_cfg(base, "sana0", (ds1, ds2),
                               **{"compound_graph.method": "SANA",
                                  "compound_graph.gaussian.sigma": TUNED["sigma"],
                                  "alignment.alpha": 0, "alignment.beta": 1})
                if all(run(s, c0, quiet) for s in ("08_hungarian_alignment.py",
                                                   "09_evaluate_alignment.py")):
                    rename(ds1, ds2, "SANA", "SANA_a0")

        if "CENA" in todo:
            print(f"  CENA (K={TUNED['K']}, structural candidates) ...", flush=True)
            c = write_cfg(base, "cena", (ds1, ds2),
                          **{"compound_graph.method": "CENA", "structural.K": TUNED["K"],
                             "structural.candidates": "all", "structural.top_m": TUNED["top_m"]})
            all(run(s, c, quiet) for s in ("03_structural_similarity.py",
                                           "04_build_compound_graph.py",
                                           "05_generate_random_walks.py",
                                           "06_generate_embeddings.py",
                                           "08_hungarian_alignment.py",
                                           "09_evaluate_alignment.py"))

    summary(pairs)


if __name__ == "__main__":
    main()
