#!/usr/bin/env python3
"""
Step 0 - harness check on the identity control.

Z1 is aligned against a relabelled, unperturbed copy of itself with every method
at its fixed settings. Pass: F1 = 1.000, or every error is a swap between
structurally equivalent nodes (scripts/check_swaps.py).

  Method        how it runs                                   files named
  NN            scripts/main_nearest_neighbor.py              NearestNeighbor
  SANA a=1      scripts/main_SANA_CENA.py (alpha 1)           SANA_a1
  SANA a=0      steps 08-09 again with alpha 0                SANA_a0
  CENA          scripts/main_SANA_CENA.py (method CENA)       CENA
  DeepWalk      scripts/main_posthoc.py --method deepwalk     DeepWalk
  struc2vec     scripts/main_posthoc.py --method struc2vec    Struc2vec_L<k>

Usage (repo root, main .venv)
  python scripts/step0_identity.py                 # all methods
  python scripts/step0_identity.py --only NN SANA  # a subset
  python scripts/step0_identity.py --summary       # just print the table again
"""
import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "scripts"))
from check_swaps import check  # noqa: E402

VARIANT = ROOT / "data" / "processed" / "Z1__identity_L0_s01"
POSTHOC_CFG = "config/posthoc.yaml"
METHODS = ["NN", "SANA", "SANA0", "CENA", "DeepWalk", "struc2vec"]


def run(cmd, cfg_path=None):
    env = dict(os.environ)
    if cfg_path:
        env["SANA_CONFIG"] = str(cfg_path)
    print("\n$ " + " ".join(str(c) for c in cmd) + (f"   [SANA_CONFIG={cfg_path.name}]" if cfg_path else ""))
    res = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env)
    if res.returncode != 0:
        sys.exit(f"FAILED (exit {res.returncode})")


def write_cfg(base, name, **changes):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    for dotted, value in changes.items():
        d = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    cfg["export_gpkg"] = False
    cfg["paths"]["outputs"] = str(VARIANT / "step0" / "outputs")
    path = VARIANT / "step0" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def rename(ds1, ds2, src, dst):
    for kind in ("hungarian", "evaluation"):
        a = VARIANT / ds1 / f"alignment_{kind}_{src}_{ds2}.pkl"
        shutil.copy2(a, VARIANT / ds1 / f"alignment_{kind}_{dst}_{ds2}.pkl")


def summary(names, ds1, ds2):
    print("\n" + "=" * 72)
    print("STEP 0 - identity control")
    print("=" * 72)
    print(f"{'method':18s} {'scored':>7s} {'TP':>7s} {'FP':>5s} {'FN':>5s} {'F1':>8s}  result")
    for label, name in names:
        p = VARIANT / ds1 / f"alignment_evaluation_{name}_{ds2}.pkl"
        if not p.exists():
            print(f"{label:18s}  (not run)")
            continue
        ev = pickle.load(open(p, "rb"))
        tp, fp, fn = ev["true_positives"], ev["false_positives"], ev["false_negatives"]
        n = tp + fp + fn
        f1 = ev["f1_score"]
        if fp == 0 and fn == 0:
            verdict = "PASS"
        else:
            c = check(VARIANT, name, verbose=False)
            verdict = ("PASS (all errors explained)" if fn == 0 and not c["other"]
                       else f"CHECK: {len(c['other'])} unexplained error(s), FN {fn}")
        print(f"{label:18s} {n:7,d} {tp:7,d} {fp:5,d} {fn:5,d} {f1:8.4f}  {verdict}")
    for label, name in names:
        p = VARIANT / ds1 / f"alignment_evaluation_{name}_{ds2}.pkl"
        if p.exists() and pickle.load(open(p, "rb"))["false_positives"]:
            check(VARIANT, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=METHODS)
    ap.add_argument("--layer", type=int,
                    help="struc2vec until_layer (default: the value in the posthoc config)")
    ap.add_argument("--summary", action="store_true", help="print the table without running")
    ap.add_argument("--force", action="store_true", help="recompute DeepWalk / struc2vec")
    args = ap.parse_args()
    os.chdir(ROOT)

    base = yaml.safe_load(open(VARIANT / "config.yaml"))
    ds1, ds2 = base["dataset_pairs"][0]
    layer = args.layer or yaml.safe_load(open(ROOT / POSTHOC_CFG))["struc2vec"]["until_layer"]
    names = [("NN", "NearestNeighbor"), ("SANA (a=1)", "SANA_a1"), ("SANA (a=0)", "SANA_a0"),
             ("CENA", "CENA"), ("DeepWalk", "DeepWalk"), (f"struc2vec (L={layer})", f"Struc2vec_L{layer}")]

    if not args.summary:
        todo = args.only or METHODS
        py = sys.executable
        force = ["--force"] if args.force else []
        if "NN" in todo:
            run([py, "scripts/main_nearest_neighbor.py"], write_cfg(base, "nn"))
        if "SANA" in todo or "SANA0" in todo:
            if "SANA" in todo or not (VARIANT / ds1 / f"embeddings_SANA_{ds2}.pkl").exists():
                run([py, "scripts/main_SANA_CENA.py"],
                    write_cfg(base, "sana_a1", **{"compound_graph.method": "SANA",
                                                   "alignment.alpha": 1, "alignment.beta": 0}))
                rename(ds1, ds2, "SANA", "SANA_a1")
            if "SANA0" in todo:
                c = write_cfg(base, "sana_a0", **{"compound_graph.method": "SANA",
                                                   "alignment.alpha": 0, "alignment.beta": 1})
                run([py, "scripts/08_hungarian_alignment.py"], c)
                run([py, "scripts/09_evaluate_alignment.py"], c)
                rename(ds1, ds2, "SANA", "SANA_a0")
        if "CENA" in todo:
            run([py, "scripts/main_SANA_CENA.py"],
                write_cfg(base, "cena", **{"compound_graph.method": "CENA"}))
        if "DeepWalk" in todo:
            run([py, "scripts/main_posthoc.py", "--config", POSTHOC_CFG, "--method", "deepwalk",
                 "--variants", str(VARIANT)] + force)
        if "struc2vec" in todo:
            run([py, "scripts/main_posthoc.py", "--config", POSTHOC_CFG, "--method", "struc2vec",
                 "--variants", str(VARIANT), "--layers", str(layer)] + force)

    summary(names, ds1, ds2)


if __name__ == "__main__":
    main()
