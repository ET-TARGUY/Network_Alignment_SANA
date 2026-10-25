#!/usr/bin/env python3
"""
struc2vec baseline, run through the shared pipeline.

Runs in the MAIN environment (.venv). Only embedding training is delegated to the
struc2vec environment (venv_struc2vec), through an edge list: the two
environments pin incompatible networkx versions, so Graph.pkl cannot be loaded
on the struc2vec side.

For every (base config, dataset pair, until_layer):
  1. train struc2vec on each graph (cached per graph, layer and settings)
  2. map G1's embedding space onto G2's with orthogonal Procrustes on anchors
  3. score radius-limited candidates by cosine similarity - same radius as NN/SANA
  4. exact one-to-one assignment with block_assignment
  5. evaluate with scripts/09_evaluate_alignment.py (the shared evaluator)
  6. per-fold metrics -> CSV; until_layer selected by 2-fold spatial cross-fitting

Anchors
-------
`mutual_nn` (default): mutual nearest neighbours within the tolerance. It never
reads the ground truth under either protocol. `colocated` is the old rule (unique
partner within 0.1 m): on the real pairs it IS the ground truth, and under
Protocol A it finds almost no anchors once nodes are displaced.

Usage (from the repo root)
--------------------------
  .venv/bin/python scripts/main_struc2vec.py --config config/struc2vec_tuning.yaml --dry-run
  .venv/bin/python scripts/main_struc2vec.py --config config/struc2vec_tuning.yaml
  .venv/bin/python scripts/main_struc2vec.py --config config/struc2vec_tuning.yaml \
      --variants "data/processed/Z1__identity_L0_s01" --layers 2
  .venv/bin/python scripts/main_struc2vec.py --config config/struc2vec_all_pairs.yaml

Outputs
-------
  <out_dir>/<out_prefix>_raw.csv       one row per (variant, pair, layer, fold)
  <out_dir>/<out_prefix>_selected.csv  cross-fitted result per (variant, pair)
  data/processed/<zone>/<ds1>/alignment_{hungarian,evaluation}_Struc2vec_L<k>_<ds2>.pkl
  data/processed/<zone>/<ds>/struc2vec/emb_L<k>_<settings>.emb (+ .log)
"""

import argparse
import csv
import glob
import importlib.util
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.matching.assignment import block_assignment  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
LIB_DIR = ROOT / "src" / "struc2vec_lib"
LIB_FILES = ["main.py", "struc2vec.py", "algorithms.py",
             "algorithms_distances.py", "graph.py", "utils.py"]

RAW_FIELDS = [
    "variant", "dataset1", "dataset2", "until_layer", "fold", "status",
    "n_eval", "TP", "FP", "FN", "precision", "recall", "f1",
    "radius_m", "gt_mode", "expected_no_match", "false_alarms",
    "anchor_mode", "n_anchors", "no_embedding_g1", "train_seconds", "seconds",
]
SEL_FIELDS = [
    "variant", "dataset1", "dataset2", "layer_A", "layer_B", "layer_best_all",
    "TP", "FP", "FN", "precision", "recall", "f1",
]


def _load_sweep_helpers():
    """make_folds / fold_metrics from run_sweep.py, so folds and per-fold
    metrics are computed exactly as for SANA and NN."""
    spec = importlib.util.spec_from_file_location("run_sweep", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEP = _load_sweep_helpers()


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def resolve_bases(scfg, args):
    """Return [(label, base_config_path)]."""
    patterns = as_list(args.variants) or as_list(scfg.get("variants"))
    if patterns:
        dirs = set()
        for p in patterns:
            p = p if os.path.isabs(p) else str(ROOT / p)
            dirs.update(glob.glob(p))
        out = []
        for d in sorted(dirs):
            cfg = Path(d) / "config.yaml"
            if cfg.exists():
                out.append((Path(d).name, cfg))
        return out
    base = args.base or scfg.get("base_config")
    if not base:
        sys.exit("No `variants` and no `base_config` given.")
    base = Path(base) if os.path.isabs(base) else ROOT / base
    return [(Path(base).stem, base)]


def zone_dir(base_cfg):
    zone = str(base_cfg.get("zone", "full"))
    return PROCESSED if zone.lower() == "full" else PROCESSED / zone


def settings_tag(p):
    return (f"d{p['dimensions']}_n{p['num_walks']}_l{p['walk_length']}"
            f"_w{p['window_size']}_e{p['epochs']}")


# ----------------------------------------------------------------------
# struc2vec training (in venv_struc2vec)
# ----------------------------------------------------------------------

def write_edgelist(G, path):
    """Integer edge list; isolated nodes as single-token lines (the struc2vec
    loader accepts them), so every node receives an embedding."""
    H = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    with open(path, "w") as f:
        for u, v in H.edges():
            f.write(f"{int(u)} {int(v)}\n")
        for n in H.nodes():
            if H.degree(n) == 0:
                f.write(f"{int(n)}\n")


def train_struc2vec(G, emb_path, layer, p):
    """Train once and cache. Each run gets its own copy of the library, because
    the library writes its pickles next to itself and random_walks.txt into the
    working directory: a shared copy leaks state between runs (it reloads every
    `distances_nets_weights-layer-*` pickle it finds, including stale layers from
    a previous graph) and makes parallel runs collide."""
    if emb_path.exists():
        return 0.0
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = emb_path.parent / f"_run_{emb_path.stem}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "lib").mkdir(parents=True)
    (run_dir / "pickles").mkdir()
    for fn in LIB_FILES:
        shutil.copy2(LIB_DIR / fn, run_dir / "lib" / fn)

    edgelist = run_dir / "input.edgelist"
    write_edgelist(G, edgelist)
    tmp_out = run_dir / "output.emb"

    py = p["python"]
    py = py if os.path.isabs(py) else str(ROOT / py)
    cmd = [py, str(run_dir / "lib" / "main.py"),
           "--input", str(edgelist), "--output", str(tmp_out),
           "--dimensions", str(p["dimensions"]),
           "--num-walks", str(p["num_walks"]),
           "--walk-length", str(p["walk_length"]),
           "--window-size", str(p["window_size"]),
           "--iter", str(p["epochs"]),
           "--workers", str(p["workers"]),
           "--until-layer", str(layer)]
    # The library parses OPT flags with type=bool, so ANY string (even "False")
    # turns them on. Pass a flag only to switch it on.
    for k in ("OPT1", "OPT2", "OPT3"):
        if p.get(k):
            cmd += [f"--{k}", "1"]

    log_path = emb_path.with_suffix(".log")
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        res = subprocess.run(cmd, cwd=str(run_dir), stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if res.returncode != 0 or not tmp_out.exists():
        raise RuntimeError(f"struc2vec failed (exit {res.returncode}); see {log_path}")
    shutil.move(str(tmp_out), str(emb_path))
    if not p.get("keep_run_dir", False):
        shutil.rmtree(run_dir, ignore_errors=True)
    print(f"      trained {emb_path.name} in {dt:.0f}s")
    return dt


def read_embeddings(path):
    emb = {}
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) > 1:
                emb[int(parts[0])] = np.asarray(parts[1:], dtype=float)
    return emb


# ----------------------------------------------------------------------
# alignment
# ----------------------------------------------------------------------

def find_anchors(X1, X2, mode, tol):
    """Index pairs (i, j) used to fit Procrustes. Coordinates only."""
    t2 = cKDTree(X2)
    if mode == "colocated":
        lists = t2.query_ball_point(X1, r=tol)
        pairs = [(i, l[0]) for i, l in enumerate(lists) if len(l) == 1]
        used = {}
        for _, j in pairs:
            used[j] = used.get(j, 0) + 1
        return [(i, j) for i, j in pairs if used[j] == 1]
    if mode == "mutual_nn":
        t1 = cKDTree(X1)
        d12, j12 = t2.query(X1, distance_upper_bound=tol)
        d21, i21 = t1.query(X2, distance_upper_bound=tol)
        n2 = len(X2)
        return [(i, int(j)) for i, (d, j) in enumerate(zip(d12, j12))
                if np.isfinite(d) and j < n2 and i21[j] == i]
    raise ValueError(f"unknown anchor mode: {mode}")


def unit(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def align_pair(G1, G2, emb1, emb2, radius, acfg):
    nodes1 = [n for n in G1.nodes() if int(n) in emb1]
    nodes2 = [n for n in G2.nodes() if int(n) in emb2]
    no_emb = G1.number_of_nodes() - len(nodes1)
    X1 = np.array([[G1.nodes[n]["x"], G1.nodes[n]["y"]] for n in nodes1])
    X2 = np.array([[G2.nodes[n]["x"], G2.nodes[n]["y"]] for n in nodes2])
    E1 = np.array([emb1[int(n)] for n in nodes1])
    E2 = np.array([emb2[int(n)] for n in nodes2])

    mode = acfg.get("mode", "procrustes")
    anchor_mode = acfg.get("anchors", "mutual_nn")
    n_anchors = 0
    if mode == "procrustes":
        tol = acfg.get("anchor_tolerance_m")
        tol = radius if tol is None else float(tol)
        pairs = find_anchors(X1, X2, anchor_mode, tol)
        n_anchors = len(pairs)
        if n_anchors < int(acfg.get("min_anchors", 10)):
            return None, dict(n_anchors=n_anchors, no_emb=no_emb, anchor_mode=anchor_mode)
        ai = [a for a, _ in pairs]
        bj = [b for _, b in pairs]
        R, _ = orthogonal_procrustes(E1[ai], E2[bj])
        E1 = E1 @ R
    elif mode != "direct":
        raise ValueError(f"unknown alignment mode: {mode}")

    A, B = unit(E1), unit(E2)
    lists = cKDTree(X2).query_ball_point(X1, r=radius)
    rows = np.fromiter((i for i, l in enumerate(lists) for _ in l), dtype=np.int64)
    cols = np.fromiter((j for l in lists for j in l), dtype=np.int64)
    if len(rows) == 0:
        return [], dict(n_anchors=n_anchors, no_emb=no_emb, anchor_mode=anchor_mode)
    sims = np.einsum("ij,ij->i", A[rows], B[cols])
    costs = 1.0 - sims

    sim_of = dict(zip(zip(rows.tolist(), cols.tolist()), sims.tolist()))
    matches = []
    for i, j in block_assignment(rows, cols, costs):
        i, j = int(i), int(j)
        matches.append({
            "g1_id": nodes1[i], "g2_id": nodes2[j],
            "x_g1": X1[i][0], "y_g1": X1[i][1],
            "x_g2": X2[j][0], "y_g2": X2[j][1],
            "distance_m": float(np.linalg.norm(X1[i] - X2[j])),
            "similarity": float(sim_of[(i, j)]),
        })
    return matches, dict(n_anchors=n_anchors, no_emb=no_emb, anchor_mode=anchor_mode)


# ----------------------------------------------------------------------
# evaluation (shared step 09)
# ----------------------------------------------------------------------

def evaluate(base_cfg, base_dir, ds1, ds2, method, quiet):
    cfg = dict(base_cfg)
    cfg["dataset_pairs"] = [[ds1, ds2]]
    cfg["compound_graph"] = dict(cfg["compound_graph"])
    cfg["compound_graph"]["method"] = method
    cfg["export_gpkg"] = False
    if str(cfg.get("zone", "full")).lower() != "full":
        cfg["paths"] = dict(cfg["paths"])
        cfg["paths"]["outputs"] = str(base_dir / "outputs")
    cfg_path = base_dir / ds1 / "struc2vec" / f"_eval_{method}_{ds2}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "09_evaluate_alignment.py")],
        cwd=str(ROOT), env=dict(os.environ, SANA_CONFIG=str(cfg_path)),
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    if res.returncode != 0:
        tail = (res.stdout or b"").decode(errors="replace").strip().splitlines()[-15:]
        raise RuntimeError("step 09 failed:\n      " + "\n      ".join(tail))
    return base_dir / ds1 / f"alignment_evaluation_{method}_{ds2}.pkl"


# ----------------------------------------------------------------------
# results
# ----------------------------------------------------------------------

def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def cross_fit(raw_rows):
    """Pick until_layer on one fold, score on the other. Recall is TP/(TP+FN),
    matching run_sweep.cross_fit(). Ties go to the smaller
    layer, which is cheaper."""
    groups = {}
    for r in raw_rows:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["variant"], r["dataset1"], r["dataset2"]), []).append(r)

    out = []
    for (variant, d1, d2), rs in sorted(groups.items()):
        idx = {(int(r["until_layer"]), r["fold"]): r for r in rs}
        layers = sorted({int(r["until_layer"]) for r in rs})

        def best_on(fold):
            best, best_f1 = None, -1.0
            for L in layers:
                r = idx.get((L, fold))
                if r and float(r["f1"]) > best_f1:
                    best, best_f1 = L, float(r["f1"])
            return best

        tot = {"TP": 0, "FP": 0, "FN": 0}
        chosen = {}
        for fold, other in (("A", "B"), ("B", "A")):
            L = best_on(other)
            chosen[fold] = L
            r = idx.get((L, fold)) if L is not None else None
            if r:
                for k in tot:
                    tot[k] += int(r[k])
        tp, fp, fn = tot["TP"], tot["FP"], tot["FN"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({
            "variant": variant, "dataset1": d1, "dataset2": d2,
            "layer_A": chosen.get("A"), "layer_B": chosen.get("B"),
            "layer_best_all": best_on("all"),
            **tot, "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        })
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="struc2vec config (config/struc2vec_*.yaml)")
    ap.add_argument("--variants", nargs="+", help="override `variants` (glob patterns)")
    ap.add_argument("--base", help="override `base_config` when no variants are used")
    ap.add_argument("--layers", nargs="+", type=int, help="override grid.until_layer")
    ap.add_argument("--force", action="store_true", help="recompute rows already in the CSV")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="show step-09 output")
    args = ap.parse_args()

    os.chdir(ROOT)
    scfg = load_yaml(ROOT / args.config if not os.path.isabs(args.config) else args.config)
    p = scfg["struc2vec"]
    if not p.get("OPT3"):
        sys.exit("struc2vec.OPT3 must be true: the library ignores until_layer otherwise.")
    acfg = scfg.get("alignment", {})
    layers = args.layers or as_list(scfg["grid"]["until_layer"])
    out_dir = ROOT / scfg.get("out_dir", "results")
    prefix = scfg.get("out_prefix", "struc2vec")
    raw_path = out_dir / f"{prefix}_raw.csv"
    sel_path = out_dir / f"{prefix}_selected.csv"
    tag = settings_tag(p)

    bases = resolve_bases(scfg, args)
    if not bases:
        sys.exit("No base configs / variant directories matched.")

    done = set()
    if not args.force:
        for r in read_rows(raw_path):
            if not (r["status"] == "ok" or r["status"] == "too_few_anchors"):
                continue            # failed runs are retried
            done.add((r["variant"], r["dataset1"], r["dataset2"], int(r["until_layer"])))

    plan = []
    for label, cfg_path in bases:
        bcfg = load_yaml(cfg_path)
        for ds1, ds2 in bcfg["dataset_pairs"]:
            for L in layers:
                if (label, ds1, ds2, int(L)) not in done:
                    plan.append((label, cfg_path, ds1, ds2, int(L)))

    print(f"struc2vec: {len(bases)} base config(s), layers {layers}, {len(plan)} run(s) to do")
    print(f"  anchors: {acfg.get('anchors', 'mutual_nn')}  mode: {acfg.get('mode', 'procrustes')}")
    print(f"  raw CSV: {raw_path.relative_to(ROOT)}")
    if args.dry_run:
        for label, _, ds1, ds2, L in plan:
            print(f"  {label:32s} {ds1}->{ds2}  L={L}")
        return

    graphs, folds_cache = {}, {}

    def graph(base_dir, ds):
        key = (base_dir, ds)
        if key not in graphs:
            if len(graphs) > 4:          # keep memory bounded on full-scale runs
                graphs.clear()
            with open(base_dir / ds / "Graph.pkl", "rb") as f:
                graphs[key] = pickle.load(f)
        return graphs[key]

    for k, (label, cfg_path, ds1, ds2, L) in enumerate(plan, 1):
        bcfg = load_yaml(cfg_path)
        base_dir = zone_dir(bcfg)
        radius = float(bcfg["spatial"]["radius"])
        method = f"Struc2vec_L{L}"
        print(f"\n[{k}/{len(plan)}] {label}  {ds1}->{ds2}  until_layer={L}  radius={radius:g} m")
        t0 = time.time()
        base_row = {"variant": label, "dataset1": ds1, "dataset2": ds2, "until_layer": L,
                    "radius_m": radius, "anchor_mode": acfg.get("anchors", "mutual_nn")}
        try:
            G1, G2 = graph(base_dir, ds1), graph(base_dir, ds2)
            train_s = 0.0
            embs = []
            for ds, G in ((ds1, G1), (ds2, G2)):
                emb_path = base_dir / ds / "struc2vec" / f"emb_L{L}_{tag}.emb"
                train_s += train_struc2vec(G, emb_path, L, p)
                embs.append(read_embeddings(emb_path))

            matches, info = align_pair(G1, G2, embs[0], embs[1], radius, acfg)
            base_row.update(n_anchors=info["n_anchors"], no_embedding_g1=info["no_emb"],
                            train_seconds=round(train_s, 1))
            if matches is None:
                print(f"      skipped: only {info['n_anchors']} anchors")
                append_rows(raw_path, [dict(base_row, fold="all", status="too_few_anchors",
                                            seconds=round(time.time() - t0, 1))])
                continue
            print(f"      anchors {info['n_anchors']:,}  matches {len(matches):,}"
                  f"  no-embedding G1 nodes {info['no_emb']}")

            with open(base_dir / ds1 / f"alignment_hungarian_{method}_{ds2}.pkl", "wb") as f:
                pickle.dump({"matches": matches, "method": method, "dataset1": ds1, "dataset2": ds2,
                             "parameters": {"until_layer": L, "radius": radius, **p, **acfg}}, f)

            eval_pkl = evaluate(bcfg, base_dir, ds1, ds2, method, quiet=not args.verbose)

            fkey = (base_dir, ds1)
            if fkey not in folds_cache:
                fp = base_dir / "folds.pkl"
                if fp.exists() and str(bcfg.get("zone", "full")).lower() != "full":
                    with open(fp, "rb") as f:
                        folds_cache[fkey] = pickle.load(f)
                else:
                    folds_cache[fkey] = SWEEP.make_folds(base_dir / ds1 / "Graph.pkl")
            fm = SWEEP.fold_metrics(eval_pkl, folds_cache[fkey])

            secs = round(time.time() - t0, 1)
            rows = []
            for fold in ("A", "B", "all"):
                r = fm[fold]
                rows.append(dict(base_row, fold=fold, status="ok", seconds=secs,
                                 gt_mode=fm["_meta"]["gt_mode"],
                                 expected_no_match=fm["_meta"]["expected_no_match"],
                                 false_alarms=fm["_meta"]["false_alarms"],
                                 **{kk: (round(v, 6) if isinstance(v, float) else v)
                                    for kk, v in r.items()}))
            append_rows(raw_path, rows)
            print(f"      F1 all {fm['all']['f1']:.4f}  (A {fm['A']['f1']:.4f}, B {fm['B']['f1']:.4f})  {secs:.0f}s")
        except Exception as e:  # keep the sweep going; the failure is recorded
            print(f"      FAILED: {e}")
            append_rows(raw_path, [dict(base_row, fold="all", status=f"error: {str(e)[:120]}",
                                        seconds=round(time.time() - t0, 1))])

    sel = cross_fit(read_rows(raw_path))
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sel_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEL_FIELDS)
        w.writeheader()
        w.writerows(sel)
    print(f"\n✓ {len(sel)} cross-fitted result(s) -> {sel_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
