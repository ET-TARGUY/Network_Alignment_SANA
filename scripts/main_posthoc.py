#!/usr/bin/env python3
"""
Post-hoc embedding baselines (DeepWalk, struc2vec) through the shared pipeline.

Each network is embedded separately, G1's space is mapped onto G2's by
orthogonal Procrustes on a SAMPLE of anchor pairs, radius-limited candidates are
scored by cosine similarity and assigned with block_assignment, and the result
is scored by the shared evaluator (scripts/09_evaluate_alignment.py). Anchor
nodes are excluded from scoring.

Runs in the MAIN environment (.venv). struc2vec training alone is delegated to
venv_struc2vec through an edge list (the environments pin incompatible networkx
versions, so Graph.pkl cannot be read there).

Anchors (alignment.anchors)
  provenance  sample of the provenance map (Protocol A variants)
  colocated   sample of the unique pairs within anchor_tolerance_m (real pairs)
  mutual_nn   all mutual nearest neighbours within anchor_tolerance_m (unsupervised)

Usage (repo root)
  python scripts/main_posthoc.py --config config/posthoc.yaml --method deepwalk \
      --variants data/processed/Z1__identity_L0_s01
  python scripts/main_posthoc.py --config config/posthoc.yaml --method struc2vec \
      --variants data/processed/Z1__identity_L0_s01 --layers 6
  add --dry-run to list the runs, --force to recompute.

Outputs
  <out_dir>/posthoc_<method>_raw.csv       one row per (variant, pair, param, fold)
  <out_dir>/posthoc_<method>_selected.csv  cross-fitted result per (variant, pair)
  data/processed/<zone>/<ds1>/alignment_{hungarian,evaluation}_<Name>_<ds2>.pkl
      Name = DeepWalk | Struc2vec_L<k>
"""

import argparse
import csv
import glob
import importlib.util
import os
import pickle
import random
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
    "variant", "dataset1", "dataset2", "method", "param", "fold", "status",
    "n_eval", "TP", "FP", "FN", "precision", "recall", "f1",
    "radius_m", "gt_mode", "expected_no_match", "false_alarms",
    "anchor_mode", "n_anchors", "no_embedding_g1", "train_seconds", "seconds",
]
SEL_FIELDS = ["variant", "dataset1", "dataset2", "method", "param_A", "param_B",
              "param_best_all", "TP", "FP", "FN", "precision", "recall", "f1"]


def _load_sweep_helpers():
    spec = importlib.util.spec_from_file_location("run_sweep", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEP = _load_sweep_helpers()   # make_folds / fold_metrics, identical to SANA and NN


# ----------------------------------------------------------------------
# config helpers
# ----------------------------------------------------------------------

def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def as_list(x):
    """A config value may be a single scalar (until_layer: 6) or a list."""
    if x is None:
        return []
    if isinstance(x, (str, int, float)):
        return [x]
    return list(x)


def rooted(p):
    return Path(p) if os.path.isabs(str(p)) else ROOT / p


def resolve_bases(cfg, args):
    patterns = as_list(args.variants) or as_list(cfg.get("variants"))
    if patterns:
        dirs = set()
        for p in patterns:
            dirs.update(glob.glob(str(rooted(p))))
        return [(Path(d).name, Path(d) / "config.yaml")
                for d in sorted(dirs) if (Path(d) / "config.yaml").exists()]
    base = args.base or cfg.get("base_config")
    if not base:
        sys.exit("No variants and no base_config given.")
    return [(rooted(base).stem, rooted(base))]


def zone_dir(bcfg):
    zone = str(bcfg.get("zone", "full"))
    return PROCESSED if zone.lower() == "full" else PROCESSED / zone


def emb_tag(s):
    return (f"d{s['dimensions']}_n{s['num_walks']}_l{s['walk_length']}"
            f"_w{s['window_size']}_e{s['epochs']}_neg{s['negative']}")


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
# embedders
# ----------------------------------------------------------------------

def _train_deepwalk_inner(G, emb_path, s, seed):
    """Uniform random walks + Skip-gram with the shared settings."""
    if emb_path.exists():
        return 0.0
    from gensim.models import Word2Vec
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    rng = random.Random(seed)
    nbrs = {n: list(H.neighbors(n)) for n in H.nodes()}
    nodes = list(H.nodes())
    walks = []
    for _ in range(s["num_walks"]):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < s["walk_length"]:
                nb = nbrs[walk[-1]]
                if not nb:
                    break
                walk.append(rng.choice(nb))
            walks.append([str(n) for n in walk])
    model = Word2Vec(walks, vector_size=s["dimensions"], window=s["window_size"],
                     min_count=0, sg=1, hs=0, negative=s["negative"],
                     workers=s["workers"], epochs=s["epochs"], seed=seed)
    tmp = emb_path.with_suffix(f".{os.getpid()}.tmp")
    model.wv.save_word2vec_format(str(tmp))
    shutil.move(str(tmp), str(emb_path))
    dt = time.time() - t0
    print(f"      trained {emb_path.parent.parent.name}/{emb_path.name} in {dt:.0f}s")
    return dt


def _write_int_edgelist(G, path):
    H = nx.Graph(G)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    with open(path, "w") as f:
        for u, v in H.edges():
            f.write(f"{int(u)} {int(v)}\n")
        for n in H.nodes():
            if H.degree(n) == 0:
                f.write(f"{int(n)}\n")   # the struc2vec loader accepts single-node lines


def _train_struc2vec_inner(G, emb_path, layer, s, sc):
    """One isolated copy of the library per run: the library writes its pickles
    next to itself and reloads every `distances_nets_weights-layer-*` pickle it
    finds, so a shared copy leaks stale layers between graphs; it also writes
    random_walks.txt into the working directory."""
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

    # The library trains with hierarchical softmax; switch it to the shared
    # negative-sampling setting.
    main_py = run_dir / "lib" / "main.py"
    src = main_py.read_text()
    old = "hs=1, sg=1"
    if old not in src:
        raise RuntimeError("struc2vec main.py changed: cannot set negative sampling")
    main_py.write_text(src.replace(old, f"hs=0, negative={int(s['negative'])}, sg=1"))

    edgelist = run_dir / "input.edgelist"
    _write_int_edgelist(G, edgelist)
    out = run_dir / "output.emb"
    py = str(rooted(sc["python"]))
    cmd = [py, str(main_py), "--input", str(edgelist), "--output", str(out),
           "--dimensions", str(s["dimensions"]), "--num-walks", str(s["num_walks"]),
           "--walk-length", str(s["walk_length"]), "--window-size", str(s["window_size"]),
           "--iter", str(s["epochs"]), "--workers", str(s["workers"]),
           "--until-layer", str(layer)]
    # OPT flags are parsed with type=bool: any string, even "False", means True.
    for k in ("OPT1", "OPT2", "OPT3"):
        if sc.get(k):
            cmd += [f"--{k}", "1"]

    log = emb_path.with_suffix(".log")
    t0 = time.time()
    with open(log, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        res = subprocess.run(cmd, cwd=str(run_dir), stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"struc2vec failed (exit {res.returncode}); see {log}")
    shutil.move(str(out), str(emb_path))
    if not sc.get("keep_run_dir", False):
        shutil.rmtree(run_dir, ignore_errors=True)
    print(f"      trained {emb_path.parent.parent.name}/{emb_path.name} in {dt:.0f}s")
    return dt


def _emb_lock(emb_path, timeout=14400, poll=5.0):
    """Serialise training of a shared embedding across parallel jobs.

    Two runs can need the same data/processed/<ds>/posthoc/<emb>: without this
    they train it concurrently through the same paths and one reads a partially
    written file.  Returns the lock dir to release, or None if another process
    finished the embedding while we waited.
    """
    lock = emb_path.with_suffix(emb_path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    waited = 0.0
    while True:
        if emb_path.exists():
            return None
        try:
            lock.mkdir()
            return lock
        except FileExistsError:
            if waited == 0.0:
                print(f"      waiting for {emb_path.name} (another job is training it)")
            time.sleep(poll)
            waited += poll
            if waited > timeout:
                raise RuntimeError(f"timed out waiting for {lock}; "
                                   f"remove it by hand if it is stale")


def train_deepwalk(G, emb_path, s, seed):
    if emb_path.exists():
        return 0.0
    lock = _emb_lock(emb_path)
    if lock is None:
        return 0.0
    try:
        return _train_deepwalk_inner(G, emb_path, s, seed)
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def train_struc2vec(G, emb_path, layer, s, sc):
    if emb_path.exists():
        return 0.0
    lock = _emb_lock(emb_path)
    if lock is None:
        return 0.0
    try:
        return _train_struc2vec_inner(G, emb_path, layer, s, sc)
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


# ----------------------------------------------------------------------
# anchors and alignment
# ----------------------------------------------------------------------

def sample(pairs, fraction, seed):
    if fraction is None or fraction >= 1:
        return list(pairs)
    pairs = sorted(pairs)
    k = max(1, int(round(fraction * len(pairs))))
    return random.Random(seed).sample(pairs, k)


def choose_anchors(G1, nodes1, X1, nodes2, X2, bcfg, acfg, radius):
    """Return anchor index pairs (i into nodes1, j into nodes2)."""
    mode = acfg["anchors"]
    frac = acfg.get("anchor_fraction")
    seed = int(acfg.get("anchor_seed", 0))
    idx1 = {n: i for i, n in enumerate(nodes1)}
    idx2 = {n: j for j, n in enumerate(nodes2)}
    roles = {n: d.get("role") for n, d in G1.nodes(data=True)}
    zoned = any(r is not None for r in roles.values())
    scored = {n for n in nodes1 if (roles[n] == "core" or not zoned)}

    if mode == "provenance":
        prov_path = bcfg.get("provenance")
        if not prov_path:
            raise RuntimeError("anchors=provenance needs a variant config with a provenance file")
        with open(rooted(prov_path), "rb") as f:
            inverse = pickle.load(f)["inverse"]          # origin id -> copy id
        pool = [(idx1[a], idx2[b]) for a, b in inverse.items()
                if a in scored and a in idx1 and b in idx2]
        return sample(pool, frac, seed)

    tol = acfg.get("anchor_tolerance_m")
    tol = radius if tol is None else float(tol)
    t2 = cKDTree(X2)
    if mode == "colocated":
        lists = t2.query_ball_point(X1, r=tol)
        pool = [(i, l[0]) for i, l in enumerate(lists) if len(l) == 1 and nodes1[i] in scored]
        count = {}
        for _, j in pool:
            count[j] = count.get(j, 0) + 1
        pool = [(i, j) for i, j in pool if count[j] == 1]
        return sample(pool, frac, seed)
    if mode == "mutual_nn":
        d12, j12 = t2.query(X1, distance_upper_bound=tol)
        d21, i21 = cKDTree(X1).query(X2, distance_upper_bound=tol)
        return [(i, int(j)) for i, (d, j) in enumerate(zip(d12, j12))
                if np.isfinite(d) and j < len(X2) and i21[j] == i]
    raise ValueError(f"unknown anchor mode: {mode}")


def unit(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def align(G1, G2, emb1, emb2, bcfg, acfg, radius):
    nodes1 = [n for n in G1.nodes() if int(n) in emb1]
    nodes2 = [n for n in G2.nodes() if int(n) in emb2]
    X1 = np.array([[G1.nodes[n]["x"], G1.nodes[n]["y"]] for n in nodes1])
    X2 = np.array([[G2.nodes[n]["x"], G2.nodes[n]["y"]] for n in nodes2])
    E1 = np.array([emb1[int(n)] for n in nodes1])
    E2 = np.array([emb2[int(n)] for n in nodes2])
    info = {"no_emb": G1.number_of_nodes() - len(nodes1), "anchor_nodes": []}

    anchors = choose_anchors(G1, nodes1, X1, nodes2, X2, bcfg, acfg, radius)
    info["n_anchors"] = len(anchors)
    if len(anchors) < int(acfg.get("min_anchors", 10)):
        return None, info
    ai = [a for a, _ in anchors]
    bj = [b for _, b in anchors]
    R, _ = orthogonal_procrustes(E1[ai], E2[bj])
    A, B = unit(E1 @ R), unit(E2)
    if acfg.get("exclude_anchors_from_eval", True) and acfg["anchors"] != "mutual_nn":
        info["anchor_nodes"] = [nodes1[i] for i in ai]

    # Candidate set. DeepWalk and struc2vec are structural methods: giving them
    # the spatial radius would hand them the answer set, so by default each node
    # keeps its top_k most similar partners in the aligned embedding space - its
    # own signal, and a set of comparable size to the radius one.
    mode = acfg.get("candidates", "embedding_topk")
    if mode == "radius":
        lists = cKDTree(X2).query_ball_point(X1, r=radius)
        rows = np.fromiter((i for i, l in enumerate(lists) for _ in l), dtype=np.int64)
        cols = np.fromiter((j for l in lists for j in l), dtype=np.int64)
    elif mode == "embedding_topk":
        k = min(int(acfg.get("top_k", 50)), len(nodes2))
        r_parts, c_parts = [], []
        for start in range(0, len(A), 1024):
            S = A[start:start + 1024] @ B.T
            part = np.argpartition(-S, k - 1, axis=1)[:, :k]
            r_parts.append(np.repeat(np.arange(start, start + S.shape[0]), k))
            c_parts.append(part.ravel())
        rows = np.concatenate(r_parts) if r_parts else np.empty(0, dtype=np.int64)
        cols = np.concatenate(c_parts) if c_parts else np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown candidate mode: {mode}")
    info["candidate_pairs"] = int(len(rows))

    matches = []
    if len(rows):
        sims = np.einsum("ij,ij->i", A[rows], B[cols])
        sim_of = dict(zip(zip(rows.tolist(), cols.tolist()), sims.tolist()))
        for i, j in block_assignment(rows, cols, 1.0 - sims):
            i, j = int(i), int(j)
            matches.append({
                "g1_id": nodes1[i], "g2_id": nodes2[j],
                "x_g1": X1[i][0], "y_g1": X1[i][1], "x_g2": X2[j][0], "y_g2": X2[j][1],
                "distance_m": float(np.linalg.norm(X1[i] - X2[j])),
                "similarity": float(sim_of[(i, j)]),
            })
    return matches, info


# ----------------------------------------------------------------------
# evaluation and results
# ----------------------------------------------------------------------

def evaluate(bcfg, base_dir, ds1, ds2, name, exclude_path, quiet):
    cfg = dict(bcfg)
    cfg["dataset_pairs"] = [[ds1, ds2]]
    cfg["compound_graph"] = dict(cfg["compound_graph"], method=name)
    cfg["export_gpkg"] = False
    cfg["export_figures"] = False
    cfg["save_model"] = False
    if exclude_path:
        cfg["eval_exclude"] = str(exclude_path)
    else:
        cfg.pop("eval_exclude", None)
    if str(cfg.get("zone", "full")).lower() != "full":
        cfg["paths"] = dict(cfg["paths"], outputs=str(base_dir / "outputs"))
    cfg_path = base_dir / ds1 / "posthoc" / f"_eval_{name}_{ds2}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "09_evaluate_alignment.py")],
                         cwd=str(ROOT), env=dict(os.environ, SANA_CONFIG=str(cfg_path)),
                         stdout=subprocess.PIPE if quiet else None,
                         stderr=subprocess.STDOUT if quiet else None)
    if res.returncode != 0:
        tail = (res.stdout or b"").decode(errors="replace").strip().splitlines()[-15:]
        raise RuntimeError("step 09 failed:\n      " + "\n      ".join(tail))
    return base_dir / ds1 / f"alignment_evaluation_{name}_{ds2}.pkl"


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


def cross_fit(raw):
    """Parameter chosen on one fold, scored on the other. Recall uses
    recall = TP/(TP+FN), matching run_sweep.cross_fit(). Ties go to the first value."""
    groups = {}
    for r in raw:
        if r["status"] == "ok":
            groups.setdefault((r["variant"], r["dataset1"], r["dataset2"], r["method"]), []).append(r)
    out = []
    for (variant, d1, d2, m), rs in sorted(groups.items()):
        idx = {(r["param"], r["fold"]): r for r in rs}
        params = sorted({r["param"] for r in rs}, key=lambda v: (len(v), v))

        def best_on(fold):
            best, bf = None, -1.0
            for p in params:
                r = idx.get((p, fold))
                if r and float(r["f1"]) > bf:
                    best, bf = p, float(r["f1"])
            return best

        tot = {"TP": 0, "FP": 0, "FN": 0}
        chosen = {}
        for fold, other in (("A", "B"), ("B", "A")):
            chosen[fold] = best_on(other)
            r = idx.get((chosen[fold], fold))
            if r:
                for k in tot:
                    tot[k] += int(r[k])
        tp, fp, fn = tot["TP"], tot["FP"], tot["FN"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"variant": variant, "dataset1": d1, "dataset2": d2, "method": m,
                    "param_A": chosen.get("A"), "param_B": chosen.get("B"),
                    "param_best_all": best_on("all"), **tot,
                    "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)})
    return out


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------

def do_run(task, method, s, sc, acfg, tag, seed, verbose):
    """Train (if needed), align, evaluate one (variant, pair, param). Returns rows."""
    label, cpath, ds1, ds2, p = task
    bcfg = load_yaml(cpath)
    bdir = zone_dir(bcfg)
    radius = float(bcfg["spatial"]["radius"])
    name = "DeepWalk" if method == "deepwalk" else f"Struc2vec_L{p}"
    print(f"\n[{label} {ds1}->{ds2} {name}] radius={radius:g} m", flush=True)
    t0 = time.time()
    row = {"variant": label, "dataset1": ds1, "dataset2": ds2, "method": method,
           "param": p, "radius_m": radius, "anchor_mode": acfg["anchors"]}
    try:
        with open(bdir / ds1 / "Graph.pkl", "rb") as f:
            G1 = pickle.load(f)
        with open(bdir / ds2 / "Graph.pkl", "rb") as f:
            G2 = pickle.load(f)
        train_s, embs = 0.0, []
        for ds, G in ((ds1, G1), (ds2, G2)):
            if method == "deepwalk":
                ep = bdir / ds / "posthoc" / f"deepwalk_{tag}_seed{seed}.emb"
                train_s += train_deepwalk(G, ep, s, seed)
            else:
                ep = bdir / ds / "posthoc" / f"struc2vec_L{p}_{tag}.emb"
                train_s += train_struc2vec(G, ep, int(p), s, sc)
            embs.append(read_embeddings(ep))

        matches, info = align(G1, G2, embs[0], embs[1], bcfg, acfg, radius)
        row.update(n_anchors=info["n_anchors"], no_embedding_g1=info["no_emb"],
                   train_seconds=round(train_s, 1))
        if matches is None:
            print(f"      skipped: only {info['n_anchors']} anchors", flush=True)
            return [dict(row, fold="all", status="too_few_anchors",
                         seconds=round(time.time() - t0, 1))]
        print(f"      anchors {info['n_anchors']:,}  cand {info.get('candidate_pairs', 0):,}  matches {len(matches):,}  "
              f"G1 nodes without embedding {info['no_emb']}", flush=True)

        with open(bdir / ds1 / f"alignment_hungarian_{name}_{ds2}.pkl", "wb") as f:
            pickle.dump({"matches": matches, "method": name, "dataset1": ds1, "dataset2": ds2,
                         "anchor_nodes": info["anchor_nodes"],
                         "parameters": {"param": p, "radius": radius, **s, **acfg}}, f)
        excl = None
        if info["anchor_nodes"]:
            excl = bdir / ds1 / "posthoc" / f"anchors_{name}_{ds2}.pkl"
            with open(excl, "wb") as f:
                pickle.dump(info["anchor_nodes"], f)

        eval_pkl = evaluate(bcfg, bdir, ds1, ds2, name, excl, quiet=not verbose)

        fp = bdir / "folds.pkl"
        if fp.exists() and str(bcfg.get("zone", "full")).lower() != "full":
            with open(fp, "rb") as f:
                folds = pickle.load(f)
        else:
            folds = SWEEP.make_folds(bdir / ds1 / "Graph.pkl")
        fm = SWEEP.fold_metrics(eval_pkl, folds)
        secs = round(time.time() - t0, 1)
        rows = []
        for fold in ("A", "B", "all"):
            r = fm[fold]
            rows.append(dict(row, fold=fold, status="ok", seconds=secs,
                             gt_mode=fm["_meta"]["gt_mode"],
                             expected_no_match=fm["_meta"]["expected_no_match"],
                             false_alarms=fm["_meta"]["false_alarms"],
                             **{kk: (round(v, 6) if isinstance(v, float) else v)
                                for kk, v in r.items()}))
        a = fm["all"]
        print(f"      F1 {a['f1']:.4f}  TP {a['TP']:,}  FP {a['FP']:,}  FN {a['FN']:,}  "
              f"(scored {a['n_eval']:,})  {secs:.0f}s", flush=True)
        return rows
    except Exception as e:
        print(f"      FAILED: {e}", flush=True)
        return [dict(row, fold="all", status=f"error: {str(e)[:120]}",
                     seconds=round(time.time() - t0, 1))]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["deepwalk", "struc2vec"])
    ap.add_argument("--variants", nargs="+", help="variant directories (glob patterns)")
    ap.add_argument("--base", help="base config when no variants are used")
    ap.add_argument("--layers", nargs="+", type=int, help="struc2vec until_layer values")
    ap.add_argument("--params-from", help="a *_selected.csv from the tuning seed: each variant "
                                          "reuses the parameter chosen for its (family, level), "
                                          "instead of running the whole grid")
    ap.add_argument("--pairs", nargs="+", metavar="DS1,DS2",
                    help="restrict to these dataset pairs, e.g. Dataset4,Dataset5 "
                         "(order-insensitive); default is every pair in the base config.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="runs to process in parallel (1 = sequential). Keep "
                         "jobs x shared.workers near the core count.")
    args = ap.parse_args()

    os.chdir(ROOT)
    cfg = load_yaml(rooted(args.config))
    s = cfg["shared"]
    acfg = cfg["alignment"]
    if args.method == "struc2vec":
        sc = cfg["struc2vec"]
        if not sc.get("OPT3"):
            sys.exit("struc2vec.OPT3 must be true: the library ignores until_layer otherwise.")
        params = [str(v) for v in (args.layers or as_list(sc["until_layer"]))]
    else:
        sc = None
        params = ["-"]

    out_dir = rooted(cfg.get("out_dir", "results"))
    raw_path = out_dir / f"{cfg.get('out_prefix', 'posthoc')}_{args.method}_raw.csv"
    sel_path = out_dir / f"{cfg.get('out_prefix', 'posthoc')}_{args.method}_selected.csv"

    bases = resolve_bases(cfg, args)
    if not bases:
        sys.exit("No base configs / variant directories matched.")
    done = set()
    if not args.force:
        for r in read_rows(raw_path):
            if r["status"] in ("ok", "too_few_anchors"):
                done.add((r["variant"], r["dataset1"], r["dataset2"], r["param"]))
    # Reuse the tuned parameter: variants of the same (family, level) differ only
    # by their seed, so the value selected on the tuning seed applies to all.
    tuned = {}
    if args.params_from:
        import re as _re
        for r in read_rows(rooted(args.params_from)):
            m = _re.match(r".+__(.+)_L([\dp]+)_s\d+$", r["variant"])
            if m and r.get("param_best_all"):
                tuned[(m.group(1), m.group(2))] = str(r["param_best_all"])

    def params_for(label):
        if not tuned:
            return params
        import re as _re
        m = _re.match(r".+__(.+)_L([\dp]+)_s\d+$", label)
        v = tuned.get((m.group(1), m.group(2))) if m else None
        return [v] if v else params

    wanted = None
    if args.pairs:
        known = {frozenset(pr) for _, cpath in bases
                 for pr in load_yaml(cpath)["dataset_pairs"]}
        wanted = set()
        for spec in args.pairs:
            parts = [x.strip() for x in spec.split(",") if x.strip()]
            if len(parts) != 2:
                sys.exit(f"--pairs expects DS1,DS2 (got {spec!r})")
            key = frozenset(parts)
            if key not in known:
                sys.exit(f"--pairs {spec!r} is not a pair in the base config")
            wanted.add(key)

    plan = []
    for label, cpath in bases:
        for ds1, ds2 in load_yaml(cpath)["dataset_pairs"]:
            if wanted is not None and frozenset((ds1, ds2)) not in wanted:
                continue
            for p in params_for(label):
                if (label, ds1, ds2, p) not in done:
                    plan.append((label, cpath, ds1, ds2, p))
    if wanted is not None and not plan and not done:
        sys.exit(f"--pairs matched nothing in {[c for _, c in bases]}")

    print(f"{args.method}: {len(bases)} base config(s), params {params}, {len(plan)} run(s) to do")
    print(f"  anchors: {acfg['anchors']} (fraction {acfg.get('anchor_fraction')})  -> {raw_path.relative_to(ROOT)}")
    if args.dry_run:
        for label, _, ds1, ds2, p in plan:
            print(f"  {label:32s} {ds1}->{ds2}  param={p}")
        return

    tag = emb_tag(s)
    seed = int(s.get("seed", 42))
    common = dict(method=args.method, s=s, sc=sc, acfg=acfg, tag=tag, seed=seed,
                  verbose=args.verbose)

    if args.jobs <= 1:
        for k, task in enumerate(plan, 1):
            print(f"\n[{k}/{len(plan)}]", end="")
            append_rows(raw_path, do_run(task, **common))
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f"  running {args.jobs} at a time")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(do_run, t, **common) for t in plan]
            for fut in futs:
                append_rows(raw_path, fut.result())

    sel = cross_fit(read_rows(raw_path))
    with open(sel_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEL_FIELDS)
        w.writeheader()
        w.writerows(sel)
    print(f"\n✓ {len(sel)} cross-fitted result(s) -> {sel_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
