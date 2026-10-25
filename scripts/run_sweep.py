"""
Sweep driver for the Protocol A stress protocol.

Builds every variant, runs every method over the hyperparameter grids, and
appends one row per (variant, method, sigma, alpha, fold) to a results CSV.

Design notes
------------
COST ASYMMETRY. sigma enters at step 04 (compound-graph cross-edge weights), so
each value invalidates the walks and embeddings and costs a full pipeline run.
alpha enters only at step 08, so it costs a few seconds. The loops are nested
accordingly: sigma outer, alpha inner, and step 02 runs once per variant since
the candidate radius is fixed by the coverage rule.

FAIR TUNING. sigma and alpha are selected by 2-fold CROSS-FITTING over a spatial
split of the zone core: each fold is scored under parameters chosen on the other
fold, so no node is ever scored under parameters fitted on itself, and all core
nodes still contribute. The split is spatial, not random - a random node split
leaks heavily, since a tuning node's neighbours land in the evaluation fold.
sigma is tuned on one seed and reused for the others (tuning.tune_on_seed),
which keeps the cost near 2x rather than 6x.

REPRODUCIBILITY. random_walks.seed is varied per variant seed; leaving it fixed
would make every seed share identical walk randomness and understate variance.
gensim with workers > 1 stays non-deterministic, so seed-to-seed spread includes
embedding-initialisation variance. Report it as such.

ISOLATION. paths.outputs is redirected into the variant directory: steps 08/09
name their figures by method and dataset only, which are identical for every
variant, so a shared outputs/ directory would have each variant overwrite the last.

Usage
-----
    python scripts/run_sweep.py --dry-run
    python scripts/run_sweep.py --resume
    python scripts/run_sweep.py --families warp --levels 0.25 1.0 --seeds 1
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

RAW_FIELDS = [
    "variant", "family", "level", "seed", "method", "sigma", "alpha", "fold",
    "n_eval", "TP", "FP", "FN", "precision", "recall", "f1",
    "radius_m", "gt_mode", "expected_no_match", "false_alarms", "seconds",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def run_step(script, cfg_path, quiet=True):
    """Run one pipeline step with SANA_CONFIG pointed at cfg_path."""
    env = dict(os.environ, SANA_CONFIG=str(cfg_path))
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    dt = time.time() - t0
    if res.returncode != 0:
        out = (res.stdout or b"").decode(errors="replace")
        print(f"      FAILED {script} (exit {res.returncode})")
        print("      " + "\n      ".join(out.strip().splitlines()[-15:]))
        return False, dt
    return True, dt


def write_config(cfg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def make_folds(graph_path, split="x_median"):
    """Spatial 2-fold split of the core nodes. Returns {node: 'A'|'B'}."""
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    core = [n for n, d in G.nodes(data=True) if d.get("role", "core") == "core"]
    xs = np.array([G.nodes[n]["x"] for n in core])
    cut = np.median(xs)
    return {n: ("A" if G.nodes[n]["x"] < cut else "B") for n in core}


def fold_metrics(eval_pkl, folds):
    """Recompute P/R/F1 per fold from a step-09 evaluation pickle."""
    with open(eval_pkl, "rb") as f:
        ev = pickle.load(f)
    correct = set(ev.get("correct_matches", []))
    wrong = {m["g1_id"] for m in ev.get("incorrect_matches", [])}
    missed = set(ev.get("missed_anchors", []))

    out = {}
    for fold in ("A", "B", "all"):
        sel = (lambda n: True) if fold == "all" else (lambda n, f=fold: folds.get(n) == f)
        TP = sum(1 for n in correct if sel(n))
        FP = sum(1 for n in wrong if sel(n))
        FN = sum(1 for n in missed if sel(n))
        # Standard recall, matching cross_fit() below: TP/(TP+FN). Under the
        # complete Hungarian assignment used here (FN=0 whenever
        # min_similarity=0), this reads as recall 1.0 and f1 collapses to
        # precision -- expected, not a bug; see subsec:control.
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[fold] = dict(n_eval=TP + FP + FN, TP=TP, FP=FP, FN=FN,
                         precision=prec, recall=rec, f1=f1)
    out["_meta"] = dict(
        gt_mode=ev.get("ground_truth_mode", "spatial"),
        expected_no_match=ev.get("expected_no_match", 0),
        false_alarms=ev.get("false_alarms", 0),
    )
    return out


def as_num(v):
    """CSV values come back as text; sigma and K must reach the pipeline as
    numbers (sigma multiplies a float, K indexes an array size)."""
    if isinstance(v, str):
        return int(v) if v.strip().lstrip("-").isdigit() else float(v)
    return v


def fmt_level(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")


# ----------------------------------------------------------------------
# one variant
# ----------------------------------------------------------------------

def build_variant(base_cfg, zone, family, level, seed, params, processed):
    """Run 01c for this cell and return (variant_dir, variant_config)."""
    cfg = dict(base_cfg)
    cfg["zone"] = zone
    cfg["perturbation"] = dict(base_cfg.get("perturbation") or {})
    cfg["perturbation"].update(family=family, level=float(level), seed=int(seed))
    if params:
        cfg["perturbation"][family] = dict(params)

    variant_name = f"{zone}__{family}_L{fmt_level(float(level))}_s{int(seed):02d}"
    tmp = processed / f"_sweep_tmp_build_{variant_name}.yaml"
    write_config(cfg, tmp)
    ok, _ = run_step("01c_perturb.py", tmp)
    if not ok:
        return None, None

    vdir = processed / variant_name
    with open(vdir / "config.yaml") as f:
        vcfg = yaml.safe_load(f)
    # keep every variant's figures and geopackages inside its own directory
    vcfg["paths"] = dict(vcfg["paths"])
    vcfg["paths"]["outputs"] = str(vdir / "outputs")
    # vary walk randomness with the seed, otherwise all seeds share it
    vcfg["random_walks"] = dict(vcfg["random_walks"])
    vcfg["random_walks"]["seed"] = 1000 + int(seed)
    # visualisation artifacts are not results; 850 runs do not need them
    for k in ("export_gpkg", "export_figures", "save_model", "save_walk_stats"):
        vcfg[k] = False
    tmp.unlink(missing_ok=True)
    return vdir, vcfg


def run_variant(vdir, vcfg, grids, methods, folds, rows, meta, resume_keys, quiet):
    variant = vdir.name
    clean, pert = vcfg["dataset_pairs"][0]
    radius = vcfg["spatial"]["radius"]

    def key(m, s, a):
        return (variant, m, str(s), str(a))

    def record(m, s, a, eval_pkl, secs):
        fm = fold_metrics(eval_pkl, folds)
        for fold in ("A", "B", "all"):
            r = fm[fold]
            rows.append({
                "variant": variant, "family": meta["family"], "level": meta["level"],
                "seed": meta["seed"], "method": m, "sigma": s, "alpha": a, "fold": fold,
                "radius_m": radius, "seconds": round(secs, 1),
                "gt_mode": fm["_meta"]["gt_mode"],
                "expected_no_match": fm["_meta"]["expected_no_match"],
                "false_alarms": fm["_meta"]["false_alarms"],
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()},
            })

    # ---- spatial candidates: once per variant (radius is fixed by the rule) ----
    cfg_path = vdir / "_run.yaml"
    write_config(vcfg, cfg_path)
    ok, _ = run_step("02_spatial_candidates.py", cfg_path, quiet)
    if not ok:
        return

    # ---- nearest-neighbour baseline: no sigma, no alpha ----
    if "NearestNeighbor" in methods and key("NearestNeighbor", "", "") not in resume_keys:
        ok, dt = run_step("main_nearest_neighbor.py", cfg_path, quiet)
        if ok:
            p = vdir / clean / f"alignment_evaluation_NearestNeighbor_{pert}.pkl"
            if p.exists():
                record("NearestNeighbor", "", "", p, dt)
                print(f"    NearestNeighbor: f1={rows[-1]['f1']:.4f}")

    # ---- CENA: one pipeline per K (structural similarity, radius-limited) ----
    # K is recorded in the "sigma" column: it is CENA's single tuned parameter,
    # so the selection machinery below needs no special case.
    if "CENA" in methods:
        for K in grids.get("cena_K", [vcfg["structural"]["K"]]):
            if key("CENA", K, "") in resume_keys:
                continue
            cfg = dict(vcfg)
            cfg["structural"] = dict(vcfg["structural"], K=K)
            cfg["compound_graph"] = dict(vcfg["compound_graph"], method="CENA")
            write_config(cfg, cfg_path)
            t0 = time.time()
            for step in ("03_structural_similarity.py", "04_build_compound_graph.py",
                         "05_generate_random_walks.py", "06_generate_embeddings.py",
                         "08_hungarian_alignment.py", "09_evaluate_alignment.py"):
                ok, _ = run_step(step, cfg_path, quiet)
                if not ok:
                    break
            else:
                p = vdir / clean / f"alignment_evaluation_CENA_{pert}.pkl"
                if p.exists():
                    record("CENA", K, "", p, time.time() - t0)
                    print(f"    CENA         K={K:<3} f1={rows[-1]['f1']:.4f}")
        write_config(vcfg, cfg_path)      # restore the variant's own config

    if "SANA" not in methods:
        return

    # ---- sigma outer (expensive), alpha inner (cheap) ----
    for sigma in grids["sigma"]:
        cfg = dict(vcfg)
        cfg["compound_graph"] = dict(vcfg["compound_graph"])
        cfg["compound_graph"]["gaussian"] = dict(vcfg["compound_graph"]["gaussian"])
        cfg["compound_graph"]["gaussian"]["sigma"] = sigma
        write_config(cfg, cfg_path)

        t0 = time.time()
        for step in ("04_build_compound_graph.py", "05_generate_random_walks.py",
                     "06_generate_embeddings.py"):
            ok, _ = run_step(step, cfg_path, quiet)
            if not ok:
                break
        else:
            embed_secs = time.time() - t0
            for alpha in grids["alpha"]:
                # alpha = 0 discards the embeddings entirely, so the result does
                # not depend on sigma: it is the spatial-only Hungarian baseline,
                # not a SANA setting. Run it once and record it under its own
                # name - otherwise it is duplicated once per sigma AND can win
                # the sigma selection below, pinning every later seed to whichever
                # sigma happened to come first.
                spatial_only = float(alpha) == 0.0
                if spatial_only and sigma != grids["sigma"][0]:
                    continue
                m_name = "SpatialOnly" if spatial_only else "SANA"
                m_sigma = "" if spatial_only else sigma
                if key(m_name, m_sigma, alpha) in resume_keys:
                    continue
                cfg["alignment"] = dict(vcfg["alignment"])
                cfg["alignment"]["alpha"] = alpha
                cfg["alignment"]["beta"] = round(1.0 - float(alpha), 6)
                write_config(cfg, cfg_path)
                ok1, t1 = run_step("08_hungarian_alignment.py", cfg_path, quiet)
                ok2, t2 = run_step("09_evaluate_alignment.py", cfg_path, quiet)
                if not (ok1 and ok2):
                    continue
                p = vdir / clean / f"alignment_evaluation_SANA_{pert}.pkl"
                if p.exists():
                    record(m_name, m_sigma, alpha, p, embed_secs + t1 + t2)
                    print(f"    {m_name:<12} sigma={str(m_sigma):<3} alpha={alpha:<4} "
                          f"f1={rows[-1]['f1']:.4f}")


# ----------------------------------------------------------------------
# cross-fitted selection
# ----------------------------------------------------------------------

def cross_fit(rows):
    """For each (variant, method): pick params on one fold, score on the other."""
    by = {}
    for r in rows:
        by.setdefault((r["variant"], r["method"]), []).append(r)

    out = []
    for (variant, method), rs in sorted(by.items()):
        # resumed rows carry strings while fresh rows carry numbers: compare as
        # text so the two never meet in a sort
        params = sorted({(str(r["sigma"]), str(r["alpha"])) for r in rs})
        idx = {(str(r["sigma"]), str(r["alpha"]), r["fold"]): r for r in rs}
        total = {"TP": 0, "FP": 0, "FN": 0}
        chosen = {}
        for fold, other in (("A", "B"), ("B", "A")):
            best, best_f1 = None, -1.0
            for p in params:
                r = idx.get((str(p[0]), str(p[1]), other))
                if r and r["f1"] > best_f1:
                    best, best_f1 = p, r["f1"]
            if best is None:
                continue
            chosen[fold] = best
            r = idx.get((str(best[0]), str(best[1]), fold))
            if r:
                for k in total:
                    total[k] += r[k]
        prec = total["TP"] / (total["TP"] + total["FP"]) if total["TP"] + total["FP"] else 0.0
        rec = total["TP"] / (total["TP"] + total["FN"]) if total["TP"] + total["FN"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ex = rs[0]
        out.append({
            "variant": variant, "family": ex["family"], "level": ex["level"],
            "seed": ex["seed"], "method": method,
            "sigma_A": chosen.get("A", ("", ""))[0], "alpha_A": chosen.get("A", ("", ""))[1],
            "sigma_B": chosen.get("B", ("", ""))[0], "alpha_B": chosen.get("B", ("", ""))[1],
            **total,
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        })
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def do_cell(cell, base_cfg, zone, processed, grids, methods, sweep,
            resume_keys, quiet, sigma_override=None, cena_override=None):
    """Build one variant and run every method on it. Returns (cell, rows)."""
    family, level, seed, params = cell
    tag = f"{family} L{level} s{seed}"
    print(f"\n=== {tag} ===", flush=True)
    vdir, vcfg = build_variant(base_cfg, zone, family, level, seed, params, processed)
    if vdir is None:
        print(f"    [{tag}] variant build FAILED", flush=True)
        return cell, []

    folds = make_folds(vdir / vcfg["dataset_pairs"][0][0] / "Graph.pkl",
                       sweep.get("tuning", {}).get("split", "x_median"))
    with open(vdir / "folds.pkl", "wb") as f:
        pickle.dump(folds, f)

    g = dict(grids)
    if sigma_override is not None:
        g["sigma"] = sigma_override
    if cena_override is not None:
        g["cena_K"] = cena_override
    rows = []
    run_variant(vdir, vcfg, g, methods, folds, rows,
                {"family": family, "level": level, "seed": seed}, resume_keys, quiet)
    return cell, rows


def pick_cena_K(rows):
    """K chosen for this cell, reused by the other seeds like sigma."""
    new = [r for r in rows if r["method"] == "CENA" and r["fold"] != "all"]
    return [as_num(max(new, key=lambda r: r["f1"])["sigma"])] if new else None


def pick_sigma(rows):
    """sigma chosen for this cell: best fold row among the rows that use the
    embeddings (SANA only; SpatialOnly and CENA do not depend on sigma)."""
    new = [r for r in rows if r["method"] == "SANA" and r["fold"] != "all"]
    return [as_num(max(new, key=lambda r: r["f1"])["sigma"])] if new else None


def write_raw(raw_csv, rows):
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="config/sweep.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--levels", nargs="*", type=float)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="variants to process in parallel (1 = sequential). Keep "
                         "jobs x embeddings.workers near the core count.")
    args = ap.parse_args()

    with open(ROOT / args.sweep) as f:
        sweep = yaml.safe_load(f)
    with open(ROOT / sweep["base_config"]) as f:
        base_cfg = yaml.safe_load(f)

    zone = sweep["zone"]
    processed = ROOT / base_cfg["paths"]["data_processed"]
    grids = sweep["grids"]
    methods = sweep["methods"]
    seeds = args.seeds or sweep["seeds"]
    tune_seed = sweep.get("tuning", {}).get("tune_on_seed", seeds[0])

    cells = []
    for family, spec in sweep["families"].items():
        if args.families and family not in args.families:
            continue
        for level in spec["levels"]:
            if args.levels and level not in args.levels:
                continue
            for seed in seeds:
                cells.append((family, level, seed, spec.get("params")))
    # the tuning seed must run first: it decides sigma for the others
    cells.sort(key=lambda c: (c[2] != tune_seed, c[0], c[1], c[2]))

    n_sig, n_alp = len(grids["sigma"]), len(grids["alpha"])
    tuned = sum(1 for c in cells if c[2] == tune_seed)
    others = len(cells) - tuned
    print(f"Sweep: {len(cells)} variants  ({tuned} at the tuning seed, {others} others)")
    print(f"  sigma grid: {grids['sigma']}   alpha grid: {grids['alpha']}")
    n_K = len(grids.get("cena_K", [])) if "CENA" in methods else 0
    print(f"  pipeline runs (04-06): ~{tuned * n_sig + others} "
          + (f"+ {tuned * n_K + others} for CENA" if n_K else ""))
    print(f"  assignment runs (08-09): ~{(tuned * n_sig + others) * n_alp}")
    if args.dry_run:
        for f, l, s, _ in cells:
            print(f"    {zone}__{f}_L{fmt_level(l)}_s{s:02d}")
        return

    out_dir = ROOT / sweep.get("out_dir", "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "sweep_raw.csv"

    rows, resume_keys = [], set()
    if args.resume and raw_csv.exists():
        with open(raw_csv) as f:
            for r in csv.DictReader(f):
                # CSV gives strings; cross_fit compares and sums these numerically
                for k in ("n_eval", "TP", "FP", "FN", "expected_no_match", "false_alarms"):
                    r[k] = int(r[k]) if r.get(k) not in (None, "") else 0
                # Derived metrics are recomputed from the counts rather than
                # trusted: rows written under the old TP/(TP+FP+FN) recall
                # convention carry deflated F1 relative to the current
                # TP/(TP+FN) convention (matches cross_fit()).
                TP, FP, FN = r["TP"], r["FP"], r["FN"]
                pr = TP / (TP + FP) if TP + FP else 0.0
                rc = TP / (TP + FN) if TP + FN else 0.0
                r["precision"] = round(pr, 6)
                r["recall"] = round(rc, 6)
                r["f1"] = round(2 * pr * rc / (pr + rc) if pr + rc else 0.0, 6)
                rows.append(r)
                resume_keys.add((r["variant"], r["method"], r["sigma"], r["alpha"]))
        print(f"  resuming: {len(resume_keys)} rows already present")

    # A resumed run returns no new rows for the tuning seed, so its sigma choice
    # would be lost and every other seed would re-run the whole sigma grid.
    # Rebuild it from the rows already on disk.
    resumed_sigma, resumed_K = {}, {}
    for r in rows:
        if r["fold"] == "all" or int(r["seed"]) != tune_seed:
            continue
        store = resumed_sigma if r["method"] == "SANA" else (resumed_K if r["method"] == "CENA" else None)
        if store is None:
            continue
        k = (r["family"], float(r["level"]))
        best = store.get(k)
        if best is None or float(r["f1"]) > best[1]:
            store[k] = (r["sigma"], float(r["f1"]))
    resumed_sigma = {k: [as_num(v[0])] for k, v in resumed_sigma.items()}
    resumed_K = {k: [as_num(v[0])] for k, v in resumed_K.items()}
    if resumed_sigma or resumed_K:
        print(f"  reused from the tuning seed: sigma for {len(resumed_sigma)} cells, "
              f"K for {len(resumed_K)} cells")

    common = dict(base_cfg=base_cfg, zone=zone, processed=processed, grids=grids,
                  methods=methods, sweep=sweep, resume_keys=resume_keys,
                  quiet=not args.verbose)

    if args.jobs <= 1:
        sigma_choice, K_choice = dict(resumed_sigma), dict(resumed_K)
        for cell in cells:
            tuning = cell[2] == tune_seed
            _, new = do_cell(cell,
                             sigma_override=None if tuning else sigma_choice.get((cell[0], cell[1])),
                             cena_override=None if tuning else K_choice.get((cell[0], cell[1])),
                             **common)
            rows.extend(new)
            if tuning:
                for pick, store, name in ((pick_sigma(new), sigma_choice, "sigma"),
                                          (pick_cena_K(new), K_choice, "K")):
                    if pick:
                        store[(cell[0], cell[1])] = pick
                        print(f"    {name} selected for this cell: {pick[0]}")
            write_raw(raw_csv, rows)
    else:
        # Two phases: the tuning seed first, because its sigma is reused by the
        # other seeds. Within a phase the cells are independent.
        from concurrent.futures import ProcessPoolExecutor
        phase1 = [c for c in cells if c[2] == tune_seed]
        phase2 = [c for c in cells if c[2] != tune_seed]
        sigma_choice, K_choice = dict(resumed_sigma), dict(resumed_K)
        for phase, label in ((phase1, "tuning seed"), (phase2, "other seeds")):
            if not phase:
                continue
            print(f"\n--- {label}: {len(phase)} variants, {args.jobs} at a time ---")
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futs = [pool.submit(do_cell, c,
                                    sigma_override=(None if c[2] == tune_seed
                                                    else sigma_choice.get((c[0], c[1]))),
                                    cena_override=(None if c[2] == tune_seed
                                                   else K_choice.get((c[0], c[1]))),
                                    **common)
                        for c in phase]
                for fut in futs:
                    cell, new = fut.result()
                    rows.extend(new)
                    if cell[2] == tune_seed:
                        for pick, store in ((pick_sigma(new), sigma_choice),
                                            (pick_cena_K(new), K_choice)):
                            if pick:
                                store[(cell[0], cell[1])] = pick
                    write_raw(raw_csv, rows)

    sel = cross_fit([r for r in rows if r["fold"] != "all"])
    sel_csv = out_dir / "sweep_selected.csv"
    if sel:
        with open(sel_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
            w.writeheader()
            w.writerows(sel)

    print(f"\nRaw rows:      {raw_csv}  ({len(rows)})")
    print(f"Cross-fitted:  {sel_csv}  ({len(sel)})")


if __name__ == "__main__":
    main()
