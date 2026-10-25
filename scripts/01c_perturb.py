"""
Step 1c: Build one perturbed variant of a carved zone (Protocol A).

PROTOCOL A - self-alignment
---------------------------
The zone graph is aligned against a PERTURBED COPY OF ITSELF. The provenance map
is then the entire ground truth: exact, covering every node, and with nothing
spatial about it. That is what makes a comparison against spatial-only baselines
fair - under the 0.1 m co-location ground truth, a nearest-neighbour baseline is
scored on the very criterion that defines the labels.

Because the copy starts out identical, Protocol A is EASIER than the real
Dataset1/Dataset5 pair at low noise. Absolute F1 here is not comparable to the
real-pair result; only comparisons BETWEEN METHODS AT THE SAME NOISE LEVEL are.

Output (one directory per variant):

    <data_processed>/<ZONE>__<family>_L<level>_s<seed>/
        Dataset1/     clean zone graph  (unchanged, original ids)
        Dataset1P/    perturbed copy    (fresh ids)
        provenance.pkl
        perturb_manifest.json
        config.yaml   base config with zone / dataset_pairs / radius overridden

CANDIDATE RADIUS
----------------
The radius is set by a COVERAGE RULE, not by tuning: the smallest radius (rounded
up to 5 m, never below the configured base) such that >= 99% of true
correspondences fall inside the candidate gate at this noise level. Applied
identically to every method, this stops the 20 m gate from silently becoming the
experiment once displacement approaches the node spacing.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.perturb import FAMILIES, apply_perturbation


# ============================================
# LOAD CONFIGURATION
# ============================================

print("=" * 60)
print("STEP 1c: PERTURB ZONE  (Protocol A - self-alignment)")
print("=" * 60)

config_path = Path(os.environ.get("SANA_CONFIG", "config/default.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

ZONE = config.get("zone", "full")
pert = config.get("perturbation") or {}

FAMILY = pert.get("family", "identity")
LEVEL = float(pert.get("level", 0.0))
SEED = int(pert.get("seed", 1))
SOURCE = pert.get("source_dataset", "Dataset1")
CLEAN_NAME = pert.get("clean_name", SOURCE)
PERT_NAME = pert.get("perturbed_name", SOURCE + "P")
ID_OFFSET = int(pert.get("id_offset", 10_000_000))
FAMILY_PARAMS = dict(pert.get(FAMILY) or {})
COVERAGE = float(pert.get("gt_coverage", 0.99))

if FAMILY not in FAMILIES:
    print(f"  ERROR: unknown family '{FAMILY}'. Known: {FAMILIES}")
    sys.exit(1)

PROCESSED_DATA_DIR = Path(config["paths"]["data_processed"])
SOURCE_DIR = PROCESSED_DATA_DIR / ZONE / SOURCE
BASE_RADIUS = float(config["spatial"]["radius"])


def fmt_level(v):
    return f"{v:g}".replace(".", "p").replace("-", "m")


VARIANT = f"{ZONE}__{FAMILY}_L{fmt_level(LEVEL)}_s{SEED:02d}"
OUT_DIR = PROCESSED_DATA_DIR / VARIANT
SUFFIX = f"_{VARIANT.lower()}"

print(f"\nConfiguration:")
print(f"  Config file: {config_path}")
print(f"  Source zone: {ZONE}   dataset: {SOURCE}")
print(f"  Family: {FAMILY}   level: {LEVEL}   seed: {SEED}")
if FAMILY_PARAMS:
    print(f"  Family params: {FAMILY_PARAMS}")
print(f"  Variant: {VARIANT}")
print(f"  Output: {OUT_DIR}")

if not (SOURCE_DIR / "Graph.pkl").exists():
    print(f"\n  ERROR: source graph not found: {SOURCE_DIR / 'Graph.pkl'}")
    print(f"  Run scripts/01b_carve_zone.py first.")
    sys.exit(1)


# ============================================
# LOAD SOURCE GRAPH + CALIBRATE THE LEVEL
# ============================================

with open(SOURCE_DIR / "Graph.pkl", "rb") as f:
    G = pickle.load(f)

n_core = sum(1 for _, d in G.nodes(data=True) if d.get("role", "core") == "core")
print(f"\nSource graph: {G.number_of_nodes():,} nodes "
      f"({n_core:,} core), {G.number_of_edges():,} edges")

# median nearest-neighbour spacing s: the natural scale for geometric noise, so
# levels transfer to any other network instead of being arbitrary metres.
nodes = list(G.nodes())
xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
dists, _ = cKDTree(xy).query(xy, k=2)
SPACING = float(np.median(dists[:, 1]))
print(f"Median node spacing s = {SPACING:.2f} m")
if FAMILY in ("jitter", "warp"):
    print(f"  level {LEVEL} x s  ->  {LEVEL * SPACING:.2f} m")
elif FAMILY == "composite":
    for st in (FAMILY_PARAMS.get("steps") or []):
        lv = st.get("level", "@level")
        lv = LEVEL if isinstance(lv, str) else lv
        unit = "m" if st["family"] in ("jitter", "warp") else "of edges"
        val = f"{lv * SPACING:.2f}" if st["family"] in ("jitter", "warp") else f"{lv:g}"
        print(f"    step: {st['family']:<9} {val} {unit}")


# ============================================
# PERTURB
# ============================================

print(f"\n{'=' * 60}")
print(f"APPLYING PERTURBATION: {FAMILY}")
print(f"{'=' * 60}")

H, prov = apply_perturbation(
    G, family=FAMILY, level=LEVEL, seed=SEED,
    node_spacing_m=SPACING, id_offset=ID_OFFSET, params=FAMILY_PARAMS,
)

prov.update({
    "protocol": "A_self_alignment",
    "zone": ZONE,
    "variant": VARIANT,
    "clean_dataset": CLEAN_NAME,
    "perturbed_dataset": PERT_NAME,
    "source_dataset": SOURCE,
})

print(f"  Perturbed graph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")
print(f"  Surviving: {sum(1 for o in prov['origin'].values() if o is not None):,}")
print(f"  Invented (no correct match exists): {len(prov['invented']):,}")
print(f"  Deleted origins (expected no match): {len(prov['deleted_origins']):,}")
if prov["params"]:
    print(f"  Applied: {prov['params']}")


# ============================================
# COVERAGE RULE -> CANDIDATE RADIUS
# ============================================

print(f"\n{'=' * 60}")
print(f"CANDIDATE RADIUS (coverage rule)")
print(f"{'=' * 60}")

# For Protocol A the displacement of a surviving node IS the distance between a
# true correspondence pair, so the gate can be sized exactly.
core_disp = np.array([
    d for n, d in prov["displacement"].items()
    if prov["origin"].get(n) is not None and prov["roles"].get(n) == "core"
])
if core_disp.size == 0:
    core_disp = np.zeros(1)

p99 = float(np.quantile(core_disp, COVERAGE))
radius = max(BASE_RADIUS, 5.0 * np.ceil(p99 / 5.0))
coverage_at_radius = float((core_disp <= radius).mean())

print(f"  Displacement of true pairs (core, surviving):")
print(f"    mean {core_disp.mean():.2f} m | median {np.median(core_disp):.2f} m | "
      f"p{COVERAGE*100:.0f} {p99:.2f} m | max {core_disp.max():.2f} m")
print(f"  Base radius: {BASE_RADIUS:.0f} m  ->  selected radius: {radius:.0f} m")
print(f"  True correspondences inside the gate: {100*coverage_at_radius:.2f}%")
if coverage_at_radius < COVERAGE:
    print(f"  WARNING: coverage below target - the gate, not the method, will "
          f"limit recall at this level.")


# ============================================
# WRITE VARIANT
# ============================================

print(f"\n{'=' * 60}")
print(f"WRITING VARIANT")
print(f"{'=' * 60}")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_dataset(graph, name):
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "Graph.pkl", "wb") as fh:
        pickle.dump(graph, fh)
    nx.write_edgelist(graph, d / f"{name}{SUFFIX}.edgelist", data=False)
    coords = {n: (a.get("x"), a.get("y")) for n, a in graph.nodes(data=True)}
    pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y"]).to_csv(
        d / f"{name}{SUFFIX}_coords.csv", index_label="node_id")
    print(f"  {name}: {graph.number_of_nodes():,} nodes -> {d}")


write_dataset(G, CLEAN_NAME)
write_dataset(H, PERT_NAME)

with open(OUT_DIR / "provenance.pkl", "wb") as f:
    pickle.dump(prov, f)
print(f"  provenance.pkl written")

# per-variant config: the sweep driver points SANA_CONFIG at this file
variant_config = dict(config)
variant_config["zone"] = VARIANT
variant_config["dataset_pairs"] = [[CLEAN_NAME, PERT_NAME]]
variant_config["spatial"] = dict(config["spatial"])
variant_config["spatial"]["radius"] = int(radius)
variant_config["provenance"] = str(OUT_DIR / "provenance.pkl")
with open(OUT_DIR / "config.yaml", "w") as f:
    yaml.safe_dump(variant_config, f, sort_keys=False)

manifest = {
    "variant": VARIANT,
    "protocol": "A_self_alignment",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "zone": ZONE,
    "source_dataset": SOURCE,
    "clean_dataset": CLEAN_NAME,
    "perturbed_dataset": PERT_NAME,
    "family": FAMILY,
    "level": LEVEL,
    "seed": SEED,
    "node_spacing_m": SPACING,
    "applied_params": prov["params"],
    "clean_nodes": G.number_of_nodes(),
    "clean_edges": G.number_of_edges(),
    "perturbed_nodes": H.number_of_nodes(),
    "perturbed_edges": H.number_of_edges(),
    "core_nodes": n_core,
    "surviving": int(sum(1 for o in prov["origin"].values() if o is not None)),
    "invented": len(prov["invented"]),
    "deleted_origins": len(prov["deleted_origins"]),
    "displacement_m": {
        "mean": float(core_disp.mean()),
        "median": float(np.median(core_disp)),
        f"p{int(COVERAGE*100)}": p99,
        "max": float(core_disp.max()),
    },
    "base_radius_m": BASE_RADIUS,
    "selected_radius_m": float(radius),
    "gt_coverage_at_radius": coverage_at_radius,
    "file_suffix": SUFFIX,
}
with open(OUT_DIR / "perturb_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)


# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Variant:  {VARIANT}")
print(f"  Pair:     {CLEAN_NAME} <-> {PERT_NAME}")
print(f"  Radius:   {radius:.0f} m  (coverage {100*coverage_at_radius:.2f}%)")
print(f"  Config:   {OUT_DIR / 'config.yaml'}")
print("")
print("  Run the pipeline on this variant with:")
print(f"    SANA_CONFIG={OUT_DIR / 'config.yaml'} python scripts/02_spatial_candidates.py")
print("    ... then 04, 05, 06, 08  (03 only for CENA)")
print("")
print("  Note: steps 02-09 do not read SANA_CONFIG yet - that override is the")
print("  next change. Until then, point config/default.yaml at this variant.")
print("=" * 60)
