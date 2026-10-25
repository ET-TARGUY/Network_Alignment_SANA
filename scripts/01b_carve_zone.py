"""
Step 1b: Carve a derived zone out of the full processed network.

Reads the full graphs written by 01_prepare_data.py and produces a
self-contained zone directory that the rest of the pipeline (02..09 and the
baselines) runs on unchanged, simply by setting `zone: <ZONE_ID>` in the config.

Design decisions (see also the `zones:` block in config/default.yaml):

  * CORE  - the same metric box is applied to BOTH networks. They are never cut
            independently: that would manufacture boundary mismatch which is not
            present in the data and would flatter or punish methods arbitrarily.

  * HALO  - defined in HOPS, per graph, on that graph's own topology, so each
            network keeps as much context as its own structure requires. The hop
            count must cover the receptive field of the algorithms downstream
            (CENA reads structural.K hops; skip-gram reads embeddings.window
            steps of walk context), otherwise core nodes get truncated
            neighbourhoods and their signatures are corrupted by the cut.

  * ROLES - every node is tagged role='core' or role='halo'. Evaluation must be
            restricted to core nodes. Halo nodes stay in the graph as context and
            remain eligible as candidates, so a core->halo prediction is simply
            wrong and is counted as an error rather than hidden.

  * IDs   - parent node identifiers are preserved, so ground truth and any later
            perturbation provenance chain back to the full network.

Outputs, for each dataset D:
    <data_processed>/<ZONE>/D/Graph.pkl
    <data_processed>/<ZONE>/D/D_<zone>.edgelist
    <data_processed>/<ZONE>/D/D_<zone>_coords.csv
    <data_processed>/<ZONE>/zone_manifest.json
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ============================================
# LOAD CONFIGURATION
# ============================================

print("=" * 60)
print("STEP 1b: CARVE ZONE")
print("=" * 60)

config_path = Path(os.environ.get("SANA_CONFIG", "config/default.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

ZONE = config.get("zone", "full")
ZONES = config.get("zones") or {}

if ZONE not in ZONES:
    print(f"\n  Zone '{ZONE}' is not a derived zone (nothing under `zones:` in {config_path}).")
    print(f"  Derived zones available: {sorted(ZONES) or 'none'}")
    print("  Nothing to carve - exiting.")
    sys.exit(0)

spec = ZONES[ZONE]
XMIN, XMAX = float(spec["xmin"]), float(spec["xmax"])
YMIN, YMAX = float(spec["ymin"]), float(spec["ymax"])
HALO_HOPS = int(spec.get("halo_hops", 8))

PROCESSED_DATA_DIR = Path(config["paths"]["data_processed"])
OUT_DIR = PROCESSED_DATA_DIR / ZONE
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUFFIX = f"_{ZONE.lower()}"

DATASETS = list(dict.fromkeys(ds for pair in config["dataset_pairs"] for ds in pair))

# Sanity: the halo must cover the receptive field of everything downstream.
required_hops = max(
    int(config.get("structural", {}).get("K", 0)),
    int(config.get("embeddings", {}).get("window", 0)),
)

print(f"\nConfiguration:")
print(f"  Config file: {config_path}")
print(f"  Zone: {ZONE}  ({spec.get('description', '')})")
print(f"  Box: x [{XMIN:.1f}, {XMAX:.1f})  y [{YMIN:.1f}, {YMAX:.1f})")
print(f"  Box size: {(XMAX - XMIN) / 1000:.2f} x {(YMAX - YMIN) / 1000:.2f} km")
print(f"  Halo: {HALO_HOPS} hops  (receptive field downstream needs >= {required_hops})")
print(f"  Datasets: {DATASETS}")
print(f"  Output: {OUT_DIR}")

if HALO_HOPS < required_hops:
    print(f"\n  WARNING: halo_hops={HALO_HOPS} is smaller than the downstream "
          f"receptive field ({required_hops}). Core-node neighbourhoods will be "
          f"truncated by the cut.")


# ============================================
# HELPERS
# ============================================

def select_core(G, xmin, xmax, ymin, ymax):
    """Nodes whose coordinates fall inside the half-open box [xmin,xmax) x [ymin,ymax)."""
    core = set()
    missing_coords = 0
    for node, data in G.nodes(data=True):
        x, y = data.get("x"), data.get("y")
        if x is None or y is None:
            missing_coords += 1
            continue
        if xmin <= x < xmax and ymin <= y < ymax:
            core.add(node)
    return core, missing_coords


def expand_halo(G, core, hops):
    """Nodes within `hops` hops of the core, excluding the core itself (BFS by level)."""
    halo = set()
    seen = set(core)
    frontier = set(core)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        nxt -= seen
        if not nxt:
            break
        halo |= nxt
        seen |= nxt
        frontier = nxt
    return halo


# ============================================
# CARVE EACH DATASET
# ============================================

manifest = {
    "zone": ZONE,
    "description": spec.get("description", ""),
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "config_file": str(config_path),
    "box": {"xmin": XMIN, "xmax": XMAX, "ymin": YMIN, "ymax": YMAX},
    "box_size_km": [(XMAX - XMIN) / 1000.0, (YMAX - YMIN) / 1000.0],
    "halo_hops": HALO_HOPS,
    "required_receptive_field_hops": required_hops,
    "file_suffix": SUFFIX,
    "datasets": {},
}

for i, dataset_name in enumerate(DATASETS, 1):
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(DATASETS)}] {dataset_name}")
    print(f"{'=' * 60}")

    graph_path = PROCESSED_DATA_DIR / dataset_name / "Graph.pkl"
    if not graph_path.exists():
        print(f"  ERROR: full graph not found: {graph_path}")
        print(f"  Run scripts/01_prepare_data.py first (with zone 'full').")
        sys.exit(1)

    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  Full network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # --- core ---
    core, missing_coords = select_core(G, XMIN, XMAX, YMIN, YMAX)
    if missing_coords:
        print(f"  WARNING: {missing_coords:,} nodes have no coordinates and were skipped")
    if not core:
        print(f"  ERROR: no nodes fall inside the box - check the zone coordinates.")
        sys.exit(1)
    print(f"  Core nodes (inside box): {len(core):,}")

    # --- halo ---
    halo = expand_halo(G, core, HALO_HOPS)
    keep = core | halo
    print(f"  Halo nodes ({HALO_HOPS} hops):    {len(halo):,}")
    print(f"  Kept total:               {len(keep):,}")

    # --- induced subgraph ---
    Gz = G.subgraph(keep).copy()
    nx.set_node_attributes(
        Gz, {n: ("core" if n in core else "halo") for n in Gz.nodes()}, "role"
    )

    # --- boundary diagnostics ---
    severed = sum(1 for u, v in G.edges() if (u in keep) != (v in keep))
    core_degree_changed = sum(1 for n in core if G.degree(n) != Gz.degree(n))
    halo_degree_changed = sum(1 for n in halo if G.degree(n) != Gz.degree(n))

    print(f"\n  Zone graph: {Gz.number_of_nodes():,} nodes, {Gz.number_of_edges():,} edges")
    print(f"  Boundary diagnostics:")
    print(f"    Severed edges (one endpoint outside): {severed:,}")
    print(f"    Core nodes with changed degree:       {core_degree_changed:,}  (expect 0)")
    print(f"    Halo nodes with changed degree:       {halo_degree_changed:,}  (expected, they are the ring)")
    if core_degree_changed:
        print(f"    WARNING: core degrees changed - the halo is not covering the core boundary.")

    components = nx.number_connected_components(Gz)
    largest = max((len(c) for c in nx.connected_components(Gz)), default=0)
    print(f"    Connected components: {components:,}  (largest {largest:,} nodes)")

    # --- write outputs ---
    out_dataset_dir = OUT_DIR / dataset_name
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    graph_out = out_dataset_dir / "Graph.pkl"
    with open(graph_out, "wb") as f:
        pickle.dump(Gz, f)
    print(f"\n  Saved graph:    {graph_out}")

    edgelist_out = out_dataset_dir / f"{dataset_name}{SUFFIX}.edgelist"
    nx.write_edgelist(Gz, edgelist_out, data=False)
    print(f"  Saved edgelist: {edgelist_out}")

    coords_out = out_dataset_dir / f"{dataset_name}{SUFFIX}_coords.csv"
    coords = {n: (d.get("x"), d.get("y")) for n, d in Gz.nodes(data=True)}
    pd.DataFrame.from_dict(coords, orient="index", columns=["x", "y"]).to_csv(
        coords_out, index_label="node_id"
    )
    print(f"  Saved coords:   {coords_out}")

    manifest["datasets"][dataset_name] = {
        "source_graph": str(graph_path),
        "full_nodes": G.number_of_nodes(),
        "full_edges": G.number_of_edges(),
        "core_nodes": len(core),
        "halo_nodes": len(halo),
        "zone_nodes": Gz.number_of_nodes(),
        "zone_edges": Gz.number_of_edges(),
        "nodes_missing_coords": missing_coords,
        "severed_edges": severed,
        "core_degree_changed": core_degree_changed,
        "halo_degree_changed": halo_degree_changed,
        "connected_components": components,
        "largest_component": largest,
    }


# ============================================
# MANIFEST + SUMMARY
# ============================================

manifest_path = OUT_DIR / "zone_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nZone '{ZONE}' carved into: {OUT_DIR}")
for name, info in manifest["datasets"].items():
    print(f"  {name}: {info['core_nodes']:,} core + {info['halo_nodes']:,} halo "
          f"= {info['zone_nodes']:,} nodes, {info['zone_edges']:,} edges")
print(f"\nManifest: {manifest_path}")
print(f"\nThe rest of the pipeline now runs on this zone with  zone: \"{ZONE}\"  in the config.")
print("Evaluation must be restricted to nodes with role == 'core'.")
print("\n" + "=" * 60)
print("ZONE CARVING COMPLETE")
print("=" * 60)
