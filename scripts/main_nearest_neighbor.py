"""
Baseline: Nearest Neighbor Alignment

For each node u in G1, assign:
    π(u) = argmin_{v ∈ G2} d(u, v)

No embeddings, no random walks — pure Euclidean distance.
One-to-one matching enforced via Hungarian algorithm on the distance matrix
(with spatial candidates pre-filtered by radius to keep the matrix tractable).

Outputs:
  - data/processed/<dataset1>/alignment_hungarian_NearestNeighbor_<dataset2>.pkl
  - outputs/alignment_evaluation_NearestNeighbor_<dataset1>_<dataset2>.png
"""

import sys
import os
import pickle
import yaml
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')   # file-only backend: no GUI window, no Dock icon, and safe in parallel runs
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.matching.assignment import block_assignment

METHOD = "NearestNeighbor"

# Ground-truth mode must match step 09 exactly, otherwise this baseline is
# scored by a different rule than the method it is being compared against.
def _provenance_path(cfg):
    pth = cfg.get("provenance")
    return pth if pth and Path(pth).exists() else None

# ============================================
# LOAD CONFIGURATION
# ============================================

print("=" * 60)
print("BASELINE: NEAREST NEIGHBOR ALIGNMENT")
print("=" * 60)

config_path = Path(os.environ.get("SANA_CONFIG", "config/default.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

ANCHOR_TOLERANCE = float(config.get('evaluation', {}).get('gt_tolerance_m', 0.1))  # same as step 9

dataset_pairs   = config['dataset_pairs']
zone            = config['zone']
radius          = config['spatial']['radius']

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUTS_DIR        = Path(config['paths']['outputs'])
# GeoPackage export is a visualisation artifact, not a result. It is skipped
# during sweeps: writing one per run costs disk and time, and re-running a
# variant fails outright because pyogrio refuses to recreate an existing layer.
EXPORT_GPKG = bool(config.get("export_gpkg", True))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

base_dir = PROCESSED_DATA_DIR if str(zone).lower() == "full" else PROCESSED_DATA_DIR / zone

print(f"  Zone:   {zone}")
print(f"  Radius: {radius}m  (spatial pre-filter)")
print(f"  Method: {METHOD}")


# ============================================
# PROCESS EACH DATASET PAIR
# ============================================

for pair_idx, (dataset1_name, dataset2_name) in enumerate(dataset_pairs, 1):
    print(f"\n{'=' * 60}")
    print(f"PAIR {pair_idx}/{len(dataset_pairs)}: {dataset1_name} ↔ {dataset2_name}")
    print(f"{'=' * 60}")

    # ------ Load graphs ------
    graph1_path = base_dir / dataset1_name / "Graph.pkl"
    graph2_path = base_dir / dataset2_name / "Graph.pkl"

    if not graph1_path.exists() or not graph2_path.exists():
        print(f"  ❌ Graphs not found — skipping pair")
        continue

    with open(graph1_path, 'rb') as f:
        G1 = pickle.load(f)
    with open(graph2_path, 'rb') as f:
        G2 = pickle.load(f)

    print(f"  ✓ {dataset1_name}: {G1.number_of_nodes():,} nodes")
    print(f"  ✓ {dataset2_name}: {G2.number_of_nodes():,} nodes")

    # ------ Build coordinate arrays ------
    nodes_G1 = list(G1.nodes())
    nodes_G2 = list(G2.nodes())

    coords_G1 = np.array([[G1.nodes[n]['x'], G1.nodes[n]['y']] for n in nodes_G1])
    coords_G2 = np.array([[G2.nodes[n]['x'], G2.nodes[n]['y']] for n in nodes_G2])

    tree_G2 = KDTree(coords_G2)

    # ------ Spatial pre-filter (candidates within radius) ------
    print(f"\nFinding candidates within {radius}m ...")
    candidates = tree_G2.query_ball_point(coords_G1, r=radius)

    nodes_with_candidates = [i for i, c in enumerate(candidates) if len(c) > 0]
    print(f"  G1 nodes with ≥1 candidate: {len(nodes_with_candidates):,} / {len(nodes_G1):,}")

    # ------ Assignment over the radius-limited candidate set ------
    # The candidate graph decomposes into small components, so the exact optimum
    # is obtained without ever building a dense 42,000 x 42,000 matrix.
    print(f"\nSolving assignment ...")
    rows, cols, costs = [], [], []
    for i, cand in enumerate(candidates):
        for j in cand:
            rows.append(i)
            cols.append(j)
            costs.append(float(np.linalg.norm(coords_G1[i] - coords_G2[j])))
    print(f"  candidate pairs: {len(rows):,}")

    matches = []
    for i, j in block_assignment(rows, cols, costs, verbose=True):
        matches.append({
            'g1_id':      nodes_G1[i],
            'g2_id':      nodes_G2[j],
            'x_g1':       coords_G1[i][0],
            'y_g1':       coords_G1[i][1],
            'x_g2':       coords_G2[j][0],
            'y_g2':       coords_G2[j][1],
            'distance_m': float(np.linalg.norm(coords_G1[i] - coords_G2[j])),
        })

    print(f"  ✓ {len(matches):,} matches produced")

    # ------ Save alignment pickle ------
    output_dir = base_dir / dataset1_name
    alignment_path = output_dir / f"alignment_hungarian_{METHOD}_{dataset2_name}.pkl"

    with open(alignment_path, 'wb') as f:
        pickle.dump({
            'matches':    matches,
            'method':     METHOD,
            'parameters': {'radius': radius},
            'dataset1':   dataset1_name,
            'dataset2':   dataset2_name,
        }, f)

    print(f"  ✓ Saved: {alignment_path}")

    # ------ Export GeoPackage ------
    features = []
    for m in matches:
        features.append({
            'g1_id':      m['g1_id'],
            'g2_id':      m['g2_id'],
            'distance_m': m['distance_m'],
            'geometry':   LineString([(m['x_g1'], m['y_g1']), (m['x_g2'], m['y_g2'])]),
        })

    gdf = gpd.GeoDataFrame(features, crs='EPSG:2154')
    gpkg_path = OUTPUTS_DIR / f"alignment_hungarian_{METHOD}_{dataset1_name}_{dataset2_name}.gpkg"
    if EXPORT_GPKG:
        if gpkg_path.exists():
            gpkg_path.unlink()
        gdf.to_file(gpkg_path, layer='matches', driver='GPKG')
    else:
        print('  Skipping GeoPackage export (export_gpkg = false)')
    print(f"  ✓ GeoPackage: {gpkg_path}")

    # ============================================
    # EVALUATION  (same logic as step 9)
    # ============================================

    print(f"\n{'=' * 60}")
    print(f"EVALUATION")
    print(f"{'=' * 60}")

    # Ground truth anchors
    coords_G1_dict = {n: (G1.nodes[n]['x'], G1.nodes[n]['y']) for n in G1.nodes()}
    coords_G2_dict = {n: (G2.nodes[n]['x'], G2.nodes[n]['y']) for n in G2.nodes()}

    nodes_G2_list  = list(coords_G2_dict.keys())
    coords_G2_arr  = np.array([coords_G2_dict[n] for n in nodes_G2_list])
    tree_gt        = KDTree(coords_G2_arr)

    # ------------------------------------------------------------------
    # Restrict evaluation to CORE nodes when the graph was carved into a zone.
    # Halo nodes exist only to give core nodes an intact neighbourhood across the
    # cut: their own neighbourhoods ARE truncated, so their signatures and
    # embeddings are unreliable and they must never be scored. They stay eligible
    # as match candidates, so a core -> halo prediction still counts as an error.
    # Graphs with no 'role' attribute (full network, Prades) are unaffected.
    # ------------------------------------------------------------------
    roles_G1 = {n: d.get("role") for n, d in G1.nodes(data=True)}
    if any(r is not None for r in roles_G1.values()):
        eval_nodes_G1 = {n for n, r in roles_G1.items() if r == "core"}
        print(f"  Zone graph detected: evaluating on {len(eval_nodes_G1):,} core nodes "
              f"({G1.number_of_nodes() - len(eval_nodes_G1):,} halo nodes excluded)")
    else:
        eval_nodes_G1 = set(G1.nodes())
    eval_coords_G1 = {n: c for n, c in coords_G1_dict.items() if n in eval_nodes_G1}

    ground_truth = {}
    for node_g1, coord in tqdm(eval_coords_G1.items(), desc="Building ground truth"):
        idxs = tree_gt.query_ball_point(coord, ANCHOR_TOLERANCE)
        if len(idxs) == 1:
            ground_truth[node_g1] = nodes_G2_list[idxs[0]]

    # Provenance ground truth (Protocol A) replaces the 0.1 m co-location rule.
    expected_no_match = set()
    false_alarms = 0
    correct_rejections = 0
    PROVENANCE_PATH = _provenance_path(config)
    if PROVENANCE_PATH:
        with open(PROVENANCE_PATH, "rb") as f:
            prov = pickle.load(f)
        inverse = prov["inverse"]
        ground_truth = {}
        for node_g1 in eval_nodes_G1:
            new_id = inverse.get(node_g1)
            if new_id is None:
                expected_no_match.add(node_g1)
            else:
                ground_truth[node_g1] = new_id
        print(f"  PROVENANCE ground truth: {prov.get('family')} "
              f"level {prov.get('level')} seed {prov.get('seed')}")
        print(f"  Expected no match: {len(expected_no_match):,}")

    print(f"  Ground truth anchors: {len(ground_truth):,}")

    predicted_dict = {m['g1_id']: m['g2_id'] for m in matches}

    TP = FP = FN = 0
    correct_matches   = []
    incorrect_matches = []
    missed_anchors    = []

    for node_g1, gt_g2 in ground_truth.items():
        if node_g1 in predicted_dict:
            if predicted_dict[node_g1] == gt_g2:
                TP += 1
                correct_matches.append(node_g1)
            else:
                FP += 1
                err_dist = np.linalg.norm(
                    np.array(coords_G1_dict[node_g1]) -
                    np.array(coords_G2_dict[predicted_dict[node_g1]])
                )
                incorrect_matches.append({'g1_id': node_g1, 'error_distance': err_dist})
        else:
            FN += 1
            missed_anchors.append(node_g1)

    false_alarms = sum(1 for g1 in expected_no_match if g1 in predicted_dict)
    correct_rejections = len(expected_no_match) - false_alarms
    FP += false_alarms

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  TP: {TP:,}  FP: {FP:,}  FN: {FN:,}")
    print(f"  Precision : {precision:.4f}  ({100*precision:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({100*recall:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({100*f1:.2f}%)")

    # ------ Visualization ------
    if config.get('export_figures', True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{METHOD} — {dataset1_name} ↔ {dataset2_name}", fontsize=13)

        metrics = ['Precision', 'Recall', 'F1-Score']
        values  = [precision, recall, f1]
        colors  = ['green' if v > 0.9 else 'orange' if v > 0.7 else 'red' for v in values]
        axes[0].bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
        axes[0].set_ylim([0, 1])
        axes[0].set_title('Alignment Quality Metrics')
        axes[0].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='>0.9')
        axes[0].axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='>0.7')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')

        axes[1].bar(['TP', 'FP', 'FN'], [TP, FP, FN],
                    color=['green', 'red', 'orange'], edgecolor='black', alpha=0.7)
        axes[1].set_title('Confusion Matrix')
        axes[1].grid(alpha=0.3, axis='y')

        if incorrect_matches:
            err_dists = [m['error_distance'] for m in incorrect_matches]
            axes[2].hist(err_dists, bins=30, edgecolor='black', alpha=0.7, color='red')
            axes[2].axvline(np.median(err_dists), color='blue', linestyle='--',
                            label=f'Median={np.median(err_dists):.2f}m')
            axes[2].set_xlabel('Error Distance (m)')
            axes[2].set_title('Incorrect Match Distances')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No incorrect matches!',
                         ha='center', va='center', fontsize=18, color='green')
            axes[2].axis('off')

        plt.tight_layout()
        viz_path = OUTPUTS_DIR / f"alignment_evaluation_{METHOD}_{dataset1_name}_{dataset2_name}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot: {viz_path}")

    # ------ Save evaluation pickle ------
    eval_path = output_dir / f"alignment_evaluation_{METHOD}_{dataset2_name}.pkl"
    with open(eval_path, 'wb') as f:
        pickle.dump({
            'ground_truth_mode':    'provenance' if PROVENANCE_PATH else 'spatial',
            'ground_truth_anchors': len(ground_truth),
            'expected_no_match':    len(expected_no_match),
            'false_alarms':         false_alarms,
            'correct_rejections':   correct_rejections,
            'predicted_matches':    len(matches),
            'true_positives':       TP,
            'false_positives':      FP,
            'false_negatives':      FN,
            'precision':            precision,
            'recall':               recall,
            'f1_score':             f1,
            'correct_matches':      correct_matches,
            'incorrect_matches':    incorrect_matches,
            'missed_anchors':       missed_anchors,
            'ground_truth_dict':    ground_truth,
        }, f)
    print(f"  ✓ Evaluation saved: {eval_path}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✓ NEAREST NEIGHBOR BASELINE COMPLETE")
print("=" * 60)