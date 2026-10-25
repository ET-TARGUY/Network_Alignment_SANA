"""
Main pipeline: runs all steps in order, stops if any step fails.
"""
import subprocess
import sys
import os
from pathlib import Path
import yaml


SCRIPTS_DIR = Path(__file__).parent
# Load config
config_path = Path(os.environ.get("SANA_CONFIG", "config/default.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get parameters
method = config['compound_graph']['method']

# A variant config (written by 01c_perturb.py) names a provenance file. Its
# datasets already exist on disk, so steps 01/01b/01c must NOT re-run: step 01
# would rebuild the FULL network from raw GeoJSON and overwrite it.
IS_VARIANT = bool(config.get("provenance"))

if method == "SANA":
    STEPS = [
        ("01_prepare_data.py",  "Prepare Data"),
        ("02_spatial_candidates.py",  "Spatial Candidates"),
        ("04_build_compound_graph.py",  "Build Compound Graph"),
        ("05_generate_random_walks.py", "Generate Random Walks"),
        ("06_generate_embeddings.py",   "Generate Embeddings"),
        ("07_embedding_analysis.py",   "Embedding Analysis"),
        ("08_hungarian_alignment.py",   "Hungarian Alignment"),
        ("09_evaluate_alignment.py",    "Evaluate Alignment"),
    ]

if method == "CENA":
    STEPS = [
        ("01_prepare_data.py",  "Prepare Data"),
        ("02_spatial_candidates.py",  "Spatial Candidates"),
        ("03_structural_similarity.py",  "Structural Similarity"),
        ("04_build_compound_graph.py",  "Build Compound Graph"),
        ("05_generate_random_walks.py", "Generate Random Walks"),
        ("06_generate_embeddings.py",   "Generate Embeddings"),
        ("07_embedding_analysis.py",   "Embedding Analysis"),
        ("08_hungarian_alignment.py",   "Hungarian Alignment"),
        ("09_evaluate_alignment.py",    "Evaluate Alignment"),
    ]


def run_step(script: str, name: str) -> None:
    script_path = SCRIPTS_DIR / script
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}  ({script})")
    print("="*60)

    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"\n[FAILED] {name} exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

    print(f"[DONE] {name}")


if __name__ == "__main__":
    if IS_VARIANT:
        STEPS = [s for s in STEPS if not s[0].startswith("01")]
        print(f"Variant config detected ({config['provenance']}) - skipping data preparation.")
    print("Starting pipeline...")
    for script, name in STEPS:
        run_step(script, name)
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*60)