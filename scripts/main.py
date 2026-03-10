"""
Main pipeline: runs all steps in order, stops if any step fails.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
     ("02_spatial_candidates.py",  "spatial candidates"),
    ("04_build_compound_graph.py",  "Build Compound Graph"),
    ("05_generate_random_walks.py", "Generate Random Walks"),
    ("06_generate_embeddings.py",   "Generate Embeddings"),
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
    print("Starting pipeline...")
    for script, name in STEPS:
        run_step(script, name)
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*60)