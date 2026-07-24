import subprocess
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="DADP BERT/SST2 Sentiment Classification Sweep Runner")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per run")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5 for BERT)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./results/bert_experiments", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    parser.add_argument("--transformer_model", type=str, default="prajjwal1/bert-mini", help="Pre-trained HuggingFace transformer model to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define experiment configurations
    runs = []

    # 1. Baseline Run
    runs.append({
        "mode": "baseline",
        "params": []
    })

    # 2. DADP (Hebbian) Runs
    hebbian_thresholds = [1e-4, 5e-5, 2e-5, 1e-5, 5e-6]
    for thr in hebbian_thresholds:
        runs.append({
            "mode": "hebbian",
            "params": ["--prune_threshold", str(thr), "--prune_interval", "200"]
        })

    # 3. Standard baselines (SNIP, Magnitude, RigL) over sparsity range
    sparsities = [0.7, 0.8, 0.9, 0.95]
    for mode in ["snip", "magnitude", "rigl"]:
        for sp in sparsities:
            runs.append({
                "mode": mode,
                "params": ["--sparsity", str(sp)]
            })

    print(f"Total runs scheduled: {len(runs)}")

    # Execute runs
    for i, run in enumerate(runs):
        mode = run["mode"]
        run_params = run["params"]
        
        # Build command
        cmd = [
            sys.executable, "scripts/main.py",
            "--arch", "bert",
            "--dataset", "SST2",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--mode", mode,
            "--transformer_model", args.transformer_model
        ] + run_params

        # Print visual status
        cmd_str = " ".join(cmd)
        print(f"\n==================================================")
        print(f"Executing Run {i+1}/{len(runs)}: {mode.upper()}")
        print(f"Command: {cmd_str}")
        print(f"==================================================")

        if args.dry_run:
            continue

        try:
            # Run the training process
            subprocess.run(cmd, check=True)
            print(f"Run {i+1} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing run {i+1}: {e}")
            print("Continuing with next runs...")

if __name__ == "__main__":
    main()
