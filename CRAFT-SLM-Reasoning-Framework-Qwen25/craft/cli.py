#!/usr/bin/env python3
"""CRAFT CLI - Command line interface for the CRAFT framework."""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from craft.utils.logger import setup_logger
from craft.utils.hardware_detector import detect_hardware
from craft.config import load_config

logger = setup_logger()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m craft",
        description="CRAFT - Curriculum-guided Reinforced Adaptive Fine-Tuning for SLMs",
        epilog="Use 'python -m craft <command> --help' for more information."
    )
    
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show framework information")
    
    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Validate installation and hardware")
    
    # probe command
    probe_parser = subparsers.add_parser("probe", help="Phase 0: Run difficulty probing")
    probe_parser.add_argument("--dataset", "-d", type=str, help="Path to dataset or HF dataset name")
    probe_parser.add_argument("--model", "-m", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Model name/path")
    probe_parser.add_argument("--output", "-o", type=str, default="Outputs/difficulty_map.json", help="Output path")
    probe_parser.add_argument("--samples", "-n", type=int, default=5, help="Number of samples per question")
    probe_parser.add_argument("--dry-run", action="store_true", help="Run without actual model")
    probe_parser.add_argument("--config", "-c", type=str, default="phi3_mini", help="Config name")
    probe_parser.add_argument("--hardware", type=str, default="kaggle", choices=["kaggle", "local_2x24gb"])
    
    # sft command
    sft_parser = subparsers.add_parser("sft", help="Phase 1: Run Supervised Fine-Tuning")
    sft_parser.add_argument("--output", "-o", type=str, default="checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    sft_parser.add_argument("--epochs", type=int, help="Number of epochs")
    sft_parser.add_argument("--batch-size", type=int, help="Per device batch size")
    sft_parser.add_argument("--config", "-c", type=str, default="phi3_mini", help="Config name")
    sft_parser.add_argument("--hardware", type=str, default="kaggle", choices=["kaggle", "local_2x24gb"])
    
    # rl command
    rl_parser = subparsers.add_parser("rl", help="Phase 2: Run Reinforcement Learning")
    rl_parser.add_argument("--checkpoint", "-ckpt", type=str, default="checkpoints/sft/final", help="SFT checkpoint path")
    rl_parser.add_argument("--output", "-o", type=str, default="checkpoints/rl", help="Output directory")
    rl_parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    rl_parser.add_argument("--steps", "-s", type=int, default=500, help="Total RL steps")
    rl_parser.add_argument("--config", "-c", type=str, default="phi3_mini", help="Config name")
    rl_parser.add_argument("--hardware", type=str, default="kaggle", choices=["kaggle", "local_2x24gb"])
    
    # diagnostics command
    diag_parser = subparsers.add_parser("diagnostics", help="Run diagnostics")
    diag_parser.add_argument("--checkpoint", "-ckpt", type=str, help="Checkpoint path")
    diag_parser.add_argument("--output", "-o", type=str, default="diagnostics.json", help="Output file")
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", "-ckpt", type=str, help="Checkpoint path")
    eval_parser.add_argument("--output", "-o", type=str, default="evaluation.json", help="Output file")
    eval_parser.add_argument("--datasets", nargs="+", default=["gsm8k", "strategyqa", "mmlu"], help="Datasets to evaluate")
    eval_parser.add_argument("--samples", type=int, default=100, help="Samples per dataset")
    eval_parser.add_argument("--baseline", action="store_true", help="Evaluate baseline model")
    eval_parser.add_argument("--sft", action="store_true", help="Evaluate SFT model")
    eval_parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints and find best")
    
    # quantize command
    quant_parser = subparsers.add_parser("quantize", help="Quantize model to GGUF")
    quant_parser.add_argument("--checkpoint", "-ckpt", type=str, default="checkpoints/rl/final", help="Checkpoint path")
    quant_parser.add_argument("--output", "-o", type=str, default="craft_output/CRAFT_Q4_K_M.gguf", help="Output path")
    quant_parser.add_argument("--bits", type=str, default="Q4_K_M", help="Quantization type")
    quant_parser.add_argument("--base-model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Base model name")
    
    # package command
    package_parser = subparsers.add_parser("package", help="Package model for deployment")
    package_parser.add_argument("--checkpoint", "-ckpt", type=str, default="craft_output/CRAFT_Q4_K_M.gguf", help="Model path")
    package_parser.add_argument("--output", "-o", type=str, default="craft_delivery", help="Output directory")
    
    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model")
    deploy_parser.add_argument("--checkpoint", "-ckpt", type=str, default="craft_delivery", help="Package path")
    deploy_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # report command
    report_parser = subparsers.add_parser("report", help="Generate benchmark report")
    report_parser.add_argument("--baseline-json", type=str, required=True, help="Baseline results JSON")
    report_parser.add_argument("--champion-json", type=str, required=True, help="Champion results JSON")
    report_parser.add_argument("--output", "-o", type=str, default="craft_output/benchmark_report.md", help="Output file")
    
    # run-all command
    runall_parser = subparsers.add_parser("run-all", help="Run complete pipeline")
    runall_parser.add_argument("--config", "-c", type=str, default="phi3_mini", help="Config name")
    runall_parser.add_argument("--output", "-o", type=str, default="checkpoints", help="Base output directory")
    runall_parser.add_argument("--resume", action="store_true", help="Resume from where left off")
    runall_parser.add_argument("--hardware", type=str, default="kaggle", choices=["kaggle", "local_2x24gb"])
    
    # checkpoints command
    ckpt_parser = subparsers.add_parser("checkpoints", help="List checkpoints")
    ckpt_parser.add_argument("--dir", "-d", type=str, default="checkpoints", help="Checkpoint directory")
    
    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean temporary files")
    clean_parser.add_argument("--all", action="store_true", help="Remove all checkpoints")
    
    # config command
    config_parser = subparsers.add_parser("config", help="Print current configuration")
    config_parser.add_argument("--config", "-c", type=str, default="phi3_mini", help="Config name")
    config_parser.add_argument("--hardware", type=str, default="kaggle", choices=["kaggle", "local_2x24gb"])
    
    args = parser.parse_args()
    
    if args.version:
        from craft import __version__
        print(f"CRAFT Framework v{__version__}")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate handler
    if args.command == "info":
        return info_command()
    elif args.command == "doctor":
        return doctor_command()
    elif args.command == "probe":
        return probe_command(args)
    elif args.command == "sft":
        return sft_command(args)
    elif args.command == "rl":
        return rl_command(args)
    elif args.command == "diagnostics":
        return diagnostics_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    elif args.command == "quantize":
        return quantize_command(args)
    elif args.command == "package":
        return package_command(args)
    elif args.command == "deploy":
        return deploy_command(args)
    elif args.command == "report":
        return report_command(args)
    elif args.command == "run-all":
        return run_all_command(args)
    elif args.command == "checkpoints":
        return checkpoints_command(args)
    elif args.command == "clean":
        return clean_command(args)
    elif args.command == "config":
        return config_command(args)
    
    return 0


def info_command() -> int:
    """Show framework information."""
    from craft import __version__
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    CRAFT Framework v{__version__}                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Curriculum-guided Reinforced Adaptive Fine-Tuning for SLMs      ║
╠══════════════════════════════════════════════════════════════════╣
║  Components:                                                     ║
║    Component A - Execution Verifier (Math + Logical NLI)         ║
║    Component B - Step-level Contrastive DPO                      ║
║    Component C - Adaptive Curriculum Engine                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Datasets: GSM8K, StrategyQA, MMLU, AQuA-RAT                     ║
║  Models: Phi-3-Mini (3.8B), Qwen2.5-7B                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands:                                                       ║
║    probe      - Run difficulty probing                           ║
║    sft        - Run supervised fine-tuning                       ║
║    rl         - Run reinforcement learning                       ║
║    evaluate   - Evaluate model on benchmarks                     ║
║    quantize   - Quantize to GGUF                                 ║
║    package    - Package for deployment                           ║
║    run-all    - Run complete pipeline                            ║
╚══════════════════════════════════════════════════════════════════╝
""")
    return 0


def doctor_command() -> int:
    """Validate installation and hardware."""
    print("\n🔍 CRAFT Diagnostics\n" + "=" * 50)
    
    # Check Python version
    import sys
    py_version = sys.version_info
    print(f"Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print("  ⚠️ Python 3.9+ recommended")
    else:
        print("  ✅ OK")
    
    # Check torch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️ CUDA not available - training will be slow")
    except ImportError:
        print("  ❌ PyTorch not installed")
    
    # Check hardware
    hw = detect_hardware()
    print(f"\nHardware Profile:")
    print(f"  GPU: {hw.gpu_count}x {hw.gpu_name} ({hw.gpu_vram_gb} GB)")
    print(f"  CPU Cores: {hw.cpu_cores}")
    print(f"  RAM: {hw.ram_gb} GB")
    print(f"  Environment: {'Kaggle' if hw.is_kaggle else 'Local'}")
    
    # Check required directories
    import os
    dirs = ["Outputs", "checkpoints", "logs", "craft_output"]
    print("\nDirectories:")
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✅ {d}/")
        else:
            os.makedirs(d, exist_ok=True)
            print(f"  ✅ {d}/ (created)")
    
    # Check datasets (optional)
    print("\nDatasets (will be downloaded when needed):")
    datasets = ["gsm8k", "aqua_rat", "ChilleD/StrategyQA", "cais/mmlu"]
    for ds in datasets:
        print(f"  📦 {ds}")
    
    print("\n" + "=" * 50)
    print("✅ Diagnostics complete.\n")
    return 0


def probe_command(args) -> int:
    """Run difficulty probing."""
    from craft.phase0_probe.probe import CapabilityProbe
    from craft.config import load_config
    
    logger.info("Starting Phase 0: Capability Probe")
    
    config = load_config(args.config, args.hardware)
    
    # Override model if specified
    if args.model and args.model != "microsoft/Phi-3-mini-4k-instruct":
        if "model" not in config:
            config["model"] = {}
        config["model"]["name"] = args.model
    
    probe = CapabilityProbe(config)
    probe.run(output_path=args.output, dry_run=args.dry_run)
    
    logger.info(f"Probe completed. Results saved to {args.output}")
    return 0


def sft_command(args) -> int:
    """Run SFT training."""
    from craft.phase1_sft.train_sft import train_sft
    
    logger.info("Starting Phase 1: SFT Warmup")
    
    train_sft(
        config_name=args.config,
        hardware_name=args.hardware,
        output_dir=args.output,
        resume=args.resume
    )
    
    return 0


def rl_command(args) -> int:
    """Run RL training."""
    from craft.phase2_rl.craft_rl_loop import train_rl
    
    logger.info("Starting Phase 2: RL Training")
    
    # Override steps if specified
    if args.steps != 500:
        # Will be handled by train_rl
        pass
    
    train_rl(
        config_name=args.config,
        hardware=args.hardware,
        output_dir=args.output,
        resume=args.resume
    )
    
    return 0


def diagnostics_command(args) -> int:
    """Run diagnostics."""
    from craft.phase2_rl.run_diagnostics import run_diagnostics
    
    logger.info("Running diagnostics...")
    run_diagnostics()
    return 0


def evaluate_command(args) -> int:
    """Evaluate model on benchmarks."""
    from craft.phase3_deploy.evaluator import BenchmarkEvaluator
    import json
    
    logger.info("Starting evaluation...")
    evaluator = BenchmarkEvaluator(use_gpu=True)
    
    # Determine what to evaluate
    if args.all:
        # Evaluate all checkpoints and find best
        from craft.utils.checkpoint_manager import CheckpointManager
        import glob
        
        results = {}
        best_acc = 0
        best_checkpoint = None
        
        # Find all RL checkpoints
        rl_checkpoints = sorted(glob.glob("checkpoints/rl/checkpoint-*"))
        
        for cp in rl_checkpoints:
            label = os.path.basename(cp)
            logger.info(f"Evaluating {label}...")
            
            for dataset in args.datasets:
                result = evaluator.evaluate_checkpoint(
                    model_path=cp,
                    dataset_name=dataset,
                    num_samples=args.samples,
                    checkpoint_label=label,
                    model_type="hf"
                )
                if dataset not in results:
                    results[dataset] = {}
                results[dataset][label] = result["summary"]["accuracy"]
            
            # Track best
            avg_acc = sum(results[ds].get(label, 0) for ds in args.datasets) / len(args.datasets)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_checkpoint = cp
        
        if best_checkpoint:
            logger.info(f"Best checkpoint: {best_checkpoint} with avg accuracy {best_acc:.2%}")
            
            # Save results
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            
            # Also save best checkpoint path
            with open("craft_output/best_checkpoint.txt", "w") as f:
                f.write(best_checkpoint)
        
    elif args.baseline:
        # Evaluate baseline model
        base_model = "microsoft/Phi-3-mini-4k-instruct"
        results = {}
        for dataset in args.datasets:
            result = evaluator.evaluate_checkpoint(
                model_path=base_model,
                dataset_name=dataset,
                num_samples=args.samples,
                checkpoint_label="baseline",
                model_type="hf"
            )
            results[dataset] = result["summary"]
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Baseline evaluation saved to {args.output}")
        
    elif args.sft:
        # Evaluate SFT model
        sft_path = "checkpoints/sft/final"
        results = {}
        for dataset in args.datasets:
            result = evaluator.evaluate_checkpoint(
                model_path=sft_path,
                dataset_name=dataset,
                num_samples=args.samples,
                checkpoint_label="sft",
                model_type="hf"
            )
            results[dataset] = result["summary"]
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"SFT evaluation saved to {args.output}")
        
    else:
        # Evaluate single checkpoint
        checkpoint = args.checkpoint or "checkpoints/rl/final"
        results = {}
        for dataset in args.datasets:
            result = evaluator.evaluate_checkpoint(
                model_path=checkpoint,
                dataset_name=dataset,
                num_samples=args.samples,
                checkpoint_label="craft",
                model_type="hf"
            )
            results[dataset] = result["summary"]
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation saved to {args.output}")
    
    return 0


def quantize_command(args) -> int:
    """Quantize model to GGUF."""
    from craft.phase3_deploy.quantizer import main as quantizer_main
    import sys
    
    logger.info("Starting model quantization...")
    
    # Save args and run quantizer
    old_argv = sys.argv
    sys.argv = [
        "quantizer.py",
        "--base-model", args.base_model,
        "--lora-adapter", args.checkpoint,
        "--gguf-quantized", args.output,
        "--quantization", args.bits
    ]
    
    try:
        quantizer_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    
    logger.info(f"Quantization complete. Model saved to {args.output}")
    return 0


def package_command(args) -> int:
    """Package model for deployment."""
    from craft.phase3_deploy.packager import DeliveryPackager
    
    logger.info("Packaging model for deployment...")
    
    packager = DeliveryPackager(delivery_dir=args.output)
    packager.package_all(args.checkpoint)
    
    logger.info(f"Package created in {args.output}")
    return 0


def deploy_command(args) -> int:
    """Deploy model."""
    import subprocess
    import os
    
    logger.info("Starting deployment server...")
    
    # Check if package exists
    server_path = os.path.join(args.checkpoint, "src", "inference", "inference_server.py")
    
    if not os.path.exists(server_path):
        logger.error(f"Package not found at {args.checkpoint}. Run 'craft package' first.")
        return 1
    
    # Run the inference server
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.inference.inference_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    
    logger.info(f"Starting server: {' '.join(cmd)}")
    logger.info(f"Open src/ui/index.html in your browser to interact")
    
    subprocess.run(cmd, cwd=args.checkpoint)
    
    return 0


def report_command(args) -> int:
    """Generate benchmark report."""
    from craft.phase3_deploy.report_generator import ReportGenerator
    
    logger.info("Generating benchmark report...")
    
    gen = ReportGenerator()
    gen.generate_full_report(
        baseline_json=args.baseline_json,
        champion_json=args.champion_json,
        champion_label="CRAFT",
        extra_checkpoints={},
        report_md_path=args.output
    )
    
    logger.info(f"Report saved to {args.output}")
    return 0


def run_all_command(args) -> int:
    """Run complete pipeline."""
    logger.info("🚀 Running complete CRAFT pipeline")
    print("\n" + "=" * 60)
    print("CRAFT - Complete Pipeline")
    print("=" * 60 + "\n")
    
    # Phase 0: Probe
    print("📊 Phase 0: Difficulty Probing")
    print("-" * 40)
    probe_args = argparse.Namespace(
        config=args.config,
        hardware=args.hardware,
        output="Outputs/difficulty_map.json",
        model="microsoft/Phi-3-mini-4k-instruct",
        dry_run=False
    )
    probe_command(probe_args)
    
    # Phase 1: SFT
    print("\n📚 Phase 1: Supervised Fine-Tuning")
    print("-" * 40)
    sft_args = argparse.Namespace(
        config=args.config,
        hardware=args.hardware,
        output=f"{args.output}/sft",
        resume=args.resume,
        epochs=None,
        batch_size=None
    )
    sft_command(sft_args)
    
    # Phase 2: RL
    print("\n🎯 Phase 2: Reinforcement Learning")
    print("-" * 40)
    rl_args = argparse.Namespace(
        config=args.config,
        hardware=args.hardware,
        checkpoint=f"{args.output}/sft/final",
        output=f"{args.output}/rl",
        resume=args.resume,
        steps=500
    )
    rl_command(rl_args)
    
    # Phase 3: Evaluate and find best
    print("\n📈 Phase 3: Evaluation")
    print("-" * 40)
    eval_args = argparse.Namespace(
        checkpoint=f"{args.output}/rl/final",
        output="evaluation.json",
        datasets=["gsm8k", "strategyqa", "mmlu"],
        samples=100,
        baseline=False,
        sft=False,
        all=True
    )
    evaluate_command(eval_args)
    
    # Phase 3: Quantize
    print("\n🔧 Quantization")
    print("-" * 40)
    quant_args = argparse.Namespace(
        checkpoint=f"{args.output}/rl/final",
        output="craft_output/craft_phi3_Q4_K_M.gguf",
        bits="Q4_K_M",
        base_model="microsoft/Phi-3-mini-4k-instruct"
    )
    quantize_command(quant_args)
    
    # Phase 3: Package
    print("\n📦 Packaging")
    print("-" * 40)
    package_args = argparse.Namespace(
        checkpoint="craft_output/craft_phi3_Q4_K_M.gguf",
        output="craft_delivery"
    )
    package_command(package_args)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    
    return 0


def checkpoints_command(args) -> int:
    """List checkpoints."""
    from craft.utils.checkpoint_manager import CheckpointManager
    import glob
    
    print(f"\n📁 Checkpoints in {args.dir}/")
    print("-" * 50)
    
    # SFT checkpoints
    sft_dirs = sorted(glob.glob(f"{args.dir}/sft/checkpoint-*"))
    if sft_dirs:
        print("\nSFT Checkpoints:")
        for d in sft_dirs:
            print(f"  📍 {d}")
    
    # RL checkpoints
    rl_dirs = sorted(glob.glob(f"{args.dir}/rl/checkpoint-*"))
    if rl_dirs:
        print("\nRL Checkpoints:")
        for d in rl_dirs:
            print(f"  📍 {d}")
    
    # Final models
    final_dirs = [
        f"{args.dir}/sft/final",
        f"{args.dir}/rl/final",
        "craft_output/craft_phi3_Q4_K_M.gguf"
    ]
    print("\nFinal Models:")
    for d in final_dirs:
        if os.path.exists(d):
            print(f"  ✅ {d}")
    
    return 0


def clean_command(args) -> int:
    """Clean temporary files."""
    import shutil
    
    dirs_to_clean = ["logs", "__pycache__"]
    if args.all:
        dirs_to_clean.extend(["checkpoints", "craft_output", "Outputs"])
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"🗑️ Removed {d}/")
    
    # Also clean Python cache files
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            shutil.rmtree(cache_path)
            print(f"🗑️ Removed {cache_path}")
    
    print("✅ Clean completed")
    return 0


def config_command(args) -> int:
    """Print current configuration."""
    from craft.config import load_config
    
    config = load_config(args.config, args.hardware)
    
    print(f"\n📋 Configuration: {args.config} on {args.hardware}")
    print("-" * 50)
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                print_dict(value, indent + 1)
            else:
                print(f"{'  ' * indent}{key}: {value}")
    
    print_dict(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())