#!/usr/bin/env python3
"""
Unified Experiment Pipeline for OMAGD

Runs complete training pipeline: Stage 1 → Stage 2 → Stage 3 (+ optional baselines)

Usage:
    # Run with config file
    python src/run_pipeline.py --config experiments/8m_vs_9m.yaml
    
    # Run specific stages
    python src/run_pipeline.py --config experiments/8m_vs_9m.yaml --stages stage1,stage2
    
    # Resume from existing experiment
    python src/run_pipeline.py --resume results/experiments/my_exp_20260121/
    
    # Quick run with defaults
    python src/run_pipeline.py --map 8m_vs_9m --seed 1
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.experiment_config import ExperimentConfig, create_default_config
from utils.stage_manager import StageManager, StageStatus
from utils.results_aggregator import ResultsAggregator
from utils.gpu_monitor import GPUMonitor, get_gpu_info


class PipelineRunner:
    """
    Orchestrates the multi-stage training pipeline.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        stages: Optional[List[str]] = None,
        force: bool = False,
        dry_run: bool = False
    ):
        """
        Initialize pipeline runner.
        
        Args:
            config: Experiment configuration
            stages: List of stages to run (None = all enabled stages)
            force: Force re-run of completed stages
            dry_run: Print commands without executing
        """
        self.config = config
        self.force = force
        self.dry_run = dry_run
        
        # Determine which stages to run: stage1 → stage2 → stage3
        all_stages = []
        if config.stage1.enabled:
            all_stages.append("stage1")
        if config.stage2.enabled:
            all_stages.append("stage2")
        if config.stage3.enabled:
            all_stages.append("stage3")
        
        self.stages = stages if stages else all_stages
        
        # Initialize managers
        self.stage_manager = StageManager(config.experiment_dir)
        self.results_aggregator = ResultsAggregator(
            experiment_dir=config.experiment_dir,
            experiment_name=config.name,
            map_name=config.map_name,
            seed=config.seeds[0]  # Primary seed
        )
        
        # GPU monitor
        self.gpu_monitor = GPUMonitor(gpu_id=config.gpu_id, sample_interval=2.0)
        
        # Paths for sharing between stages
        self.tokenizer_path: Optional[str] = None
        self.mask_predictor_path: Optional[str] = None
        self.buffer_path: Optional[str] = None
        
        # Load paths from previously completed stages (for resume)
        self._load_completed_stage_paths()
    
    def _load_completed_stage_paths(self):
        """Load model/buffer paths from previously completed stages."""
        # Check stage1
        if self.stage_manager.get_stage_status("stage1").value == "completed":
            self.tokenizer_path = self.stage_manager.get_model_path("stage1")
            self.buffer_path = self.stage_manager.get_buffer_path("stage1")
            if self.tokenizer_path:
                print(f"[Resume] Loaded stage1 tokenizer: {self.tokenizer_path}")
            if self.buffer_path:
                print(f"[Resume] Loaded stage1 buffer: {self.buffer_path}")
        
        # Check stage2
        if self.stage_manager.get_stage_status("stage2").value == "completed":
            self.mask_predictor_path = self.stage_manager.get_model_path("stage2")
            if self.mask_predictor_path:
                print(f"[Resume] Loaded stage2 mask predictor: {self.mask_predictor_path}")
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            True if all stages completed successfully
        """
        print("\n" + "=" * 70)
        print("OMAGD EXPERIMENT PIPELINE")
        print("=" * 70)
        print(f"Experiment: {self.config.name}")
        print(f"Map: {self.config.map_name}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Stages to run: {self.stages}")
        print(f"Output directory: {self.config.experiment_dir}")
        print("=" * 70 + "\n")
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        # Save configuration
        os.makedirs(self.config.experiment_dir, exist_ok=True)
        self.config.save()
        print(f"Configuration saved to {self.config.experiment_dir}/config.yaml\n")
        
        # Print GPU info
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            print(f"GPU: {gpu_info['devices'][self.config.gpu_id]['name']}")
            print(f"GPU Memory: {gpu_info['devices'][self.config.gpu_id]['memory_total_mb']:.0f} MB")
        print()
        
        success = True
        
        # Run each stage
        for seed in self.config.seeds:
            print(f"\n{'='*50}")
            print(f"Running with seed: {seed}")
            print(f"{'='*50}\n")
            
            for stage in self.stages:
                try:
                    stage_success = self._run_stage(stage, seed)
                    if not stage_success:
                        print(f"\n[PIPELINE] Stage {stage} failed. Stopping pipeline.")
                        success = False
                        break
                except Exception as e:
                    print(f"\n[PIPELINE] Error in stage {stage}: {e}")
                    self.stage_manager.fail_stage(stage, str(e))
                    success = False
                    break
        
        # Generate final report
        if not self.dry_run:
            self.results_aggregator.print_summary()
            report_path = self.results_aggregator.generate_report()
            print(f"\nReport saved to: {report_path}")
        
        # Cleanup
        self.gpu_monitor.shutdown()
        
        return success
    
    def _run_stage(self, stage: str, seed: int) -> bool:
        """
        Run a single stage.
        
        Args:
            stage: Stage name
            seed: Random seed
            
        Returns:
            True if stage completed successfully
        """
        # Check if should run
        if not self.stage_manager.should_run(stage, force=self.force):
            print(f"[{stage}] Already completed, skipping (use --force to re-run)")
            
            # Load paths from completed stage
            if stage == "stage1":
                self.tokenizer_path = self.stage_manager.get_model_path("stage1")
                self.buffer_path = self.stage_manager.get_buffer_path("stage1")
            elif stage == "stage2":
                self.mask_predictor_path = self.stage_manager.get_model_path("stage2")
            
            return True
        
        # Check dependencies
        deps_met, missing = self.stage_manager.check_dependencies(stage)
        if not deps_met:
            print(f"[{stage}] Dependencies not met. Missing: {missing}")
            return False
        
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.upper()}")
        print(f"{'='*60}")
        
        # Get stage arguments
        args = self.config.get_stage_args(stage, seed)
        
        # Add pretrained paths for stage2/stage3
        if stage == "stage2":
            if self.tokenizer_path:
                args["pretrained_tokenizer_path"] = self.tokenizer_path
            else:
                print(f"[{stage}] ERROR: Tokenizer path not available")
                return False
            
            # Use shared buffer from stage1
            if self.buffer_path and self.config.stage2.reuse_stage1_buffer:
                args["pretrain_buffer_path"] = self.buffer_path
        
        elif stage == "stage3":
            if self.tokenizer_path:
                args["pretrained_tokenizer_path"] = self.tokenizer_path
            else:
                print(f"[{stage}] ERROR: Tokenizer path not available")
                return False
            
            if self.mask_predictor_path:
                args["pretrained_mask_predictor_path"] = self.mask_predictor_path
            else:
                print(f"[{stage}] ERROR: Mask predictor path not available")
                return False
        
        # Set output directory
        stage_dir = self.config.get_experiment_subdir(stage, seed)
        args["local_results_path"] = stage_dir
        
        # Build command
        cmd = self._build_command(stage, args)
        
        if self.dry_run:
            print(f"[DRY RUN] Would execute:\n  {cmd}\n")
            return True
        
        # Start monitoring
        self.gpu_monitor.start_stage(stage)
        self.stage_manager.start_stage(stage)
        start_time = time.time()
        
        # Execute
        print(f"[{stage}] Starting training...")
        print(f"[{stage}] Command: {cmd[:200]}..." if len(cmd) > 200 else f"[{stage}] Command: {cmd}")
        
        success = self._execute_command(cmd, stage)
        
        # End monitoring
        gpu_stats = self.gpu_monitor.end_stage()
        duration = time.time() - start_time
        
        if success:
            # Find model path
            model_path = self._find_model_path(stage, stage_dir)
            buffer_path_found = self._find_buffer_path(stage_dir) if stage == "stage1" else ""
            
            # Update paths for next stages
            if stage == "stage1":
                self.tokenizer_path = model_path
                self.buffer_path = buffer_path_found
            elif stage == "stage2":
                self.mask_predictor_path = model_path
            
            # Parse metrics from log (simplified - could be enhanced)
            metrics = self._parse_metrics(stage, stage_dir)
            
            # Record completion
            self.stage_manager.complete_stage(
                stage,
                model_path=model_path,
                buffer_path=buffer_path_found,
                metrics=metrics,
                gpu_stats=gpu_stats.to_dict() if gpu_stats else {}
            )
            
            self.results_aggregator.add_stage_result(
                stage,
                status="completed",
                duration_sec=duration,
                model_path=model_path,
                metrics=metrics,
                gpu_stats=gpu_stats.to_dict() if gpu_stats else {}
            )
            
            print(f"[{stage}] Completed successfully in {duration:.1f}s")
            if model_path:
                print(f"[{stage}] Model saved to: {model_path}")
            
            return True
        else:
            self.stage_manager.fail_stage(stage, "Training process failed")
            self.results_aggregator.add_stage_result(
                stage,
                status="failed",
                duration_sec=duration,
                gpu_stats=gpu_stats.to_dict() if gpu_stats else {}
            )
            return False
    
    def _build_command(self, stage: str, args: Dict[str, Any]) -> str:
        """Build the command to execute."""
        # All stages use omagd config
        cmd_parts = [
            sys.executable,  # Python interpreter
            "src/main.py",
            "--config=omagd",
            "--env-config=sc2",
            "with"
        ]
        
        # Add arguments
        for key, value in args.items():
            if isinstance(value, bool):
                cmd_parts.append(f"{key}={str(value)}")
            elif isinstance(value, str):
                if value:  # Only add non-empty strings
                    cmd_parts.append(f'{key}="{value}"')
            else:
                cmd_parts.append(f"{key}={value}")
        
        return " ".join(cmd_parts)
    
    def _execute_command(self, cmd: str, stage: str) -> bool:
        """Execute a command and return success status."""
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_id)
        
        # Log file
        log_dir = os.path.join(self.config.experiment_dir, stage, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        try:
            # Run with output to both console and log file
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Stream output
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
                return process.returncode == 0
                
        except Exception as e:
            print(f"[{stage}] Execution error: {e}")
            return False
    
    def _find_model_path(self, stage: str, stage_dir: str) -> str:
        """Find the model path after training."""
        import glob
        
        # Look for best model first
        if stage in ["stage1", "stage2"]:
            pattern = os.path.join(stage_dir, "models", "**", f"pretrain_{stage}_best")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return max(matches, key=os.path.getmtime)
            
            # Fall back to final model
            pattern = os.path.join(stage_dir, "models", "**", f"pretrain_{stage}")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return max(matches, key=os.path.getmtime)
        
        elif stage == "stage3":
            # Look for latest checkpoint
            pattern = os.path.join(stage_dir, "models", "**", "[0-9]*")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Get the highest timestep
                return max(matches, key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else 0)
        
        return ""
    
    def _find_buffer_path(self, stage_dir: str) -> str:
        """Find saved buffer path."""
        import glob
        pattern = os.path.join(stage_dir, "**", "pretrain_buffer_*.pt")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return max(matches, key=os.path.getmtime)
        return ""
    
    def _parse_metrics(self, stage: str, stage_dir: str) -> Dict:
        """Parse metrics from training logs (placeholder - can be enhanced)."""
        # This is a simplified version - could be enhanced to parse actual logs
        metrics = {}
        
        # Try to find metrics.json if it exists
        metrics_file = os.path.join(stage_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            except Exception:
                pass
        
        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OMAGD Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (auto-resumes if incomplete experiment exists)
  python src/run_pipeline.py --map 8m_vs_9m

  # Specify stage3 training steps
  python src/run_pipeline.py --map 8m_vs_9m --t-max 2000000

  # Force start new experiment (don't auto-resume)
  python src/run_pipeline.py --map 8m_vs_9m --new

  # Run with multiple seeds
  python src/run_pipeline.py --map 8m_vs_9m --seed 1 2 3

  # Run only pretrain stages
  python src/run_pipeline.py --map 8m_vs_9m --stages stage1,stage2

  # Resume from specific experiment directory
  python src/run_pipeline.py --resume results/experiments/omagd_8m_vs_9m_20260121/

For baselines, use existing scripts:
  ./run_gnn_baseline.sh
  ./run_gnn_full_obs.sh
        """
    )
    
    # Config options
    parser.add_argument("--config", type=str, help="Path to experiment config YAML file")
    parser.add_argument("--resume", type=str, help="Resume from existing experiment directory")
    
    # Quick setup options (used when no config file)
    parser.add_argument("--map", type=str, default="8m_vs_9m", help="Map name")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--seed", type=int, nargs="+", default=[1], help="Random seed(s)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--t-max", type=int, default=5000000, help="Stage 3 training steps (default: 5000000)")
    
    # Stage control
    parser.add_argument("--stages", type=str, help="Comma-separated list of stages to run (stage1,stage2,stage3)")
    
    # Execution options
    parser.add_argument("--force", action="store_true", help="Force re-run of completed stages")
    parser.add_argument("--new", action="store_true", help="Start new experiment (don't auto-resume)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results/experiments",
                       help="Base directory for experiment results")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load or create configuration
    if args.resume:
        # Resume from existing experiment
        config_path = os.path.join(args.resume, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Error: Config not found at {config_path}")
            sys.exit(1)
        config = ExperimentConfig.from_yaml(config_path)
        config.experiment_dir = args.resume
        print(f"Resuming experiment from {args.resume}")
        
    elif args.config:
        # Load from config file
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = ExperimentConfig.from_yaml(args.config)
        
    else:
        # Create default configuration from command line args
        config = create_default_config(
            map_name=args.map,
            name=args.name or f"omagd_{args.map}"
        )
        config.seeds = args.seed
        config.gpu_id = args.gpu
        config.results_dir = args.output_dir
        
        # If --new flag, force new experiment directory
        if args.new:
            from datetime import datetime
            config.experiment_dir = os.path.join(
                config.results_dir,
                f"{config.name}_{config.map_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"[Config] Starting new experiment (--new flag)")
    
    # Override with command line arguments
    if args.gpu is not None:
        config.gpu_id = args.gpu
    
    # Set stage 3 t_max
    if hasattr(args, 't_max') and args.t_max is not None:
        config.stage3.t_max = args.t_max
    
    # Parse stages to run
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
    
    # Create and run pipeline
    runner = PipelineRunner(
        config=config,
        stages=stages,
        force=args.force,
        dry_run=args.dry_run
    )
    
    success = runner.run()
    
    if success:
        print("\n[PIPELINE] All stages completed successfully!")
        sys.exit(0)
    else:
        print("\n[PIPELINE] Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
