"""
Results Aggregator for Experiment Pipeline

Collects, aggregates, and reports results from all training stages.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class StageResults:
    """Results from a single training stage."""
    stage_name: str
    status: str
    duration_sec: float = 0.0
    
    # Paths
    model_path: str = ""
    
    # GPU statistics
    gpu_stats: Dict = field(default_factory=dict)
    
    # Stage-specific metrics
    metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 2),
            "model_path": self.model_path,
            "gpu_stats": self.gpu_stats,
            "metrics": self.metrics
        }


@dataclass
class ExperimentResults:
    """Complete results for an experiment run."""
    experiment_name: str
    map_name: str
    seed: int
    timestamp: str
    
    # Stage results
    stages: Dict[str, StageResults] = field(default_factory=dict)
    
    # Overall statistics
    total_duration_sec: float = 0.0
    total_gpu_memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "experiment_name": self.experiment_name,
            "map_name": self.map_name,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "total_gpu_memory_peak_mb": round(self.total_gpu_memory_peak_mb, 2),
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()}
        }
    
    def add_stage(self, stage: StageResults):
        """Add a stage result."""
        self.stages[stage.stage_name] = stage
        self._update_totals()
    
    def _update_totals(self):
        """Update total statistics."""
        self.total_duration_sec = sum(s.duration_sec for s in self.stages.values())
        
        peak_memories = []
        for s in self.stages.values():
            if s.gpu_stats and "memory" in s.gpu_stats:
                peak_memories.append(s.gpu_stats["memory"].get("peak_mb", 0))
        if peak_memories:
            self.total_gpu_memory_peak_mb = max(peak_memories)


class ResultsAggregator:
    """
    Aggregates and reports results from experiment pipeline.
    
    Usage:
        aggregator = ResultsAggregator(experiment_dir="/path/to/experiment")
        
        # Add results as stages complete
        aggregator.add_stage_result("stage1", metrics={...}, gpu_stats={...})
        
        # Generate final report
        aggregator.generate_report()
        
        # Get summary for comparison
        summary = aggregator.get_summary()
    """
    
    RESULTS_FILE = "all_results.json"
    REPORT_FILE = "experiment_report.md"
    
    def __init__(
        self, 
        experiment_dir: str,
        experiment_name: str = "experiment",
        map_name: str = "unknown",
        seed: int = 1
    ):
        """
        Initialize results aggregator.
        
        Args:
            experiment_dir: Root directory for experiment results
            experiment_name: Name of the experiment
            map_name: Map name for the experiment
            seed: Random seed
        """
        self.experiment_dir = experiment_dir
        self.summary_dir = os.path.join(experiment_dir, "summary")
        
        self.results = ExperimentResults(
            experiment_name=experiment_name,
            map_name=map_name,
            seed=seed,
            timestamp=datetime.now().isoformat()
        )
        
        # Load existing results if available
        self._load_results()
    
    def _load_results(self):
        """Load existing results from disk."""
        results_path = os.path.join(self.summary_dir, self.RESULTS_FILE)
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                # Restore stages
                for stage_name, stage_data in data.get("stages", {}).items():
                    self.results.stages[stage_name] = StageResults(**stage_data)
                self.results._update_totals()
                print(f"[ResultsAggregator] Loaded existing results from {results_path}")
            except Exception as e:
                print(f"[ResultsAggregator] Warning: Failed to load results: {e}")
    
    def _save_results(self):
        """Save results to disk."""
        os.makedirs(self.summary_dir, exist_ok=True)
        results_path = os.path.join(self.summary_dir, self.RESULTS_FILE)
        with open(results_path, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
    
    def add_stage_result(
        self,
        stage_name: str,
        status: str = "completed",
        duration_sec: float = 0.0,
        model_path: str = "",
        metrics: Optional[Dict] = None,
        gpu_stats: Optional[Dict] = None
    ):
        """
        Add results for a completed stage.
        
        Args:
            stage_name: Name of the stage
            status: Status ("completed", "failed", "skipped")
            duration_sec: Duration in seconds
            model_path: Path to saved model
            metrics: Stage-specific metrics
            gpu_stats: GPU usage statistics
        """
        stage = StageResults(
            stage_name=stage_name,
            status=status,
            duration_sec=duration_sec,
            model_path=model_path,
            metrics=metrics or {},
            gpu_stats=gpu_stats or {}
        )
        self.results.add_stage(stage)
        self._save_results()
        
        print(f"[ResultsAggregator] Added results for {stage_name}")
    
    def get_summary(self) -> Dict:
        """Get a summary of all results."""
        return self.results.to_dict()
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResults]:
        """Get results for a specific stage."""
        return self.results.stages.get(stage_name)
    
    def generate_report(self) -> str:
        """
        Generate a markdown report of the experiment.
        
        Returns:
            Path to the generated report file
        """
        os.makedirs(self.summary_dir, exist_ok=True)
        report_path = os.path.join(self.summary_dir, self.REPORT_FILE)
        
        lines = []
        
        # Header
        lines.append(f"# Experiment Report: {self.results.experiment_name}")
        lines.append("")
        lines.append(f"**Map:** {self.results.map_name}")
        lines.append(f"**Seed:** {self.results.seed}")
        lines.append(f"**Timestamp:** {self.results.timestamp}")
        lines.append(f"**Total Duration:** {self._format_duration(self.results.total_duration_sec)}")
        lines.append(f"**Peak GPU Memory:** {self.results.total_gpu_memory_peak_mb:.1f} MB")
        lines.append("")
        
        # Stage Summary Table
        lines.append("## Stage Summary")
        lines.append("")
        lines.append("| Stage | Status | Duration | GPU Memory (Peak) | GPU Util (Avg) |")
        lines.append("|-------|--------|----------|-------------------|----------------|")
        
        for stage_name in ["stage1", "stage2", "stage3"]:
            if stage_name in self.results.stages:
                stage = self.results.stages[stage_name]
                duration = self._format_duration(stage.duration_sec)
                
                gpu_mem = "-"
                gpu_util = "-"
                if stage.gpu_stats:
                    if "memory" in stage.gpu_stats:
                        gpu_mem = f"{stage.gpu_stats['memory'].get('peak_mb', 0):.1f} MB"
                    if "utilization" in stage.gpu_stats:
                        gpu_util = f"{stage.gpu_stats['utilization'].get('avg_percent', 0):.1f}%"
                
                status_emoji = "✅" if stage.status == "completed" else "❌" if stage.status == "failed" else "⏭️"
                lines.append(f"| {stage_name} | {status_emoji} {stage.status} | {duration} | {gpu_mem} | {gpu_util} |")
        
        lines.append("")
        
        # Detailed Stage Results
        lines.append("## Detailed Results")
        lines.append("")
        
        for stage_name, stage in self.results.stages.items():
            lines.append(f"### {stage_name.upper()}")
            lines.append("")
            
            if stage.model_path:
                lines.append(f"**Model Path:** `{stage.model_path}`")
                lines.append("")
            
            # Key metrics
            if stage.metrics:
                lines.append("**Key Metrics:**")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                
                # Select important metrics based on stage
                important_keys = self._get_important_metrics(stage_name)
                for key in important_keys:
                    if key in stage.metrics:
                        value = stage.metrics[key]
                        if isinstance(value, float):
                            lines.append(f"| {key} | {value:.4f} |")
                        else:
                            lines.append(f"| {key} | {value} |")
                
                lines.append("")
            
            # GPU stats
            if stage.gpu_stats:
                lines.append("**GPU Statistics:**")
                lines.append("")
                if "memory" in stage.gpu_stats:
                    mem = stage.gpu_stats["memory"]
                    lines.append(f"- Memory Peak: {mem.get('peak_mb', 0):.1f} MB ({mem.get('peak_percent', 0):.1f}%)")
                    lines.append(f"- Memory Avg: {mem.get('avg_mb', 0):.1f} MB")
                if "utilization" in stage.gpu_stats:
                    util = stage.gpu_stats["utilization"]
                    lines.append(f"- Utilization Peak: {util.get('peak_percent', 0):.1f}%")
                    lines.append(f"- Utilization Avg: {util.get('avg_percent', 0):.1f}%")
                if "temperature" in stage.gpu_stats:
                    temp = stage.gpu_stats["temperature"]
                    if temp.get("peak_celsius"):
                        lines.append(f"- Temperature Peak: {temp['peak_celsius']:.1f}°C")
                lines.append("")
        
        # Write report
        report_content = "\n".join(lines)
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"[ResultsAggregator] Generated report: {report_path}")
        return report_path
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    def _get_important_metrics(self, stage_name: str) -> List[str]:
        """Get list of important metrics for a stage."""
        if stage_name == "stage1":
            return [
                "loss", "best_eval_loss", "mse", "cosine_similarity",
                "perplexity", "codebook_usage", "feature_correlation"
            ]
        elif stage_name == "stage2":
            return [
                "loss", "best_eval_loss", "masked_accuracy", 
                "top3_accuracy", "top5_accuracy", "mrr", "hungarian_accuracy"
            ]
        elif stage_name == "stage3":
            return [
                "test_return_mean", "test_win_rate", "best_win_rate",
                "final_return", "episodes_trained"
            ]
        return list(self.results.stages.get(stage_name, StageResults(stage_name, "")).metrics.keys())[:10]
    
    def print_summary(self):
        """Print a summary to console."""
        print("\n" + "=" * 60)
        print(f"EXPERIMENT SUMMARY: {self.results.experiment_name}")
        print("=" * 60)
        print(f"Map: {self.results.map_name}, Seed: {self.results.seed}")
        print(f"Total Duration: {self._format_duration(self.results.total_duration_sec)}")
        print(f"Peak GPU Memory: {self.results.total_gpu_memory_peak_mb:.1f} MB")
        print("-" * 60)
        
        for stage_name in ["stage1", "stage2", "stage3"]:
            if stage_name in self.results.stages:
                stage = self.results.stages[stage_name]
                status_char = "✓" if stage.status == "completed" else "✗" if stage.status == "failed" else "→"
                print(f"  [{status_char}] {stage_name}: {self._format_duration(stage.duration_sec)}")
                
                # Print key metric
                key_metric = None
                if stage_name == "stage1" and "mse" in stage.metrics:
                    key_metric = f"MSE: {stage.metrics['mse']:.4f}"
                elif stage_name == "stage2" and "masked_accuracy" in stage.metrics:
                    key_metric = f"Accuracy: {stage.metrics['masked_accuracy']:.4f}"
                elif stage_name in ["stage3", "baseline", "fullobs"] and "test_win_rate" in stage.metrics:
                    key_metric = f"Win Rate: {stage.metrics['test_win_rate']:.4f}"
                
                if key_metric:
                    print(f"      → {key_metric}")
        
        print("=" * 60 + "\n")


def compare_experiments(experiment_dirs: List[str]) -> Dict:
    """
    Compare results across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        
    Returns:
        Comparison dictionary
    """
    comparison = {
        "experiments": [],
        "stage_comparison": {}
    }
    
    for exp_dir in experiment_dirs:
        results_path = os.path.join(exp_dir, "summary", ResultsAggregator.RESULTS_FILE)
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                comparison["experiments"].append({
                    "name": data.get("experiment_name", "unknown"),
                    "map": data.get("map_name", "unknown"),
                    "seed": data.get("seed", 0),
                    "total_duration": data.get("total_duration_sec", 0)
                })
                
                # Compare stages
                for stage_name, stage_data in data.get("stages", {}).items():
                    if stage_name not in comparison["stage_comparison"]:
                        comparison["stage_comparison"][stage_name] = []
                    comparison["stage_comparison"][stage_name].append({
                        "experiment": data.get("experiment_name", "unknown"),
                        "metrics": stage_data.get("metrics", {})
                    })
    
    return comparison


if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test aggregator
        aggregator = ResultsAggregator(
            experiment_dir=tmpdir,
            experiment_name="test_experiment",
            map_name="8m_vs_9m",
            seed=1
        )
        
        # Add stage results
        aggregator.add_stage_result(
            "stage1",
            status="completed",
            duration_sec=1823.5,
            model_path="/path/to/stage1/model",
            metrics={"loss": 0.0234, "mse": 0.0198, "codebook_usage": 0.85},
            gpu_stats={
                "memory": {"peak_mb": 4096, "avg_mb": 3500, "peak_percent": 50.0},
                "utilization": {"peak_percent": 95, "avg_percent": 75}
            }
        )
        
        aggregator.add_stage_result(
            "stage2",
            status="completed",
            duration_sec=2456.2,
            model_path="/path/to/stage2/model",
            metrics={"loss": 0.156, "masked_accuracy": 0.78, "top3_accuracy": 0.91},
            gpu_stats={
                "memory": {"peak_mb": 5120, "avg_mb": 4200, "peak_percent": 62.5},
                "utilization": {"peak_percent": 90, "avg_percent": 70}
            }
        )
        
        # Print summary
        aggregator.print_summary()
        
        # Generate report
        report_path = aggregator.generate_report()
        print(f"\nReport generated at: {report_path}")
        
        # Print report content
        with open(report_path, 'r') as f:
            print("\n" + f.read())
