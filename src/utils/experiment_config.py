"""
Experiment Configuration Management for Pipeline

Provides unified configuration management for multi-stage training pipeline.
Uses existing config/algs/omagd.yaml and config/algs/qmix.yaml as base.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from copy import deepcopy


@dataclass
class Stage1Config:
    """Stage 1: Tokenizer Training Configuration."""
    enabled: bool = True
    pretrain_episodes: int = 5000
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 2
    save_buffer: bool = True
    buffer_path: str = ""  # Optional: use existing buffer
    
    # Learning
    learning_rate: float = 0.0005
    log_interval: int = 1
    eval_interval: int = 20
    eval_episodes: int = 40


@dataclass
class Stage2Config:
    """Stage 2: Mask Predictor Training Configuration."""
    enabled: bool = True
    pretrain_epochs: int = 100
    pretrain_batch_size: int = 2
    reuse_stage1_buffer: bool = True  # Share buffer with Stage 1
    
    # Learning
    learning_rate: float = 0.0005
    log_interval: int = 1
    eval_interval: int = 20
    eval_episodes: int = 40


@dataclass
class Stage3Config:
    """Stage 3: QMIX Training with Reconstruction Configuration."""
    enabled: bool = True
    t_max: int = 5000000
    test_interval: int = 10000
    test_nepisode: int = 32
    save_model_interval: int = 100000
    log_interval: int = 10000
    
    # Agent configuration
    use_gnn: bool = True
    gnn_layer_num: int = 2


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration.
    
    Uses existing config files:
    - config/algs/omagd.yaml for Stage 1, 2, 3
    - config/algs/qmix.yaml for baselines
    
    Usage:
        # Create with command line args (most common)
        config = ExperimentConfig(
            name="my_experiment",
            map_name="8m_vs_9m",
            seeds=[1, 2, 3]
        )
        
        # Run pipeline
        python src/run_pipeline.py --map 8m_vs_9m --seed 1 2 3
    """
    # Basic info
    name: str = "omagd"
    map_name: str = "8m_vs_9m"
    seeds: List[int] = field(default_factory=lambda: [1])
    
    # GPU configuration
    gpu_id: int = 0
    use_cuda: bool = True
    
    # Stage configurations
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    
    # Output configuration
    results_dir: str = "results/experiments"
    save_metrics: bool = True
    save_models: bool = True
    generate_report: bool = True
    
    # W&B configuration
    use_wandb: bool = True
    wandb_project_prefix: str = "omagd_pipeline"
    wandb_entity: Optional[str] = None
    
    # Runtime (set automatically)
    experiment_dir: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        """Initialize runtime fields."""
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.experiment_dir:
            # Check for existing incomplete experiment to auto-resume
            existing_dir = self._find_incomplete_experiment()
            if existing_dir:
                self.experiment_dir = existing_dir
                print(f"[Config] Found incomplete experiment, will resume from: {existing_dir}")
            else:
                self.experiment_dir = os.path.join(
                    self.results_dir, 
                    f"{self.name}_{self.map_name}_{self.timestamp}"
                )
    
    def _find_incomplete_experiment(self) -> Optional[str]:
        """Find an existing incomplete experiment for the same map to auto-resume."""
        import glob
        pattern = os.path.join(self.results_dir, f"{self.name}_{self.map_name}_*")
        matches = glob.glob(pattern)
        
        if not matches:
            return None
        
        # Sort by modification time (most recent first)
        matches.sort(key=os.path.getmtime, reverse=True)
        
        # Check if the most recent experiment is incomplete
        for exp_dir in matches:
            state_file = os.path.join(exp_dir, "pipeline_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    stages = state.get("stages", {})
                    
                    # Check if any stage failed or is still running
                    has_failed = any(
                        s.get("status") in ["failed", "running"]
                        for s in stages.values()
                    )
                    
                    # Check if not all expected stages are completed
                    completed_stages = [
                        name for name, s in stages.items() 
                        if s.get("status") == "completed"
                    ]
                    all_stages_done = set(completed_stages) >= {"stage1", "stage2", "stage3"}
                    
                    if has_failed or not all_stages_done:
                        # Found incomplete experiment - resume from here
                        return exp_dir
                except Exception:
                    pass
        
        return None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        data = data.copy()
        # Extract nested configs
        stage1_data = data.pop('stage1', {})
        stage2_data = data.pop('stage2', {})
        stage3_data = data.pop('stage3', {})
        
        # Create stage configs
        stage1 = Stage1Config(**stage1_data) if stage1_data else Stage1Config()
        stage2 = Stage2Config(**stage2_data) if stage2_data else Stage2Config()
        stage3 = Stage3Config(**stage3_data) if stage3_data else Stage3Config()
        
        # Create main config
        return cls(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            **data
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        if path is None:
            path = os.path.join(self.experiment_dir, "config.yaml")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        return path
    
    def get_stage_args(self, stage: str, seed: int) -> Dict[str, Any]:
        """
        Generate command-line arguments for a specific stage.
        
        Args:
            stage: One of "stage1", "stage2", "stage3", "baseline", "fullobs"
            seed: Random seed
            
        Returns:
            Dictionary of arguments to pass to main.py
        """
        base_args = {
            "env_args.map_name": self.map_name,
            "seed": seed,
            "use_cuda": self.use_cuda,
            "use_wandb": self.use_wandb,
            "save_model": self.save_models,
            "cpu_inference": False,
        }
        
        if stage == "stage1":
            cfg = self.stage1
            return {
                **base_args,
                "mac": "n_mac",
                "recontructer_stage": "stage1",
                "pretrain_only": True,
                "use_graph_reconstruction": True,
                "graph_pretrain_episodes": cfg.pretrain_episodes,
                "graph_pretrain_epochs": cfg.pretrain_epochs,
                "graph_pretrain_batch_size": cfg.pretrain_batch_size,
                "graph_lr": cfg.learning_rate,
                "graph_pretrain_log_interval": cfg.log_interval,
                "graph_pretrain_eval_interval": cfg.eval_interval,
                "graph_pretrain_eval_episodes": cfg.eval_episodes,
                "save_pretrain_buffer": cfg.save_buffer,
            }
        
        elif stage == "stage2":
            cfg = self.stage2
            return {
                **base_args,
                "mac": "n_mac",
                "recontructer_stage": "stage2",
                "pretrain_only": True,
                "use_graph_reconstruction": True,
                "graph_pretrain_epochs": cfg.pretrain_epochs,
                "graph_pretrain_batch_size": cfg.pretrain_batch_size,
                "graph_lr": cfg.learning_rate,
                "graph_pretrain_log_interval": cfg.log_interval,
                "graph_pretrain_eval_interval": cfg.eval_interval,
                "graph_pretrain_eval_episodes": cfg.eval_episodes,
            }
        
        elif stage == "stage3":
            cfg = self.stage3
            args = {
                **base_args,
                "recontructer_stage": "stage3",
                "pretrain_only": False,
                "use_graph_reconstruction": True,
                "t_max": cfg.t_max,
                "test_interval": cfg.test_interval,
                "test_nepisode": cfg.test_nepisode,
                "save_model_interval": cfg.save_model_interval,
                "log_interval": cfg.log_interval,
            }
            if cfg.use_gnn:
                args["mac"] = "gnn_graph_mac"
                args["agent"] = "gnn_rnn"
                args["gnn_layer_num"] = cfg.gnn_layer_num
            else:
                args["mac"] = "n_graph_mac"
                args["agent"] = "n_rnn"
            return args
        
        else:
            raise ValueError(f"Unknown stage: {stage}. Valid stages: stage1, stage2, stage3")
    
    def get_experiment_subdir(self, stage: str, seed: int) -> str:
        """Get the subdirectory for a specific stage and seed."""
        if seed != self.seeds[0] or len(self.seeds) > 1:
            return os.path.join(self.experiment_dir, f"seed_{seed}", stage)
        return os.path.join(self.experiment_dir, stage)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.map_name:
            errors.append("map_name is required")
        
        if not self.seeds:
            errors.append("At least one seed is required")
        
        if self.stage2.enabled and not self.stage1.enabled and not self.stage2.reuse_stage1_buffer:
            errors.append("Stage 2 requires Stage 1 to be enabled or an existing buffer")
        
        if self.stage3.enabled and not (self.stage1.enabled and self.stage2.enabled):
            errors.append("Stage 3 requires both Stage 1 and Stage 2")
        
        return errors


def create_default_config(map_name: str, name: Optional[str] = None) -> ExperimentConfig:
    """Create a default experiment configuration for a map."""
    if name is None:
        name = f"omagd_{map_name}"
    
    return ExperimentConfig(
        name=name,
        map_name=map_name,
        seeds=[1],
        stage1=Stage1Config(),
        stage2=Stage2Config(),
        stage3=Stage3Config()
    )


if __name__ == "__main__":
    # Test configuration
    config = create_default_config("8m_vs_9m")
    print("Default config:")
    print(yaml.dump(config.to_dict(), default_flow_style=False))
    
    # Validate
    errors = config.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid!")
    
    # Get stage args
    print("\nStage 1 args:")
    print(config.get_stage_args("stage1", seed=1))
