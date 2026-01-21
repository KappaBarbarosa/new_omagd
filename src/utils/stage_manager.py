"""
Stage Manager for Multi-Stage Training Pipeline

Manages dependencies between training stages and model path resolution.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class StageStatus(Enum):
    """Status of a training stage."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageInfo:
    """Information about a completed stage."""
    stage_name: str
    status: StageStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_sec: float = 0.0
    
    # Paths
    model_path: str = ""
    buffer_path: str = ""
    metrics_path: str = ""
    
    # Key metrics (stage-specific)
    metrics: Dict = field(default_factory=dict)
    
    # GPU stats
    gpu_stats: Dict = field(default_factory=dict)
    
    # Error info (if failed)
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": self.duration_sec,
            "model_path": self.model_path,
            "buffer_path": self.buffer_path,
            "metrics_path": self.metrics_path,
            "metrics": self.metrics,
            "gpu_stats": self.gpu_stats,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StageInfo":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = StageStatus(data.get("status", "not_started"))
        return cls(**data)


class StageManager:
    """
    Manages multi-stage training pipeline.
    
    Responsibilities:
    - Track stage completion status
    - Resolve model paths between stages
    - Handle buffer sharing between Stage 1 and Stage 2
    - Support resume from interrupted training
    
    Usage:
        manager = StageManager(experiment_dir="/path/to/experiment")
        
        # Check what needs to run
        if manager.should_run("stage1"):
            # Run stage 1
            manager.start_stage("stage1")
            # ... training ...
            manager.complete_stage("stage1", model_path="...", metrics={...})
        
        # Get paths for stage 2
        tokenizer_path = manager.get_model_path("stage1")
    """
    
    STATE_FILE = "pipeline_state.json"
    
    def __init__(self, experiment_dir: str):
        """
        Initialize stage manager.
        
        Args:
            experiment_dir: Root directory for this experiment
        """
        self.experiment_dir = experiment_dir
        self.state_file = os.path.join(experiment_dir, self.STATE_FILE)
        self.stages: Dict[str, StageInfo] = {}
        
        # Load existing state if available
        self._load_state()
    
    def _load_state(self):
        """Load pipeline state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                for stage_name, stage_data in data.get("stages", {}).items():
                    self.stages[stage_name] = StageInfo.from_dict(stage_data)
                print(f"[StageManager] Loaded state from {self.state_file}")
            except Exception as e:
                print(f"[StageManager] Warning: Failed to load state: {e}")
    
    def _save_state(self):
        """Save pipeline state to disk."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        data = {
            "experiment_dir": self.experiment_dir,
            "last_updated": datetime.now().isoformat(),
            "stages": {name: info.to_dict() for name, info in self.stages.items()}
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_stage_status(self, stage_name: str) -> StageStatus:
        """Get the status of a stage."""
        if stage_name not in self.stages:
            return StageStatus.NOT_STARTED
        return self.stages[stage_name].status
    
    def should_run(self, stage_name: str, force: bool = False) -> bool:
        """
        Check if a stage should be run.
        
        Args:
            stage_name: Name of the stage
            force: If True, will return True even if completed
            
        Returns:
            True if stage should be run
        """
        if force:
            return True
        
        status = self.get_stage_status(stage_name)
        return status in [StageStatus.NOT_STARTED, StageStatus.FAILED]
    
    def check_dependencies(self, stage_name: str) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies for a stage are satisfied.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            Tuple of (dependencies_met, missing_dependencies)
        """
        dependencies = {
            "stage1": [],
            "stage2": ["stage1"],
            "stage3": ["stage1", "stage2"],
        }
        
        required = dependencies.get(stage_name, [])
        missing = []
        
        for dep in required:
            status = self.get_stage_status(dep)
            if status != StageStatus.COMPLETED:
                missing.append(dep)
        
        return len(missing) == 0, missing
    
    def start_stage(self, stage_name: str):
        """
        Mark a stage as started.
        
        Args:
            stage_name: Name of the stage
        """
        self.stages[stage_name] = StageInfo(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=datetime.now().isoformat()
        )
        self._save_state()
        print(f"[StageManager] Started stage: {stage_name}")
    
    def complete_stage(
        self, 
        stage_name: str, 
        model_path: str = "",
        buffer_path: str = "",
        metrics: Optional[Dict] = None,
        gpu_stats: Optional[Dict] = None
    ):
        """
        Mark a stage as completed.
        
        Args:
            stage_name: Name of the stage
            model_path: Path to saved model
            buffer_path: Path to saved buffer (if any)
            metrics: Training metrics
            gpu_stats: GPU usage statistics
        """
        if stage_name not in self.stages:
            self.stages[stage_name] = StageInfo(stage_name=stage_name, status=StageStatus.COMPLETED)
        
        info = self.stages[stage_name]
        info.status = StageStatus.COMPLETED
        info.end_time = datetime.now().isoformat()
        info.model_path = model_path
        info.buffer_path = buffer_path
        info.metrics = metrics or {}
        info.gpu_stats = gpu_stats or {}
        
        # Calculate duration
        if info.start_time:
            start = datetime.fromisoformat(info.start_time)
            end = datetime.fromisoformat(info.end_time)
            info.duration_sec = (end - start).total_seconds()
        
        # Save metrics to file
        if metrics:
            metrics_path = os.path.join(self.experiment_dir, stage_name, "metrics.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            info.metrics_path = metrics_path
        
        self._save_state()
        print(f"[StageManager] Completed stage: {stage_name}")
        print(f"  Duration: {info.duration_sec:.1f}s")
        if model_path:
            print(f"  Model: {model_path}")
    
    def fail_stage(self, stage_name: str, error_message: str):
        """
        Mark a stage as failed.
        
        Args:
            stage_name: Name of the stage
            error_message: Error message
        """
        if stage_name not in self.stages:
            self.stages[stage_name] = StageInfo(stage_name=stage_name, status=StageStatus.FAILED)
        
        info = self.stages[stage_name]
        info.status = StageStatus.FAILED
        info.end_time = datetime.now().isoformat()
        info.error_message = error_message
        
        if info.start_time:
            start = datetime.fromisoformat(info.start_time)
            end = datetime.fromisoformat(info.end_time)
            info.duration_sec = (end - start).total_seconds()
        
        self._save_state()
        print(f"[StageManager] Failed stage: {stage_name}")
        print(f"  Error: {error_message}")
    
    def get_model_path(self, stage_name: str) -> Optional[str]:
        """
        Get the model path from a completed stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Path to model directory, or None if not available
        """
        if stage_name not in self.stages:
            return None
        
        info = self.stages[stage_name]
        if info.status != StageStatus.COMPLETED:
            return None
        
        return info.model_path if info.model_path else None
    
    def get_buffer_path(self, stage_name: str = "stage1") -> Optional[str]:
        """
        Get the buffer path from a completed stage.
        
        Args:
            stage_name: Name of the stage (usually stage1)
            
        Returns:
            Path to buffer file, or None if not available
        """
        if stage_name not in self.stages:
            return None
        
        info = self.stages[stage_name]
        if info.status != StageStatus.COMPLETED:
            return None
        
        return info.buffer_path if info.buffer_path else None
    
