"""
GPU Usage Monitor for Experiment Pipeline

Tracks GPU memory and utilization during each training stage.
"""

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GPUSnapshot:
    """Single snapshot of GPU state."""
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_utilization: float  # GPU compute utilization %
    temperature: Optional[float] = None


@dataclass
class GPUStageStats:
    """Aggregated GPU statistics for a training stage."""
    stage_name: str
    gpu_id: int
    start_time: float
    end_time: float = 0.0
    duration_sec: float = 0.0
    
    # Memory stats (MB)
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    memory_total_mb: float = 0.0
    
    # Utilization stats (%)
    utilization_peak: float = 0.0
    utilization_avg: float = 0.0
    
    # Temperature stats (Celsius)
    temp_peak: Optional[float] = None
    temp_avg: Optional[float] = None
    
    # Raw snapshots for detailed analysis
    snapshots: List[GPUSnapshot] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "gpu_id": self.gpu_id,
            "duration_sec": round(self.duration_sec, 2),
            "memory": {
                "peak_mb": round(self.memory_peak_mb, 2),
                "avg_mb": round(self.memory_avg_mb, 2),
                "total_mb": round(self.memory_total_mb, 2),
                "peak_percent": round(self.memory_peak_mb / max(self.memory_total_mb, 1) * 100, 2)
            },
            "utilization": {
                "peak_percent": round(self.utilization_peak, 2),
                "avg_percent": round(self.utilization_avg, 2)
            },
            "temperature": {
                "peak_celsius": round(self.temp_peak, 1) if self.temp_peak else None,
                "avg_celsius": round(self.temp_avg, 1) if self.temp_avg else None
            }
        }


class GPUMonitor:
    """
    Monitor GPU usage during training.
    
    Usage:
        monitor = GPUMonitor(gpu_id=0)
        
        # Start monitoring for a stage
        monitor.start_stage("stage1")
        
        # ... run training ...
        
        # End monitoring and get stats
        stats = monitor.end_stage()
        print(f"Peak memory: {stats.memory_peak_mb} MB")
    """
    
    def __init__(self, gpu_id: int = 0, sample_interval: float = 1.0):
        """
        Initialize GPU monitor.
        
        Args:
            gpu_id: GPU device ID to monitor
            sample_interval: Sampling interval in seconds
        """
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.current_stage: Optional[str] = None
        self.current_stats: Optional[GPUStageStats] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._handle = None
        
        # Initialize NVML if available
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except Exception as e:
                print(f"[GPUMonitor] Warning: Failed to initialize NVML: {e}")
                self._nvml_initialized = False
    
    def _get_snapshot(self) -> Optional[GPUSnapshot]:
        """Get current GPU state snapshot."""
        if not self._nvml_initialized:
            # Fallback to PyTorch if NVML not available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    memory_used = torch.cuda.memory_allocated(self.gpu_id) / 1024 / 1024
                    memory_total = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024 / 1024
                    return GPUSnapshot(
                        timestamp=time.time(),
                        memory_used_mb=memory_used,
                        memory_total_mb=memory_total,
                        memory_percent=memory_used / memory_total * 100,
                        gpu_utilization=0.0,  # Not available via PyTorch
                        temperature=None
                    )
                except Exception:
                    return None
            return None
        
        try:
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            memory_used_mb = mem_info.used / 1024 / 1024
            memory_total_mb = mem_info.total / 1024 / 1024
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_util = util.gpu
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            
            return GPUSnapshot(
                timestamp=time.time(),
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_percent=memory_used_mb / memory_total_mb * 100,
                gpu_utilization=gpu_util,
                temperature=temp
            )
        except Exception as e:
            print(f"[GPUMonitor] Error getting snapshot: {e}")
            return None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            if self.current_stats is not None:
                snapshot = self._get_snapshot()
                if snapshot:
                    self.current_stats.snapshots.append(snapshot)
            self._stop_event.wait(self.sample_interval)
    
    def start_stage(self, stage_name: str):
        """
        Start monitoring a new stage.
        
        Args:
            stage_name: Name of the stage (e.g., "stage1", "stage2", "stage3")
        """
        if self.current_stage is not None:
            print(f"[GPUMonitor] Warning: Stage '{self.current_stage}' not ended. Ending now.")
            self.end_stage()
        
        self.current_stage = stage_name
        
        # Get initial snapshot for total memory
        initial_snapshot = self._get_snapshot()
        memory_total = initial_snapshot.memory_total_mb if initial_snapshot else 0.0
        
        self.current_stats = GPUStageStats(
            stage_name=stage_name,
            gpu_id=self.gpu_id,
            start_time=time.time(),
            memory_total_mb=memory_total
        )
        
        # Start background monitoring thread
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"[GPUMonitor] Started monitoring stage: {stage_name}")
    
    def end_stage(self) -> Optional[GPUStageStats]:
        """
        End current stage monitoring and compute statistics.
        
        Returns:
            GPUStageStats with aggregated statistics, or None if no stage was active
        """
        if self.current_stats is None:
            return None
        
        # Stop monitoring thread
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        
        # Compute statistics
        stats = self.current_stats
        stats.end_time = time.time()
        stats.duration_sec = stats.end_time - stats.start_time
        
        if stats.snapshots:
            # Memory stats
            memory_values = [s.memory_used_mb for s in stats.snapshots]
            stats.memory_peak_mb = max(memory_values)
            stats.memory_avg_mb = sum(memory_values) / len(memory_values)
            
            # Utilization stats
            util_values = [s.gpu_utilization for s in stats.snapshots]
            stats.utilization_peak = max(util_values)
            stats.utilization_avg = sum(util_values) / len(util_values)
            
            # Temperature stats
            temp_values = [s.temperature for s in stats.snapshots if s.temperature is not None]
            if temp_values:
                stats.temp_peak = max(temp_values)
                stats.temp_avg = sum(temp_values) / len(temp_values)
        
        print(f"[GPUMonitor] Ended stage: {stats.stage_name}")
        print(f"  Duration: {stats.duration_sec:.1f}s")
        print(f"  Memory: peak={stats.memory_peak_mb:.1f}MB, avg={stats.memory_avg_mb:.1f}MB")
        print(f"  Utilization: peak={stats.utilization_peak:.1f}%, avg={stats.utilization_avg:.1f}%")
        
        # Clear current stage
        result = stats
        self.current_stage = None
        self.current_stats = None
        
        return result
    
    def shutdown(self):
        """Clean up resources."""
        if self.current_stage is not None:
            self.end_stage()
        
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.shutdown()


def get_gpu_info() -> Dict:
    """Get basic GPU information."""
    info = {
        "available": False,
        "count": 0,
        "devices": []
    }
    
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            info["available"] = True
            info["count"] = pynvml.nvmlDeviceGetCount()
            
            for i in range(info["count"]):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                info["devices"].append({
                    "id": i,
                    "name": name,
                    "memory_total_mb": mem_info.total / 1024 / 1024
                })
            
            pynvml.nvmlShutdown()
        except Exception as e:
            info["error"] = str(e)
    elif TORCH_AVAILABLE and torch.cuda.is_available():
        info["available"] = True
        info["count"] = torch.cuda.device_count()
        for i in range(info["count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": props.name,
                "memory_total_mb": props.total_memory / 1024 / 1024
            })
    
    return info
