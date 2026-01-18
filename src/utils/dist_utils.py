"""
Distributed training utilities for multi-GPU training.
Supports both DataParallel (DP) and DistributedDataParallel (DDP).
"""

import os
import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(rank, world_size, backend='nccl', port='12355'):
    """
    Initialize distributed training environment.
    
    Args:
        rank: Current process rank (GPU index)
        world_size: Total number of processes (GPUs)
        backend: 'nccl' for GPU, 'gloo' for CPU
        port: Port for master process communication
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    th.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_mean(tensor):
    """Average tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def broadcast_dict(data_dict, src=0):
    """Broadcast dictionary from source rank to all processes."""
    if not dist.is_initialized():
        return data_dict
    
    import pickle
    if get_rank() == src:
        data_bytes = pickle.dumps(data_dict)
        size = th.tensor([len(data_bytes)], dtype=th.long, device='cuda')
    else:
        size = th.tensor([0], dtype=th.long, device='cuda')
    
    dist.broadcast(size, src)
    
    if get_rank() == src:
        data_tensor = th.ByteTensor(list(data_bytes)).cuda()
    else:
        data_tensor = th.ByteTensor(size.item()).cuda()
    
    dist.broadcast(data_tensor, src)
    
    if get_rank() != src:
        data_dict = pickle.loads(bytes(data_tensor.cpu().numpy()))
    
    return data_dict


def wrap_model_ddp(model, device_id, find_unused_parameters=True):
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: PyTorch model to wrap
        device_id: CUDA device ID for this process
        find_unused_parameters: Whether to find unused parameters (needed for some models)
    
    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        return model
    
    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters
    )


def unwrap_model(model):
    """Get the underlying model from DDP wrapper."""
    if hasattr(model, 'module'):
        return model.module
    return model


def sync_gradients(model):
    """Manually synchronize gradients across processes (usually not needed with DDP)."""
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= get_world_size()


def gather_scalar(scalar, dst=0):
    """Gather scalar values from all processes to dst."""
    if not dist.is_initialized():
        return [scalar]
    
    tensor = th.tensor([scalar], device='cuda')
    if get_rank() == dst:
        gathered = [th.zeros(1, device='cuda') for _ in range(get_world_size())]
        dist.gather(tensor, gathered, dst=dst)
        return [t.item() for t in gathered]
    else:
        dist.gather(tensor, dst=dst)
        return [scalar]


class DistributedSampler:
    """
    Simple distributed sampler for replay buffer.
    Ensures each GPU gets different samples.
    """
    
    def __init__(self, total_size, rank, world_size, shuffle=True):
        self.total_size = total_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
    
    def get_indices(self, batch_size):
        """
        Get sample indices for this process.
        Each process gets batch_size // world_size samples.
        """
        local_batch_size = batch_size // self.world_size
        
        if self.shuffle:
            g = th.Generator()
            g.manual_seed(self.epoch)
            all_indices = th.randperm(self.total_size, generator=g).tolist()
        else:
            all_indices = list(range(self.total_size))
        
        # Each process gets a different subset
        start_idx = self.rank * local_batch_size
        end_idx = start_idx + local_batch_size
        
        return all_indices[start_idx:end_idx]
    
    def set_epoch(self, epoch):
        """Set epoch for different shuffling each epoch."""
        self.epoch = epoch
