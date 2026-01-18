"""Distributed training utilities for multi-GPU pretrain."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """
    Initialize distributed training environment.
    
    Returns:
        tuple: (is_distributed, rank, world_size, local_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Not in distributed mode
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return True, rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor, average=True):
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: tensor to reduce
        average: if True, average the result; otherwise sum
        
    Returns:
        reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


def broadcast_tensor(tensor, src=0):
    """
    Broadcast tensor from src rank to all other ranks.
    
    Args:
        tensor: tensor to broadcast
        src: source rank
        
    Returns:
        broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_object(obj, src=0):
    """
    Broadcast a Python object from src rank to all other ranks.
    
    Args:
        obj: Python object to broadcast (only used on src rank)
        src: source rank
        
    Returns:
        broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def wrap_model_ddp(model, device_id, find_unused_parameters=False):
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: model to wrap
        device_id: local GPU device id
        find_unused_parameters: whether to find unused parameters
        
    Returns:
        DDP wrapped model
    """
    return DDP(
        model, 
        device_ids=[device_id], 
        output_device=device_id,
        find_unused_parameters=find_unused_parameters
    )


def unwrap_model(model):
    """
    Unwrap DDP model to get the underlying module.
    
    Args:
        model: possibly DDP-wrapped model
        
    Returns:
        underlying model
    """
    if hasattr(model, 'module'):
        return model.module
    return model
