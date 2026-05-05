# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def init_distributed_group(backend=None):
    """Initialize sequence parallel group.

    No-op if a process group is already initialized (which is the normal
    path — generate_rolling.py calls init_process_group before this).
    """
    if not dist.is_initialized():
        if backend is None:
            available = list(dist.Backend.backend_list)
            for candidate in ("neuron", "nccl", "gloo"):
                if candidate in available:
                    backend = candidate
                    break
        dist.init_process_group(backend=backend)


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
