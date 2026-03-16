import os
import psutil  
import torch

def get_memory_usage():
    """Total process RAM usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_model_memory(model):
    """Memory occupied by model parameters and buffers."""
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes  = param_bytes + buffer_bytes
    return {
        'parameters_mb': param_bytes  / 1024**2,
        'buffers_mb':    buffer_bytes / 1024**2,
        'total_mb':      total_bytes  / 1024**2,
        'n_params':      sum(p.numel() for p in model.parameters()),
        'n_trainable':   sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def get_gpu_memory():
    """Current GPU memory allocation and peak usage."""
    if not torch.cuda.is_available():
        return None
    return {
        'allocated_mb':  torch.cuda.memory_allocated()  / 1024**2,
        'reserved_mb':   torch.cuda.memory_reserved()   / 1024**2,
        'peak_mb':       torch.cuda.max_memory_allocated() / 1024**2,
        'total_mb':      torch.cuda.get_device_properties(0).total_memory / 1024**2
    }


def print_memory_report(model, label=''):
    """Print a full memory report."""
    m   = get_model_memory(model)
    gpu = get_gpu_memory()
    ram = get_memory_usage() / 1024**2

    print(f"\n=== Memory Report {f'({label}) ' if label else ''}===")
    print(f"  Model parameters:  {m['n_params']:,} total, {m['n_trainable']:,} trainable")
    print(f"  Model size:        {m['total_mb']:.2f} MB  "
          f"(params={m['parameters_mb']:.2f} MB, buffers={m['buffers_mb']:.2f} MB)")
    print(f"  Process RAM:       {ram:.1f} MB")
    if gpu:
        print(f"  GPU allocated:     {gpu['allocated_mb']:.1f} MB")
        print(f"  GPU reserved:      {gpu['reserved_mb']:.1f} MB")
        print(f"  GPU peak:          {gpu['peak_mb']:.1f} MB")
        print(f"  GPU total:         {gpu['total_mb']:.1f} MB")
        print(f"  GPU free:          {gpu['total_mb'] - gpu['reserved_mb']:.1f} MB")
