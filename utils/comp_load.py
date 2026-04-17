'''
Comprehensive model loading and profiling utilities for PyTorch models, including memory usage, GPU power, and inference benchmarking.
Designed for use with StrokeNet but can be reused for other models as well. Provides a standardized way to report model complexity, latency, throughput, and power consumption, which is crucial for understanding the trade-offs of deploying models on edge devices.
'''


import os
import time
import statistics
import subprocess
import importlib
import psutil  
import torch

def get_memory_usage():
    """Total process RAM usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_model_disk_size_mb(model):
    """Serialized model state_dict size on disk (MB)."""
    temp_path = '_temp_model_profile.pth'
    torch.save(model.state_dict(), temp_path)
    try:
        return os.path.getsize(temp_path) / (1024 ** 2)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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


def _get_gpu_power_pynvml():
    try:
        pynvml = importlib.import_module('pynvml')
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
        return {
            'power_w': draw_mw / 1000.0,
            'power_limit_w': limit_mw / 1000.0,
            'source': 'pynvml',
        }
    except Exception:
        return None


def _get_gpu_power_nvidia_smi():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=power.draw,power.limit',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        power_s, limit_s = [x.strip() for x in first_line.split(',')[:2]]
        return {
            'power_w': float(power_s),
            'power_limit_w': float(limit_s),
            'source': 'nvidia-smi',
        }
    except Exception:
        return None


def get_gpu_power():
    """Current GPU power draw. Returns None if unavailable."""
    if not torch.cuda.is_available():
        return None

    info = _get_gpu_power_pynvml()
    if info is not None:
        return info
    return _get_gpu_power_nvidia_smi()


def try_model_complexity(model, input_shape):
    """Attempt MAC/FLOP estimate using ptflops. Returns None if unavailable."""
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    return {
        'macs': macs,
        'params': params,
    }


def benchmark_inference(
    model,
    input_shape,
    device,
    batch_size=1,
    warmup=20,
    iterations=100,
    collect_power=True,
    power_sample_interval_s=0.2,
):
    """Benchmark model inference latency/throughput and optional GPU power."""
    model.eval()
    x = torch.randn(batch_size, *input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies_s = []
    power_samples_w = []
    last_power_sample_t = 0.0

    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_s.append(t1 - t0)

            if collect_power:
                now = time.perf_counter()
                if (now - last_power_sample_t) >= power_sample_interval_s:
                    p = get_gpu_power()
                    if p is not None and p.get('power_w') is not None:
                        power_samples_w.append(float(p['power_w']))
                    last_power_sample_t = now

    mean_latency_s = statistics.mean(latencies_s) if latencies_s else 0.0
    p50_latency_s = statistics.median(latencies_s) if latencies_s else 0.0
    p95_latency_s = (
        float(torch.quantile(torch.tensor(latencies_s), 0.95).item())
        if latencies_s
        else 0.0
    )
    throughput_win_s = (batch_size / mean_latency_s) if mean_latency_s > 0 else 0.0

    result = {
        'batch_size': batch_size,
        'iterations': iterations,
        'mean_latency_ms': mean_latency_s * 1000.0,
        'p50_latency_ms': p50_latency_s * 1000.0,
        'p95_latency_ms': p95_latency_s * 1000.0,
        'throughput_windows_per_s': throughput_win_s,
    }

    if power_samples_w:
        avg_power = statistics.mean(power_samples_w)
        result['power'] = {
            'samples': len(power_samples_w),
            'avg_power_w': avg_power,
            'min_power_w': min(power_samples_w),
            'max_power_w': max(power_samples_w),
            'estimated_energy_per_inference_j': avg_power * mean_latency_s,
        }
    else:
        result['power'] = None

    return result


def print_memory_report(model, label=''):
    """Print a full memory report."""
    m   = get_model_memory(model)
    gpu = get_gpu_memory()
    ram = get_memory_usage() / 1024**2

    print(f"\n=== Memory Report {f'({label}) ' if label else ''}===")
    print(f"  Model parameters:  {m['n_params']:,} total, {m['n_trainable']:,} trainable")
    print(f"  Model size:        {m['total_mb']:.2f} MB  "
          f"(params={m['parameters_mb']:.2f} MB, buffers={m['buffers_mb']:.2f} MB)")
    print(f"  State dict size:   {get_model_disk_size_mb(model):.2f} MB")
    print(f"  Process RAM:       {ram:.1f} MB")
    if gpu:
        print(f"  GPU allocated:     {gpu['allocated_mb']:.1f} MB")
        print(f"  GPU reserved:      {gpu['reserved_mb']:.1f} MB")
        print(f"  GPU peak:          {gpu['peak_mb']:.1f} MB")
        print(f"  GPU total:         {gpu['total_mb']:.1f} MB")
        print(f"  GPU free:          {gpu['total_mb'] - gpu['reserved_mb']:.1f} MB")

    power = get_gpu_power()
    if power is not None:
        print(f"  GPU power:         {power['power_w']:.1f} W / {power['power_limit_w']:.1f} W "
              f"({power['source']})")
