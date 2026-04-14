'''
StrokeNet Computational Load Profiling
This script loads the pre-trained StrokeNet model and evaluates its computational load in terms of:
1. Number of parameters (total and trainable)
2. Model size on disk (MB)
3. FLOPs / Multiply-Adds per inference (for a single window)
4. Average inference time per window (on CPU and GPU if available)
This analysis helps us understand the efficiency of StrokeNet and its suitability for real-time gait detection on wearable devices. The code is structured to be clean and focused on the computational load evaluation, without any dataset-specific processing or evaluation logic. This keeps the script simple and allows us to easily reuse it for analyzing other models in the future by simply changing the model loading part.
'''

import json
import os
import sys
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import RESULTS_DIR
from models.StrokeNet.strokenet_utils import (
    WEIGHTS_PATH,
    WINDOW_SIZE,
    load_finetuned_model,
)
from utils.comp_load import (
    print_memory_report,
    try_model_complexity,
    benchmark_inference,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Profile StrokeNet compute load.')
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Execution device. auto picks cuda when available else cpu.',
    )
    parser.add_argument('--warmup', type=int, default=30, help='Warmup iterations.')
    parser.add_argument('--iterations', type=int, default=200, help='Timed iterations.')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 16], help='Batch sizes to benchmark.')
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available on this machine.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_profile():
    args = parse_args()
    device = resolve_device(args.device)
    print(f'Profiling StrokeNet on: {device}')

    model = load_finetuned_model(WEIGHTS_PATH).to(device)
    model.eval()

    print_memory_report(model, label='StrokeNet Loaded')

    input_shape = (3, WINDOW_SIZE)
    complexity = try_model_complexity(model, input_shape)
    if complexity is None:
        print('\nCould not compute MACs/FLOPs: install ptflops to enable complexity profiling.')
    else:
        print('\n=== Complexity ===')
        print(f"  MACs per window:   {complexity['macs']}")
        print(f"  Params (ptflops):  {complexity['params']}")

    print('\n=== Latency And Throughput ===')
    benches = []
    for batch_size in args.batch_sizes:
        benches.append(
            benchmark_inference(
                model=model,
                input_shape=input_shape,
                device=device,
                batch_size=batch_size,
                warmup=args.warmup,
                iterations=args.iterations,
                collect_power=True,
            )
        )

    for bench in benches:
        print(f"\nBatch size: {bench['batch_size']}")
        print(f"  Mean latency:      {bench['mean_latency_ms']:.3f} ms")
        print(f"  P50 latency:       {bench['p50_latency_ms']:.3f} ms")
        print(f"  P95 latency:       {bench['p95_latency_ms']:.3f} ms")
        print(f"  Throughput:        {bench['throughput_windows_per_s']:.2f} windows/s")
        if bench['power'] is not None:
            power = bench['power']
            print(f"  Avg power:         {power['avg_power_w']:.2f} W")
            print(f"  Min/Max power:     {power['min_power_w']:.2f} / {power['max_power_w']:.2f} W")
            print(
                f"  Est. energy/infer: {power['estimated_energy_per_inference_j']:.6f} J"
            )
        else:
            print('  Power:             unavailable (no supported GPU power telemetry)')

    out_dir = os.path.join(RESULTS_DIR, 'StrokeNet')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'strokenet_compute_profile_{device.type}.json')

    report = {
        'device': str(device),
        'input_shape': input_shape,
        'weights_path': WEIGHTS_PATH,
        'complexity': complexity,
        'benchmarks': benches,
        'settings': {
            'warmup': args.warmup,
            'iterations': args.iterations,
            'batch_sizes': args.batch_sizes,
            'requested_device': args.device,
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'\nSaved profile report: {out_path}')


if __name__ == '__main__':
    run_profile()
