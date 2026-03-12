import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from eldernet_WearGait import (
    DATA_PATH, REPO_NAME, load_weargait_data, detect_sampling_rate,
    resample_to_30hz, prepare_windows_overlapping, create_ground_truth
)


def collect_sequences(data_path, wrists=('right', 'left')):
    csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_path}")

    all_probs = []
    all_engs = []
    all_frqs = []
    all_y = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    for filepath in csv_files:
        for wrist in wrists:
            try:
                df = load_weargait_data(filepath, wrist=wrist)
                fs = detect_sampling_rate(df['time'].values)
                df_30hz = resample_to_30hz(df, fs)
                wins, engs, frqs, acts, tmstps = prepare_windows_overlapping(df_30hz)

                if len(wins) == 0:
                    continue

                with torch.no_grad():
                    logits = model(wins.to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                probs_sm = np.convolve(probs, np.ones(3) / 3, mode='same')
                y_true = create_ground_truth(acts)

                all_probs.append(probs_sm)
                all_engs.append(engs)
                all_frqs.append(frqs)
                all_y.append(y_true)
            except Exception as e:
                print(f"Skipping {os.path.basename(filepath)} {wrist} wrist: {e}")

    if not all_probs:
        raise RuntimeError("No valid sequences collected from dataset")

    # concatenate
    all_probs = np.concatenate(all_probs)
    all_engs = np.concatenate(all_engs)
    all_frqs = np.concatenate(all_frqs)
    all_y = np.concatenate(all_y)

    return all_probs, all_engs, all_frqs, all_y


def grid_search(all_probs, all_engs, all_frqs, all_y,
                confs, energies, max_energies, min_freqs, max_freqs):
    results = []
    total = len(confs) * len(energies) * len(max_energies) * len(min_freqs) * len(max_freqs)
    print(f"Evaluating {total} threshold combinations...")

    cnt = 0
    for conf in confs:
        for eng in energies:
            for maxeng in max_energies:
                for mn in min_freqs:
                    for mx in max_freqs:
                        if mx <= mn or maxeng <= eng:
                            continue
                        y_pred = ((all_probs > conf) & (all_engs > eng) & (all_engs < maxeng) & (all_frqs > mn) & (all_frqs < mx)).astype(int)
                        if np.sum(all_y) == 0:
                            p, r, f1 = 0.0, 0.0, 0.0
                        else:
                            p, r, f1, _ = precision_recall_fscore_support(all_y, y_pred, labels=[1], average='binary', zero_division=0)
                        acc = accuracy_score(all_y, y_pred)
                        results.append({'conf': float(conf), 'energy': float(eng), 'maxenergy': float(maxeng), 'minf': float(mn), 'maxf': float(mx),
                                        'precision': float(p), 'recall': float(r), 'f1': float(f1), 'accuracy': float(acc)})
                        cnt += 1
                        if cnt % 500 == 0:
                            print(f"  evaluated {cnt}/{total}")

    return pd.DataFrame(results)


def parse_args():
    p = argparse.ArgumentParser(description='Hyperparameter grid search for thresholds')
    p.add_argument('--data-path', default=DATA_PATH)
    p.add_argument('--conf-start', type=float, default=0.1)
    p.add_argument('--conf-end', type=float, default=0.9)
    p.add_argument('--conf-steps', type=int, default=30)
    p.add_argument('--energy-start', type=float, default=0.1)
    p.add_argument('--energy-end', type=float, default=1.0)
    p.add_argument('--energy-steps', type=int, default=10)
    p.add_argument('--energy-max-start', type=float, default=0.8)
    p.add_argument('--energy-max-end', type=float, default=2.0)
    p.add_argument('--energy-max-steps', type=int, default=30)
    p.add_argument('--minf-start', type=float, default=0.5)
    p.add_argument('--minf-end', type=float, default=1.5)
    p.add_argument('--minf-steps', type=int, default=11)
    p.add_argument('--maxf-start', type=float, default=1.5)
    p.add_argument('--maxf-end', type=float, default=3.0)
    p.add_argument('--maxf-steps', type=int, default=30)
    p.add_argument('--out', default='threshold_search_results.csv')
    return p.parse_args()


def main():
    args = parse_args()

    confs = np.linspace(args.conf_start, args.conf_end, args.conf_steps)
    energies = np.linspace(args.energy_start, args.energy_end, args.energy_steps)
    max_energies = np.linspace(args.energy_max_start, args.energy_max_end, args.energy_max_steps)
    min_freqs = np.linspace(args.minf_start, args.minf_end, args.minf_steps)
    max_freqs = np.linspace(args.maxf_start, args.maxf_end, args.maxf_steps)

    print("Collecting sequences and model outputs (may take a while)...")
    all_probs, all_engs, all_frqs, all_y = collect_sequences(args.data_path)

    print("Running grid search...")
    df = grid_search(all_probs, all_engs, all_frqs, all_y, confs, energies, max_energies, min_freqs, max_freqs)

    df = df.sort_values('f1', ascending=False).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"Saved results to {args.out}")

    best = df.iloc[0]
    print("\nBest combination by F1:")
    print(best.to_string())


if __name__ == '__main__':
    main()
