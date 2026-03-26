'''
Use StrokeNet to remove all walking periods from the data, and then calculate the energy
expenditure of the remaining periods using the energy column in the data. Use the average
probability of wrist predictions to determine walking. This is for new unlabelled data.
'''

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.StrokeNet.strokenet_utils import (
    GAP_THRESHOLD,
    STEP_SIZE,
    WINDOW_SIZE,
    load_finetuned_model,
    _column_by_name_or_index,
    get_discontinuity_times,
    run_inference,
    WEIGHTS_PATH,
    CONF_THRESH,
)
from config.paths import (
    PLOTS_DIR,
    RESULTS_DIR,
    QSENSE_CLINIC,
)


def load_qsense_file(filepath, folder_name):
    """Load one QSense wrist file and return sample-level times, acc, labels, and activities."""
    df = pd.read_csv(filepath, sep=None, engine='python').reset_index(drop=True)

    # Parse timestamp from first two columns (Date + Time format used in QSense exports).
    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce'
    )
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError('No valid datetime rows')

    running_max = df['datetime'].iloc[0]
    keep = []
    for t in df['datetime']:
        if t < running_max:
            keep.append(False)
        else:
            keep.append(True)
            running_max = t
    df = df[keep].reset_index(drop=True)

    dt = df['datetime'].diff()
    jump_idx = dt[abs(dt) > pd.Timedelta(days=100)].index
    for idx in jump_idx:
        false_gap = dt[idx] - pd.Timedelta(seconds=1 / 50)
        df.loc[idx:, 'datetime'] = df.loc[idx:, 'datetime'] - false_gap
        dt = df['datetime'].diff()

    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)
    df['time_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

    acc_x = _column_by_name_or_index(df, ['ax', 'accX', 'AccX'], 5)
    acc_y = _column_by_name_or_index(df, ['ay', 'accY', 'AccY'], 6)
    acc_z = _column_by_name_or_index(df, ['az', 'accZ', 'AccZ'], 7)
    energy = _column_by_name_or_index(df, ['energy', 'Energy'], 12) 
    if acc_x is None or acc_y is None or acc_z is None:
        raise ValueError('Missing accelerometer columns in QSense file')

    acc = np.column_stack([
        pd.to_numeric(acc_x, errors='coerce').values,
        pd.to_numeric(acc_y, errors='coerce').values,
        pd.to_numeric(acc_z, errors='coerce').values,
    ])

    times = df['time_sec'].values.astype(float)
    valid = np.isfinite(times) & np.isfinite(acc).all(axis=1)
    times = times[valid]
    acc = acc[valid]
    energy = energy[valid]
    start_dt = df['datetime'].iloc[0]
    return times, acc, energy, start_dt

def extract_windows_with_gaps(times, acc_data, energy):
    dt      = np.diff(times)
    gap_idx = np.where(dt > GAP_THRESHOLD)[0] + 1
    bounds  = np.concatenate([[0], gap_idx, [len(times)]])

    wins_acc, energies, win_times= [], [], []

    for k in range(len(bounds) - 1):
        seg_start = bounds[k]
        seg_end   = bounds[k + 1]
        if (seg_end - seg_start) < WINDOW_SIZE:
            continue
        for i in range(seg_start + WINDOW_SIZE, seg_end, STEP_SIZE):
            win_acc     = acc_data[i - WINDOW_SIZE:i]
            win_energy  = energy[i - WINDOW_SIZE:i]


            wins_acc.append(win_acc.T)
            energies.append(float(np.mean(win_energy)))  #USE MEAN OR SUM?
            win_times.append(times[i - 1])

    if len(wins_acc) == 0:
        return None, None, None, None
    return (np.array(wins_acc, dtype=np.float32),
            np.array(energies, dtype=np.float32),
            np.array(win_times))


def run_subject(folder, folder_path, model, device, dataset_name):
    """Process one subject: load, infer, average, compute ratio. Returns result dict or None."""
    wrist_files = {
        'right': 's1_1RW.txt',
        'left':  's2_2LW.txt',
    }

    prelim = {}
    for wrist, fname in wrist_files.items():
        path = os.path.join(folder_path, fname)
        if not os.path.exists(path):
            continue
        try:
            times, acc, energy, start_dt = load_qsense_file(path, folder)
            discontinuity_times          = get_discontinuity_times(times)
            wins_acc, energies, win_times = extract_windows_with_gaps(times, acc, energy)
            if wins_acc is None:
                print(f"  Skipping {folder}/{fname}: no valid windows")
                continue
            probs          = run_inference(model, wins_acc, device)
            real_win_times = start_dt + pd.to_timedelta(win_times, unit='s')
            prelim[wrist]  = {
                'probs':               probs,
                'win_times':           win_times,
                'real_win_times':      real_win_times,
                'discontinuity_times': discontinuity_times,
                'energy':              energies,
                'start_dt':            start_dt,
            }
        except Exception as e:
            print(f"  Error in {folder}/{fname}: {e}")

    R = prelim.get('right')
    L = prelim.get('left')

    if R is None and L is None:
        print(f"  No valid wrist data for {folder}, skipping.")
        return None

    # ── Average probabilities ──────────────────────────────────────────────
    if R is not None and L is not None:
        df_merge = pd.merge_asof(
            pd.DataFrame({'time': R['win_times'], 'prob_R': R['probs']}).sort_values('time'),
            pd.DataFrame({'time': L['win_times'], 'prob_L': L['probs']}).sort_values('time'),
            on='time', direction='nearest', tolerance=0.05,
        )
        df_merge['avg_prob'] = df_merge[['prob_R', 'prob_L']].mean(axis=1, skipna=True)
        avg_probs  = df_merge['avg_prob'].values
        ref        = R if len(R['win_times']) >= len(L['win_times']) else L
        ref_times  = df_merge['time'].values
        real_times = ref['start_dt'] + pd.to_timedelta(ref_times, unit='s')
        start_dt   = ref['start_dt']
    elif R is not None:
        avg_probs, ref_times, real_times, start_dt = (
            R['probs'], R['win_times'], R['real_win_times'], R['start_dt'])
    else:
        avg_probs, ref_times, real_times, start_dt = (
            L['probs'], L['win_times'], L['real_win_times'], L['start_dt'])

    y_pred = (avg_probs > CONF_THRESH).astype(int)

    # ── Energy ratio (right / left) ────────────────────────────────────────
    ratio_df = None
    if R is not None and L is not None:
        df_R = pd.DataFrame({'real_time': R['real_win_times'], 'energy_R': R['energy']})
        df_L = pd.DataFrame({'real_time': L['real_win_times'], 'energy_L': L['energy']})
        ratio_df = pd.merge_asof(
            df_R.sort_values('real_time'),
            df_L.sort_values('real_time'),
            on='real_time', direction='nearest',
            tolerance=pd.Timedelta(seconds=0.6),
        )
        ratio_df.loc[ratio_df['energy_L'].abs() < 1e-3, 'energy_L'] = np.nan
        ratio_df['ratio']       = ratio_df['energy_R'] / ratio_df['energy_L']
        ratio_df['log2_ratio']  = np.log2(ratio_df['ratio'].replace([np.inf, -np.inf], np.nan))

        # Insert NaNs at discontinuities from either wrist to break the ratio line
        disc_times = sorted({
            start_dt + pd.to_timedelta(dt, unit='s')
            for dt in np.concatenate([R['discontinuity_times'], L['discontinuity_times']])
        })
        for disc_t in disc_times:
            idx = int(ratio_df['real_time'].searchsorted(disc_t, side='left'))
            if 0 < idx < len(ratio_df):
                ratio_df.loc[idx, ['ratio', 'log2_ratio']] = np.nan

    return {
        'subject':        folder,
        'dataset':        dataset_name,
        'start_dt':       start_dt,
        'win_times':      ref_times,
        'real_win_times': real_times,
        'probs':          avg_probs,
        'y_pred':         y_pred,
        'ratio_df':       ratio_df,   # None if only one wrist
        'right': {
            'energy':              R['energy']              if R else None,
            'real_win_times':      R['real_win_times']      if R else None,
            'discontinuity_times': R['discontinuity_times'] if R else [],
            'win_times':           R['win_times']           if R else None,
        },
        'left': {
            'energy':              L['energy']              if L else None,
            'real_win_times':      L['real_win_times']      if L else None,
            'discontinuity_times': L['discontinuity_times'] if L else [],
            'win_times':           L['win_times']           if L else None,
        },
    }


def qsense_energy(model, device, dataset_path):
    """Thin loop — runs each subject and collects results."""
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    results = []
    if not os.path.isdir(dataset_path):
        print(f"Path not found: {dataset_path}")
        return results
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        result = run_subject(folder, folder_path, model, device, dataset_name)
        if result is not None:
            results.append(result)
    return results

def plot_energy_results(results, output_dir):
    os.makedirs(os.path.join(output_dir, "Energy"), exist_ok=True)
    from matplotlib.patches import Patch

    for r in results:
        has_ratio = r['ratio_df'] is not None
        n_rows = 2 if has_ratio else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]  # always index as a list
        ax0 = axes[0]

        # Energy lines
        colours = {'right': 'steelblue', 'left': 'tomato'}
        energy_handles = []

        for wrist, colour in colours.items():
            w = r[wrist]
            if w['energy'] is None:
                continue

            df_w = pd.DataFrame({
                'time':   w['real_win_times'],
                'energy': w['energy'].astype(float),
            })
            gap_rows = [
                {'time': r['start_dt'] + pd.to_timedelta(dt, unit='s'), 'energy': np.nan}
                for dt in w['discontinuity_times']
            ]
            if gap_rows:
                df_w = (
                    pd.concat([df_w, pd.DataFrame(gap_rows)], ignore_index=True)
                    .sort_values('time')
                    .reset_index(drop=True)
                )
            ax0.plot(df_w['time'], df_w['energy'], linewidth=0.8, color=colour)
            energy_handles.append(
                plt.Line2D([0], [0], color=colour, linewidth=1.5,
                           label=f'Energy ({wrist.capitalize()})')
            )

        # Walking shading 
        ref_times  = r['real_win_times']
        in_walk    = False
        walk_start = None
        walk_spans = []  # collect spans first, apply after all axes exist

        for t, pred in zip(ref_times, r['y_pred']):
            if pred == 1 and not in_walk:
                in_walk    = True
                walk_start = t
            elif pred != 1 and in_walk:
                walk_spans.append((walk_start, t))
                in_walk = False
        if in_walk:
            walk_spans.append((walk_start, ref_times[-1]))

        # Ratio subplot
        if has_ratio:
            ax1 = axes[1]
            ax1.plot(r['ratio_df']['real_time'], r['ratio_df']['log2_ratio'],
                     linewidth=0.8, color='purple')
            ax1.axhline(0.0, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)
            ax1.set_title("log₂(Right / Left) Energy Ratio")
            ax1.set_ylabel("log₂(Energy Ratio)")
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))

        # Apply spans to all axes now that they all exist
        shade_axes = [ax0] + ([ax1] if has_ratio else [])
        for span_start, span_end in walk_spans:
            for ax in shade_axes:
                ax.axvspan(span_start, span_end, color='yellow', alpha=0.4, linewidth=0)

        # Legends
        energy_handles.append(Patch(facecolor='yellow', alpha=0.4, label='Walking'))
        ax0.legend(handles=energy_handles, fontsize='small')

        if has_ratio:
            ax1.legend(handles=[
                plt.Line2D([0], [0], color='purple', linewidth=1.5, label='log₂(Right/Left)'),
                plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=0.9, label='Equal energy (0)'),
                Patch(facecolor='yellow', alpha=0.4, label='Walking'),
            ], fontsize='small')

        axes[-1].set_xlabel("Time")
        fig.autofmt_xdate(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "Energy", f"{r['subject']}_energy.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)
    results = qsense_energy(model, device, QSENSE_CLINIC)

    energy_xlsx = os.path.join(RESULTS_DIR, 'strokenet_energy_results.xlsx')
    with pd.ExcelWriter(energy_xlsx, engine='openpyxl') as writer:
        for r in results:
            for wrist in ('right', 'left'):
                w = r[wrist]
                if w['energy'] is None:
                    continue

                wrist_letter = 'R' if wrist == 'right' else 'L'
                sheet_name   = f"{r['subject']}_{wrist_letter}"[:31]

                df_sheet = pd.DataFrame({
                    'real_timestamp':  w['real_win_times'],
                    'window_time_sec': w['win_times'],
                    'energy':          w['energy'],
                })

                # Align probabilities to wrist timestamps
                df_probs = pd.DataFrame({
                    'time': r['win_times'],
                    'probability': r['probs'],
                    'prediction': r['y_pred'],
                })

                df_sheet = pd.merge(
                    df_sheet,
                    df_probs,
                    left_on='window_time_sec',
                    right_on='time',
                    how='left'
                ).drop(columns='time')
                df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  Written sheet: {sheet_name} ({len(df_sheet)} windows)")

    print(f"Saved per wrist energy results: {energy_xlsx}")
    plot_energy_results(results, PLOTS_DIR)


if __name__ == "__main__":
    main()