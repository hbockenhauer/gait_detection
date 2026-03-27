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
from matplotlib.patches import Patch

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
    QSENSE_TEST,
)

MISSING_PERIOD = 10  # seconds — threshold for reporting missing data periods

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

def _align_y_pred_to_wrist(w, r):
    """
    Align the subject-level averaged prediction to a single wrist's timestamps.
    Falls back to the wrist's own prob if no averaged prediction is available.
    Returns an int array (0/1) of the same length as w['energy'].
    """
    if w['energy'] is None:
        return None

    df_w = pd.DataFrame({
        'real_timestamp': pd.to_datetime(w['real_win_times']),
    })
    df_pred = pd.DataFrame({
        'real_timestamp': pd.to_datetime(r['real_win_times']),
        'prob_avg':       r['probs'],
    }).sort_values('real_timestamp')

    df_w = pd.merge_asof(
        df_w.sort_values('real_timestamp'),
        df_pred,
        on='real_timestamp', direction='nearest',
        tolerance=pd.Timedelta(seconds=2),
    )

    # Fall back to wrist's own prob where averaged prob is missing
    missing = df_w['prob_avg'].isna()
    if missing.any() and w['probs'] is not None:
        df_w.loc[missing, 'prob_avg'] = w['probs'][missing.values]

    return (df_w['prob_avg'] > CONF_THRESH).astype(int).values


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

    if R is not None:
        R['y_pred'] = _align_y_pred_to_wrist(R, {
            'real_win_times': real_times, 'probs': avg_probs})
    if L is not None:
        L['y_pred'] = _align_y_pred_to_wrist(L, {
            'real_win_times': real_times, 'probs': avg_probs})
        
    hub_times = load_hub_times(folder_path)

    return {
        'subject':        folder,
        'dataset':        dataset_name,
        'start_dt':       start_dt,
        'win_times':      ref_times,
        'real_win_times': real_times,
        'hub_times':      hub_times,
        'probs':          avg_probs,
        'y_pred':         y_pred,
        'ratio_df':       ratio_df,   # None if only one wrist
        'right': {
            'energy':              R['energy']              if R else None,
            'real_win_times':      R['real_win_times']      if R else None,
            'discontinuity_times': R['discontinuity_times'] if R else [],
            'win_times':           R['win_times']           if R else None,
            'probs':               R['probs']               if R else None,
            'y_pred':             R['y_pred']             if R else None,
        },
        'left': {
            'energy':              L['energy']              if L else None,
            'real_win_times':      L['real_win_times']      if L else None,
            'discontinuity_times': L['discontinuity_times'] if L else [],
            'win_times':           L['win_times']           if L else None,
            'probs':               L['probs']               if L else None,
            'y_pred':             L['y_pred']             if L else None,
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

# Plot energy over time with walking periods shaded with R/L ratio subplot 
def plot_energy_results_line(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

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

        plot_path = os.path.join(output_dir, f"{r['subject']}_energy.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved energy line plot: {plot_path}")

def plot_energy_results_bar(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for r in results:
        # ── Build per-wrist hourly DataFrames ─────────────────────────────
        wrist_hourly = {}
        for wrist in ('right', 'left'):
            w = r[wrist]
            if w['energy'] is None:
                continue

            df_w = pd.DataFrame({
                'time':   pd.to_datetime(w['real_win_times']),
                'energy': w['energy'].astype(float),
            })

            # Use pre-aligned per-wrist y_pred — no merge needed
            df_w['y_pred']    = w['y_pred'].astype(float)
            df_w['hour_bin']  = df_w['time'].dt.floor('h')
            df_nonwalk        = df_w[df_w['y_pred'] != 1]
            hourly            = df_nonwalk.groupby('hour_bin')['energy'].sum().rename(wrist)
            wrist_hourly[wrist] = hourly

        if not wrist_hourly:
            print(f"  No energy data for {r['subject']}, skipping bar plot.")
            continue

        # ── Combine into one DataFrame ────────────────────────────────────
        df_all = pd.concat(wrist_hourly.values(), axis=1).sort_index()
        df_all.columns = [c for c in wrist_hourly]          # 'right', 'left', or both

        if 'right' in df_all and 'left' in df_all:
            df_all['total'] = df_all['right'].fillna(0) + df_all['left'].fillna(0)
            df_all['log2_ratio'] = np.log2(
                (df_all['right'] / df_all['left']).replace([np.inf, -np.inf], np.nan)
            )
        elif 'right' in df_all:
            df_all['total'] = df_all['right']
            df_all['log2_ratio'] = np.nan
        else:
            df_all['total'] = df_all['left']
            df_all['log2_ratio'] = np.nan

        hours     = df_all.index                 # DatetimeIndex of clock hours
        n_hours   = len(hours)
        x         = np.arange(n_hours)
        has_both  = 'right' in df_all.columns and 'left' in df_all.columns
        has_ratio = has_both and df_all['log2_ratio'].notna().any()

        # ── Plot ──────────────────────────────────────────────────────────
        n_rows = 2 if has_ratio else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(max(14, n_hours * 0.6), 5 * n_rows))
        if n_rows == 1:
            axes = [axes]
        ax0 = axes[0]

        bar_w   = 0.25
        offsets = {'right': -bar_w, 'left': 0.0, 'total': bar_w}
        colours = {'right': 'steelblue', 'left': 'tomato', 'total': 'mediumseagreen'}
        handles = []

        for metric in ('right', 'left', 'total'):
            if metric not in df_all.columns and metric != 'total':
                continue
            if metric == 'total' and 'total' not in df_all.columns:
                continue
            vals = df_all[metric].values.astype(float)
            ax0.bar(x + offsets[metric], vals, width=bar_w,
                    color=colours[metric], label=metric.capitalize(), alpha=0.85)
            handles.append(Patch(facecolor=colours[metric], alpha=0.85,
                                 label=metric.capitalize()))

        ax0.set_title(f"{r['subject']} — Non-walking Energy per Hour")
        ax0.set_ylabel("Energy (sum)")
        ax0.set_xticks(x)
        ax0.set_xticklabels(
            [h.strftime('%d/%m %H:%M') for h in hours],
            rotation=45, ha='right', fontsize=8,
        )
        ax0.legend(handles=handles, fontsize='small')

        # ── Ratio subplot ─────────────────────────────────────────────────
        if has_ratio:
            ax1 = axes[1]
            ratio_vals = df_all['log2_ratio'].values.astype(float)
            bar_colors = ['purple' if not np.isnan(v) else 'lightgray' for v in ratio_vals]
            ax1.bar(x, ratio_vals, width=0.5, color=bar_colors, alpha=0.85)
            ax1.axhline(0.0, color='gray', linestyle='--', linewidth=0.9, alpha=0.8)
            ax1.set_title("Average log₂(Right / Left) Energy Ratio per Hour")
            ax1.set_ylabel("log₂(Energy Ratio)")
            ax1.set_xticks(x)
            ax1.set_xticklabels(
                [h.strftime('%d/%m %H:%M') for h in hours],
                rotation=45, ha='right', fontsize=8,
            )
            ax1.legend(handles=[
                Patch(facecolor='purple', alpha=0.85, label='log₂(Right/Left)'),
                plt.Line2D([0], [0], color='gray', linestyle='--',
                           linewidth=0.9, label='Equal energy (0)'),
            ], fontsize='small')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{r['subject']}_energy_bar.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved energy bar plot: {plot_path}")

def load_hub_times(folder_path):
    """Load timestamps from the hub file (s0_Hub.txt), which is always correct."""
    hub_path = os.path.join(folder_path, 's0_Hub.txt')
    if not os.path.exists(hub_path):
        return None
    try:
        df = pd.read_csv(hub_path, sep=None, engine='python').reset_index(drop=True)
        df['datetime'] = pd.to_datetime(
            df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
            errors='coerce'
        )
        df = df.dropna(subset=['datetime']).reset_index(drop=True)
        return pd.to_datetime(df['datetime'].values)
    except Exception as e:
        print(f"  Warning: could not load hub file from {folder_path}: {e}")
        return None


def find_missing_periods(wrist_real_win_times, hub_times, win_duration_sec, min_gap_sec=10):
    if wrist_real_win_times is None or hub_times is None:
        return []

    # Normalise both to nanosecond precision to avoid dtype mismatch in merge_asof
    wrist_times = pd.to_datetime(wrist_real_win_times).astype('datetime64[ns]')
    hub_series  = pd.to_datetime(hub_times).astype('datetime64[ns]')

    tolerance = pd.Timedelta(seconds=win_duration_sec)

    df_hub   = pd.DataFrame({'hub_time':   hub_series}).sort_values('hub_time')
    df_wrist = pd.DataFrame({'wrist_time': wrist_times}).sort_values('wrist_time')

    df_check = pd.merge_asof(
        df_hub,
        df_wrist,
        left_on='hub_time', right_on='wrist_time',
        direction='nearest',
        tolerance=tolerance,
    )

    df_check['missing'] = df_check['wrist_time'].isna()

    gaps      = []
    in_gap    = False
    gap_start = None

    for _, row in df_check.iterrows():
        if row['missing'] and not in_gap:
            in_gap    = True
            gap_start = row['hub_time']
        elif not row['missing'] and in_gap:
            gap_end     = row['hub_time']
            gap_dur_sec = (gap_end - gap_start).total_seconds()
            if gap_dur_sec >= min_gap_sec:
                gaps.append((gap_start, gap_end, gap_dur_sec))
            in_gap = False

    if in_gap:
        gap_end     = df_check['hub_time'].iloc[-1]
        gap_dur_sec = (gap_end - gap_start).total_seconds()
        if gap_dur_sec >= min_gap_sec:
            gaps.append((gap_start, gap_end, gap_dur_sec))

    return gaps

def save_subject_excel(r, output_dir):
    """Save one Excel file per subject with a per-window sheet and a summary sheet."""
    os.makedirs(output_dir, exist_ok=True)
    R = r['right']
    L = r['left']

    # ── Sheet 1: per-window data ───────────────────────────────────────────
    def wrist_df(w, energy_col, prob_col):
        if w['energy'] is None:
            return None
        return pd.DataFrame({
            'real_timestamp':  pd.to_datetime(w['real_win_times']),
            'window_time_sec': w['win_times'],
            energy_col:        w['energy'].astype(float),
            prob_col:          w['probs'].astype(float),
        }).sort_values('real_timestamp')

    df_R = wrist_df(R, 'energy_R', 'prob_R')
    df_L = wrist_df(L, 'energy_L', 'prob_L')

    if df_R is not None and df_L is not None:
        df_windows = pd.merge_asof(
            df_R,
            df_L[['real_timestamp', 'energy_L', 'prob_L']],
            on='real_timestamp', direction='nearest',
            tolerance=pd.Timedelta(seconds=0.6),
        )
        matched_times  = set(df_windows['real_timestamp'])
        df_L_unmatched = df_L[~df_L['real_timestamp'].isin(matched_times)].copy()
        df_L_unmatched['energy_R'] = np.nan
        df_L_unmatched['prob_R']   = np.nan
        df_windows = (
            pd.concat([df_windows, df_L_unmatched], ignore_index=True)
            .sort_values('real_timestamp')
            .reset_index(drop=True)
        )
    elif df_R is not None:
        df_windows = df_R.copy()
        df_windows['energy_L'] = np.nan
        df_windows['prob_L']   = np.nan
    else:
        df_windows = df_L.copy()
        df_windows['energy_R'] = np.nan
        df_windows['prob_R']   = np.nan


    if R['y_pred'] is not None and R['real_win_times'] is not None:
        df_ypred_R = pd.DataFrame({
            'real_timestamp': pd.to_datetime(R['real_win_times']),
            'y_pred_R':       R['y_pred'],
        })
        df_windows = pd.merge_asof(
            df_windows.sort_values('real_timestamp'),
            df_ypred_R.sort_values('real_timestamp'),
            on='real_timestamp', direction='nearest',
            tolerance=pd.Timedelta(seconds=2),
        )
    else:
        df_windows['y_pred_R'] = np.nan

    if L['y_pred'] is not None and L['real_win_times'] is not None:
        df_ypred_L = pd.DataFrame({
            'real_timestamp': pd.to_datetime(L['real_win_times']),
            'y_pred_L':       L['y_pred'],
        })
        df_windows = pd.merge_asof(
            df_windows.sort_values('real_timestamp'),
            df_ypred_L.sort_values('real_timestamp'),
            on='real_timestamp', direction='nearest',
            tolerance=pd.Timedelta(seconds=2),
        )
    else:
        df_windows['y_pred_L'] = np.nan

    # Average available per-wrist predictions, then threshold
    pred_cols = [c for c in ('y_pred_R', 'y_pred_L') if c in df_windows.columns]
    df_windows['prob_avg']   = df_windows[['prob_R', 'prob_L']].mean(axis=1, skipna=True)
    df_windows['prediction'] = (
        df_windows[pred_cols].mean(axis=1, skipna=True) > 0.5
    ).astype(int)
    df_windows['energy_total'] = (
        df_windows['energy_R'].fillna(0) + df_windows['energy_L'].fillna(0)
    ).where(df_windows['energy_R'].notna() | df_windows['energy_L'].notna())

    df_windows['ratio_R_L'] = np.where(
        df_windows['energy_L'].abs() > 1e-3,
        df_windows['energy_R'] / df_windows['energy_L'],
        np.nan,
    )

    df_windows = df_windows[[
        'real_timestamp', 'window_time_sec',
        'prob_R', 'prob_L', 'prob_avg', 'prediction',
        'energy_R', 'energy_L', 'energy_total', 'ratio_R_L',
    ]]

    # ── Sheet 2: summary statistics ────────────────────────────────────────
    non_walk = df_windows[df_windows['prediction'] != 1]
    walk     = df_windows[df_windows['prediction'] == 1]

    # Total energy per wrist (non-walking only)
    total_R = non_walk['energy_R'].sum(skipna=True)
    total_L = non_walk['energy_L'].sum(skipna=True)
    total   = non_walk['energy_total'].sum(skipna=True)

    # Walking energy
    walk_R = walk['energy_R'].sum(skipna=True)
    walk_L = walk['energy_L'].sum(skipna=True)

    # Ratio stats
    ratio_mean   = df_windows['ratio_R_L'].mean(skipna=True)
    ratio_median = df_windows['ratio_R_L'].median(skipna=True)

    # Hourly energy to find most/least intense hour
    df_windows['hour_bin'] = df_windows['real_timestamp'].dt.floor('h')
    hourly = (
        non_walk.groupby(non_walk['real_timestamp'].dt.floor('h'))['energy_total']
        .sum()
        .sort_index()
    )
    most_intense_hour  = hourly.idxmax() if not hourly.empty else pd.NaT
    least_intense_hour = hourly.idxmin() if not hourly.empty else pd.NaT

    # Walking time
    n_windows        = len(df_windows)
    n_walk_windows   = int(walk['prediction'].count())
    window_dur_sec   = STEP_SIZE / 50  # assuming 50 Hz
    total_dur_min    = n_windows * window_dur_sec / 60
    walk_dur_min     = n_walk_windows * window_dur_sec / 60
    # Detect missing data periods per wrist 
    win_duration_sec = STEP_SIZE / 50  # window step duration at 50 Hz

    missing_R = find_missing_periods(
        R['real_win_times'] if R['real_win_times'] is not None else None,
        r['hub_times'], win_duration_sec,
    )
    missing_L = find_missing_periods(
        L['real_win_times'] if L['real_win_times'] is not None else None,
        r['hub_times'], win_duration_sec,
    )
    def format_missing(gaps):
        if not gaps:
            return 'None'
        return '; '.join(
            f"{s.strftime('%d/%m %H:%M:%S')} → {e.strftime('%d/%m %H:%M:%S')} ({d:.0f}s)"
            for s, e, d in gaps
        )

    summary_rows = [
        ('Subject',                         r['subject']),
        ('Recording start',                 r['start_dt'].strftime('%d/%m/%Y %H:%M:%S')),
        ('Recording end',                   df_windows['real_timestamp'].max().strftime('%d/%m/%Y %H:%M:%S')),
        ('Total windows',                   n_windows),
        ('Walking windows',                 n_walk_windows),
        ('Non-walking windows',             n_windows - n_walk_windows),
        ('Total duration (min)',            f'{total_dur_min:.1f}'),
        ('Walking duration (min)',          f'{walk_dur_min:.1f}'),
        ('',                                ''),
        ('── Energy (non-walking) ──',      ''),
        ('Total energy Right (non-walk)',   f'{total_R:.2f}'),
        ('Total energy Left (non-walk)',    f'{total_L:.2f}'),
        ('Total energy combined (non-walk)',f'{total:.2f}'),
        ('',                                ''),
        ('── Energy (walking) ──',          ''),
        ('Total energy Right (walk)',       f'{walk_R:.2f}'),
        ('Total energy Left (walk)',        f'{walk_L:.2f}'),
        ('',                                ''),
        ('── Ratio R/L ──',                 ''),
        ('Mean ratio R/L',                  f'{ratio_mean:.3f}'),
        ('Median ratio R/L',                f'{ratio_median:.3f}'),
        ('',                                ''),
        ('── Hourly breakdown ──',          ''),
        ('Most energy-intense hour',        most_intense_hour.strftime('%d/%m/%Y %H:%M') if pd.notna(most_intense_hour) else 'N/A'),
        ('Energy in most intense hour',     f'{hourly.max():.2f}' if not hourly.empty else 'N/A'),
        ('Least energy-intense hour',       least_intense_hour.strftime('%d/%m/%Y %H:%M') if pd.notna(least_intense_hour) else 'N/A'),
        ('Energy in least intense hour',    f'{hourly.min():.2f}' if not hourly.empty else 'N/A'),
        ('',                                ''),
        ('── Missing data periods ──',      ''),
        (f'Right wrist gaps (>={MISSING_PERIOD}s)',         f'{len(missing_R)} gap(s)'),
        *[(f'  Gap {i+1} (Right)',          f"{s.strftime('%d/%m %H:%M:%S')} → {e.strftime('%d/%m %H:%M:%S')} ({d:.0f}s)")
          for i, (s, e, d) in enumerate(missing_R)],
        (f'Left wrist gaps (>={MISSING_PERIOD}s)',          f'{len(missing_L)} gap(s)'),
        *[(f'  Gap {i+1} (Left)',           f"{s.strftime('%d/%m %H:%M:%S')} → {e.strftime('%d/%m %H:%M:%S')} ({d:.0f}s)")
          for i, (s, e, d) in enumerate(missing_L)],
    ]

    df_summary = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])

    # ── Write both sheets ──────────────────────────────────────────────────
    xlsx_path = os.path.join(output_dir, f"{r['subject']}_energy.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_windows.to_excel(writer, sheet_name='Per Window', index=False)
        df_summary.to_excel(writer, sheet_name='Summary',    index=False)

    print(f"  Saved: {xlsx_path} ({len(df_windows)} windows)")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)
    excel_dir = os.path.join(RESULTS_DIR, 'subject_energy')

    results = qsense_energy(model, device, QSENSE_CLINIC)
    for r in results:
        save_subject_excel(r, excel_dir)
    print(f"Saved per-subject Excel files to: {excel_dir}")
    output_dir_clinic = os.path.join(PLOTS_DIR, QSENSE_CLINIC.split(os.sep)[-1], 'energy_analysis')
    plot_energy_results_line(results, output_dir_clinic)

    results = qsense_energy(model, device, QSENSE_TEST)
    for r in results:        
        save_subject_excel(r, excel_dir)
    print(f"Saved per-subject Excel files to: {excel_dir}")    
    output_dir_test = os.path.join(PLOTS_DIR, QSENSE_TEST.split(os.sep)[-1], 'energy_analysis')
    plot_energy_results_bar(results, output_dir_test)


if __name__ == "__main__":
    main()