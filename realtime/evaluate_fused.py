import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os

from detect_fused      import simulate_realtime, load_segmented, WINDOW_SIZE, sync_wrists, fuse_predictions, FILE_NAME_R, FILE_NAME_L


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    valid      = ~np.isnan(y_pred)
    yt         = (y_true[valid] == 1).astype(int)
    yp         = y_pred[valid].astype(int)

    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)

    print("\n" + "=" * 60)
    print(f"Results for: {label}")
    print("=" * 60)
    print(f"  Evaluated samples : {valid.sum()}  (skipped {(~valid).sum()} NaN samples)")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print("-" * 60)
    print(f"  True  Positives   : {int(np.sum((yp == 1) & (yt == 1)))}")
    print(f"  False Positives   : {int(np.sum((yp == 1) & (yt == 0)))}")
    print(f"  False Negatives   : {int(np.sum((yp == 0) & (yt == 1)))}")
    print(f"  True  Negatives   : {int(np.sum((yp == 0) & (yt == 0)))}")
    print("=" * 60)

def plot_results_fused(df_sync: pd.DataFrame,
                       pred_r:  np.ndarray,
                       pred_l:  np.ndarray,
                       y_fused: np.ndarray,
                       y_true:  np.ndarray,
                       title:   str) -> None:
    # Use the timestamp column that survived the merge (always present from R side)
    ts_col = "HH:mm:ss.fff_r" if "HH:mm:ss.fff_r" in df_sync.columns else "HH:mm:ss.fff"
    time_series   = pd.to_timedelta(df_sync[ts_col].str.strip())
    time_all_sec  = time_series.dt.total_seconds()

    # Segment gap markers
    jump_times_sec_r = []
    jump_times_sec_l = []
    for seg_id, grp in df_sync.groupby("segment_r", sort=True):
        first_idx = grp.index[0]
        if first_idx != df_sync.index[0]:
            jump_times_sec_r.append(time_all_sec.iloc[first_idx])

    for seg_id, grp in df_sync.groupby("segment_l", sort=True):
        first_idx = grp.index[0]
        if first_idx != df_sync.index[0]:
            jump_times_sec_l.append(time_all_sec.iloc[first_idx])

    def _add_truth_bands(ax):
        ax.fill_between(time_all_sec, -0.1, 1.4,
                        where=(y_true == 1), alpha=0.2, color="green",
                        transform=ax.get_xaxis_transform(), label="Ground truth (walking)")
        if 2 in y_true:
            ax.fill_between(time_all_sec, -0.1, 1.4,
                            where=(y_true == 2), alpha=0.2, color="purple",
                            transform=ax.get_xaxis_transform(), label="Functional arm use")

    def _add_gaps(ax, labeled=True):
        for i, jt in enumerate(jump_times_sec_r):
            ax.axvline(x=jt, color="orange", linewidth=1.0, linestyle="--", alpha=0.8,
                       label="Time gap right" if (labeled and i == 0) else None)
        for i, jt in enumerate(jump_times_sec_l):
            ax.axvline(x=jt, color="yellow", linewidth=1.0, linestyle="--", alpha=0.8,
                       label="Time gap left" if (labeled and i == 0) else None)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    # ── Panel 1: raw acc R ────────────────────────────────────────────────────
    ax = axes[0]
    _add_truth_bands(ax)
    for col in [c for c in df_sync.columns if "acc" in c.lower() and c.endswith("_r")]:
        ax.plot(time_all_sec, df_sync[col].astype(float),
                label=col, alpha=0.8, marker=".", linestyle="None", markersize=2)
    _add_gaps(ax)
    ax.set_ylabel("Acc R (m/s²)")
    ax.legend(loc="upper left", fontsize=7)

    # ── Panel 2: raw acc L ────────────────────────────────────────────────────
    ax = axes[1]
    _add_truth_bands(ax)
    for col in [c for c in df_sync.columns if "acc" in c.lower() and c.endswith("_l")]:
        ax.plot(time_all_sec, df_sync[col].astype(float),
                label=col, alpha=0.8, marker=".", linestyle="None", markersize=2)
    _add_gaps(ax, labeled=False)
    ax.set_ylabel("Acc L (m/s²)")
    ax.legend(loc="upper left", fontsize=7)

    # ── Panel 3: per-wrist predictions ────────────────────────────────────────
    ax = axes[2]
    _add_truth_bands(ax)
    ax.plot(time_all_sec, pred_r, label="pred R", alpha=0.75,
            linewidth=1, color="steelblue")
    ax.plot(time_all_sec, pred_l, label="pred L", alpha=0.75,
            linewidth=1, color="tomato")
    _add_gaps(ax, labeled=False)
    ax.set_ylabel("Pred per wrist")
    ax.set_ylim(-0.1, 1.4)
    ax.legend(loc="upper left", fontsize=7)

    # ── Panel 4: fused prediction ─────────────────────────────────────────────
    ax = axes[3]
    _add_truth_bands(ax)
    ax.plot(time_all_sec, y_fused, label="Fused prediction", alpha=0.9,
            linewidth=1.2, color="darkorchid")
    _add_gaps(ax, labeled=False)
    ax.set_ylabel("Fused (1=walk)")
    ax.set_ylim(-0.1, 1.4)
    ax.legend(loc="upper left", fontsize=7)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
    ))
    fig.autofmt_xdate()
    plt.tight_layout()

def extract_true_labels(df_sync: pd.DataFrame, data_path: str) -> np.ndarray:
    """
    Initially look at the Label column from R; fall back to L.
    """
    label_r = df_sync.get("Label_r")
    label_l = df_sync.get("Label_l")

    combined = None
    for col in (label_r, label_l):
        if col is None:
            continue
        s = pd.to_numeric(col, errors="coerce")
        combined = s if combined is None else combined.combine_first(s)

    if combined is not None and combined.notna().any():
        return combined.fillna(0).astype(int).to_numpy()

    return np.zeros(len(df_sync), dtype=int)

def evaluate_fused(data_path: str) -> None:
    # 1. Load and scale both wrists
    df_r = load_segmented(data_path, FILE_NAME_R)
    df_l = load_segmented(data_path, FILE_NAME_L)

    for df in (df_r, df_l):
        acc_cols = [c for c in df.columns if "acc" in c.lower()]
        df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 2. Synchronise
    df_sync = sync_wrists(df_r, df_l)

    # 3. True labels (combined from whichever wrist has them)
    y_true = extract_true_labels(df_sync, data_path)

    # 4. Run GSD per segment, per wrist
    n = len(df_sync)
    pred_r = np.full(n, np.nan)
    pred_l = np.full(n, np.nan)

    # ── Right wrist: iterate over R segments ──────────────────────────────────
    for _, grp in df_sync.groupby("segment_r", sort=True):
        idx = grp.index
        acc_cols_r = [c for c in grp.columns if "acc" in c.lower() and c.endswith("_r")]
        if len(grp) < WINDOW_SIZE or len(acc_cols_r) != 3:
            pred_r[idx] = np.nan
            continue
        seg = grp[acc_cols_r].rename(columns={c: c[:-2] for c in acc_cols_r}).reset_index(drop=True)
        pred_r[idx] = simulate_realtime(seg)

    # ── Left wrist: iterate over L segments ───────────────────────────────────
    for _, grp in df_sync.groupby("segment_l", sort=True):
        idx = grp.index
        acc_cols_l = [c for c in grp.columns if "acc" in c.lower() and c.endswith("_l")]
        if len(grp) < WINDOW_SIZE or len(acc_cols_l) != 3:
            pred_l[idx] = np.nan
            continue
        seg = grp[acc_cols_l].rename(columns={c: c[:-2] for c in acc_cols_l}).reset_index(drop=True)
        pred_l[idx] = simulate_realtime(seg)

    # 5. Fuse
    y_fused = fuse_predictions(pred_r, pred_l)

    # 6. Metrics — fused, plus per-wrist for comparison
    print_metrics(y_true, pred_r,  f"Right wrist only  ({FILE_NAME_R})")
    print_metrics(y_true, pred_l,  f"Left wrist only   ({FILE_NAME_L})")
    print_metrics(y_true, y_fused, f"Fused             ({data_path})")

    # 7. Plot
    plot_results_fused(
        df_sync=df_sync,
        pred_r=pred_r,
        pred_l=pred_l,
        y_fused=y_fused,
        y_true=y_true,
        title=f"Fused evaluation — {os.path.basename(data_path)}",
    )
    plt.show()

if __name__ == "__main__":
  DATA_PATH = sys.argv[1]
  evaluate_fused(DATA_PATH)