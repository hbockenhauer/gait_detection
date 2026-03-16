import os
import numpy as np
import matplotlib.pyplot as plt
from models.StrokeNet.strokenet_utils import (
    CONF_THRESH, MIN_ENERGY, MAX_ENERGY, MIN_FREQ, MAX_FREQ,
)

def insert_nan_breaks(times, values, break_times=None):
    """Insert NaNs at specified break timestamps so plotted lines split at discontinuities."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(times) != len(values) or len(times) < 2:
        return times, values

    if break_times is None:
        return times, values

    breaks = np.asarray(break_times, dtype=float)
    breaks = breaks[np.isfinite(breaks)]
    if len(breaks) == 0:
        return times, values

    breaks = np.unique(np.sort(breaks))

    # Keep only breaks that lie strictly inside the plotted span.
    breaks = breaks[(breaks > times[0]) & (breaks < times[-1])]
    if len(breaks) == 0:
        return times, values

    times_out = [times[0]]
    values_out = [values[0]]
    b_idx = 0

    for i in range(1, len(times)):
        prev_t = times[i - 1]
        curr_t = times[i]

        while b_idx < len(breaks) and breaks[b_idx] <= prev_t:
            b_idx += 1

        if b_idx < len(breaks) and prev_t < breaks[b_idx] <= curr_t:
            times_out.append(np.nan)
            values_out.append(np.nan)

        times_out.append(curr_t)
        values_out.append(values[i])

    return np.asarray(times_out), np.asarray(values_out)


def plot_per_activity(results_list, subjects, metrics, plots_dir):
    activities_by_type = {}
    for result in results_list:
        act = result['activity_type']
        activities_by_type.setdefault(act, []).append(result)

    activity_types = sorted(activities_by_type.keys())

    # Color per subject, linestyle per wrist
    color_palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    all_subjects_in_results = sorted(set(r['subject'].capitalize() for r in results_list))
    subject_colors = {subj: color_palette[idx % len(color_palette)]
                      for idx, subj in enumerate(all_subjects_in_results)}
    wrist_styles = {'right': '-', 'left': '--'}

    for activity_type, activity_results in activities_by_type.items():
        n_rows = len(metrics) + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        fig.suptitle(f"Activity: {activity_type} (StrokeNet)",
                     fontsize=16, fontweight='bold')

        x_min = None
        x_max = None
        for result in activity_results:
            for key in ['raw_timestamps', 'timestamps']:
                t = np.asarray(result.get(key, []), dtype=float)
                t = t[np.isfinite(t)]
                if len(t) == 0:
                    continue
                t_min = float(np.min(t))
                t_max = float(np.max(t))
                x_min = t_min if x_min is None else min(x_min, t_min)
                x_max = t_max if x_max is None else max(x_max, t_max)

        has_data = False

        # --- Metric subplots ---
        for ax, metric in zip(axes[:-1], metrics):
            for subject in subjects:
                subject_results = [r for r in activity_results
                                   if r['subject'].capitalize() == subject.capitalize()]
                for result in subject_results:
                    if metric not in ['probability', 'energy', 'Q_energies', 'frequency']:
                        continue
                    has_data = True

                    subj_cap  = result['subject'].capitalize()
                    color     = subject_colors.get(subj_cap, '#333333')
                    style     = wrist_styles[result['wrist']]
                    label_str = f"{subj_cap} | {result['wrist']}"

                    if metric == 'probability':
                        values = result['probability']
                    elif metric == 'energy':
                        values = result['energy']
                    elif metric == 'Q_energies':
                        values = result['Q_energies']
                    elif metric == 'frequency':
                        values = result['frequency']

                    plot_times, plot_values = insert_nan_breaks(
                        result['timestamps'],
                        values,
                        result.get('discontinuity_times', []),
                    )

                    ax.plot(plot_times, plot_values,
                            color=color, linestyle=style,
                            linewidth=1.5, alpha=0.95, label=label_str)

            if metric == 'probability':
                ax.axhline(CONF_THRESH, color='black', linestyle='--', linewidth=1.5,
                           label=f'Threshold = {CONF_THRESH}')
                ax.set_ylim(-0.05, 1.1)
            elif metric in ['energy', 'Q_energies']:
                ax.axhline(MIN_ENERGY, color='black', linestyle='--', linewidth=1.5,
                           label=f'Min = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY, color='black', linestyle='--', linewidth=1.5,
                           label=f'Max = {MAX_ENERGY}')
            elif metric == 'frequency':
                ax.axhline(MIN_FREQ, color='black', linestyle='--', linewidth=1.5,
                           label=f'Min = {MIN_FREQ}')
                ax.axhline(MAX_FREQ, color='black', linestyle='--', linewidth=1.5,
                           label=f'Max = {MAX_FREQ}')

            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)

        # --- GT vs Prediction subplot ---
        ax_gt = axes[-1]
        for subject in subjects:
            subject_results = [r for r in activity_results
                               if r['subject'].capitalize() == subject.capitalize()]
            for result in subject_results:
                subj_cap = result['subject'].capitalize()
                color    = subject_colors.get(subj_cap, '#333333')
                style    = wrist_styles[result['wrist']]

                raw_plot_times, raw_plot_gt = insert_nan_breaks(
                    result['raw_timestamps'],
                    result['raw_gt'],
                    result.get('discontinuity_times', []),
                )
                pred_plot_times, pred_plot_values = insert_nan_breaks(
                    result['timestamps'],
                    result['y_pred'] + 0.05,
                    result.get('discontinuity_times', []),
                )

                ax_gt.fill_between(raw_plot_times, 0, raw_plot_gt,
                                   step='post', alpha=0.25, color=color,
                                   label=f"{subj_cap} | {result['wrist']} | GT")
                ax_gt.step(pred_plot_times, pred_plot_values,
                           where='post', color=color, linestyle=style, linewidth=2.5,
                           label=f"{subj_cap} | {result['wrist']} | Pred")

        ax_gt.set_ylabel('GT / Prediction', fontsize=12)
        ax_gt.set_ylim(-0.1, 1.15)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.set_xlabel('Time (s)', fontsize=12)

        if x_min is not None and x_max is not None and x_max > x_min:
            for ax in axes:
                ax.set_xlim(x_min, x_max)

        if not has_data:
            plt.close(fig)
            continue

        # Deduplicated legend
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        seen = set()
        uh, ul = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                uh.append(h)
                ul.append(l)

        fig.legend(uh, ul, loc='center left', bbox_to_anchor=(0.87, 0.5),
                   fontsize=10, title='Subject | Wrist', frameon=True, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.86, 0.95])

        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f'activity_{activity_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {save_path}")