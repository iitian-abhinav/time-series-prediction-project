"""
Walk-forward validation.

A single last-30-day holdout can be lucky or unlucky — a model might
look great (or terrible) just because of which 30 days it was tested
on. Walk-forward validation re-runs the same train/forecast/score
cycle across several consecutive rolling windows further back in
history, then averages the metrics, giving a much more trustworthy
comparison between models.

Window 1 = the most recent `horizon` days (same as the single-window
evaluation already shown). Window 2 = the `horizon` days before that,
trained on everything before it. And so on.
"""
import pandas as pd


def get_walk_forward_windows(series, horizon: int = 30, n_windows: int = 5, min_train: int = 100):
    """Split `series` into up to `n_windows` non-overlapping
    (train, test) pairs, walking backward from the most recent data.
    Stops early if there isn't enough history left for `min_train`
    training points."""
    n = len(series)
    windows = []
    for w in range(n_windows):
        test_end = n - w * horizon
        test_start = test_end - horizon
        if test_start < min_train:
            break
        windows.append((series.iloc[:test_start], series.iloc[test_start:test_end]))
    return windows


def run_walk_forward(eval_window_fn, series, horizon: int = 30, n_windows: int = 5, min_train: int = 100):
    """Run `eval_window_fn(train, test) -> metrics dict` across every
    walk-forward window. Returns (avg_metrics_dict, per_window_df),
    or (None, empty_df) if no window could be evaluated."""
    windows = get_walk_forward_windows(series, horizon, n_windows, min_train)
    rows = []
    for i, (train, test) in enumerate(windows):
        # Window 1 = most recent, matching the ordering shown to users.
        window_num = i + 1
        try:
            metrics = eval_window_fn(train, test)
        except Exception:
            continue
        metrics = dict(metrics)
        metrics['window'] = window_num
        rows.append(metrics)

    if not rows:
        return None, pd.DataFrame()

    per_window_df = pd.DataFrame(rows).set_index('window')
    avg_metrics = per_window_df.mean(numeric_only=True).round(3).to_dict()
    return avg_metrics, per_window_df
