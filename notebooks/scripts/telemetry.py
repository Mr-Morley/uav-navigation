"""
Flight 1: cleaning, smoothing, resampling.
Flight 2: timeline repair, duplicate interpolation, takeoff detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar


EARTH_RADIUS = 6_371_000  # metres

# D.2.2.1  Flight 1: Preparation, Cleaning and Re-sampling
# ─────────────────────────────────────────────────────────────

def radians_to_degrees(df, columns):
    """Convert selected columns from radians to degrees."""
    df = df.copy()
    for col in columns:
        df[col] = np.degrees(df[col])
    return df


def process_f1_time(df):
    """
    Parse DATE/TIME, enforce dominant date, construct COUNT_TIME.
    Returns (cleaned_df, estimated_frequency_hz).
    """
    df = df.copy()
    df["DATE_C"] = df["DATE"].str.extract(r"(\d{1,2}/\d{1,2}/\d{4})")
    df = df[df["DATE_C"] == df["DATE_C"].mode()[0]]
    dt = pd.to_datetime(df["DATE_C"] + " " + df["TIME"])
    interval = (dt.iloc[-1] - dt.iloc[0]).total_seconds() / len(dt)
    df["COUNT_TIME"] = np.arange(len(df)) * interval
    return df.drop(columns=["DATE", "TIME", "DATE_C"]), 1 / interval


def apply_outlier_filters(df, filters):
    """Set out-of-range values to NaN using (col, min, max) tuples."""
    df = df.copy()
    for col, lo, hi in filters:
        df[col] = df[col].where(df[col].between(lo, hi))
    return df


def interpolate_and_smooth(df, window=3):
    """Linear interpolation then rolling-median smoothing."""
    df = df.copy()
    df.interpolate(method="linear", inplace=True)
    df = df.rolling(window, center=True, min_periods=1).median()
    return df


def plot_gps_comparison(raw_df, processed_df, fields, time_col="COUNT_TIME", save_path=None):
    """Figure D.4 — raw vs filtered+smoothed GPS signals (2×3)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, field in zip(axes.flatten(), fields):
        ax.plot(raw_df[time_col], raw_df[field], alpha=0.4, lw=0.8, label="Filtered")
        ax.plot(processed_df[time_col], processed_df[field], lw=1.2, label="Filtered + Smoothed")
        ax.set_title(field)
        ax.set_xlabel(time_col)
        ax.set_ylabel(field)
        ax.grid(True); ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200); plt.close(fig)
    else:
        plt.show()


def apply_sg_filter(df, window_seconds=5, sg_window=31, sg_order=2,
                    time_col="COUNT_TIME", save_path=None):
    """Figure D.5 — Savitzky-Golay filter on LAT/LON (random slice)."""
    t = df[time_col].values
    valid = np.where((t > t.min() + window_seconds) & (t < t.max() - window_seconds))[0]
    idx = np.random.choice(valid)
    t0 = t[idx]
    mask = (t >= t0 - window_seconds) & (t <= t0 + window_seconds)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, field in zip(axes, ["LATITUDE", "LONGITUDE"]):
        raw = df.loc[mask, field].values
        sg = savgol_filter(raw, sg_window, sg_order)
        ax.plot(t[mask], raw, label="RAW", alpha=0.6)
        ax.plot(t[mask], sg, label="SG Filtered", linewidth=2)
        ax.set_ylabel(field); ax.grid(True); ax.legend()
    axes[-1].set_xlabel(time_col)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200); plt.close()
    else:
        plt.show()


def estimate_gps_polling(df, lat="LATITUDE", lon="LONGITUDE", speed="GND_SPD",
                         min_speed=1, max_speed=70, freq_bounds=(1.0, 100.0)):
    """
    Figure D.6 — estimate true GPS polling frequency from distance/speed consistency.
    Returns dict with optimal frequencies and a plot() callable.
    """
    lat_r, lon_r = np.radians(df[lat].values), np.radians(df[lon].values)
    dlat, dlon = np.diff(lat_r), np.diff(lon_r)
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:-1])*np.cos(lat_r[1:])*np.sin(dlon/2)**2
    dists = 2 * EARTH_RADIUS * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    spds = df[speed].values[:-1]

    m = np.isfinite(dists) & np.isfinite(spds) & (spds > min_speed) & (spds < max_speed) & (dists > 0.1)
    dists, spds = dists[m], spds[m]
    if len(dists) < 100:
        raise ValueError("Insufficient valid GPS points")

    def rmse(fs): dt=1/fs; return np.sqrt(np.mean((dists - spds*dt)**2))
    def medae(fs): dt=1/fs; return np.median(np.abs(dists - spds*dt))

    r_opt = minimize_scalar(rmse, bounds=freq_bounds, method="bounded")
    m_opt = minimize_scalar(medae, bounds=freq_bounds, method="bounded")
    freqs = np.linspace(*freq_bounds, 400)

    def plot(save_path=None):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot(freqs, [rmse(f) for f in freqs], label="RMSE")
        a1.axvline(r_opt.x, ls='--', label=f"RMSE opt {r_opt.x:.1f} Hz")
        a1.set(title="RMSE vs Frequency", xlabel="Hz", ylabel="metres"); a1.grid(True); a1.legend()
        a2.plot(freqs, [medae(f) for f in freqs], label="MedAE")
        a2.axvline(m_opt.x, ls='--', label=f"MedAE opt {m_opt.x:.1f} Hz")
        a2.set(title="MedAE vs Frequency", xlabel="Hz", ylabel="metres"); a2.grid(True); a2.legend()
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200); plt.close()
        else: plt.show()

    return {"opt_freq_rmse": r_opt.x, "opt_freq_medae": m_opt.x,
            "rmse_at_opt": r_opt.fun, "medae_at_opt": m_opt.fun, "plot": plot}


# ─────────────────────────────────────────────────────────────
# D.2.2.2  Flight 2: Preparation and Cleaning
# ─────────────────────────────────────────────────────────────

def repair_timeline(df, time_col='TIME', cycle_s=3600):
    """
    Figure D.7 — fix sawtooth clock resets, build SHIFTED_TIME and COUNT_TIME.
    Also converts GND_SPD from km/h → m/s.
    """
    df = df.copy()
    parts = df[time_col].str.split(':')
    raw_sec = parts.str[0].astype(float) * 60 + parts.str[1].astype(float)
    resets = (raw_sec.diff() < 0)
    df['SHIFTED_TIME'] = raw_sec + resets.cumsum() * cycle_s
    df['COUNT_TIME'] = np.arange(len(df)) / 10.0
    if 'GND_SPD' in df.columns:
        df['GND_SPD'] = df['GND_SPD'] / 3.6
    median_dt = df['SHIFTED_TIME'].diff().median()
    return df, 1 / median_dt, 10


def plot_timeline_reconstruction(df, save_path=None):
    """Figure D.7 visualisation."""
    plt.figure(figsize=(10, 4))
    plt.plot(df['SHIFTED_TIME'], label='Shifted Time (Repaired)', color='red', alpha=0.8, lw=1.5)
    plt.plot(df['COUNT_TIME'], label='10 Hz Baseline', color='blue', ls='--', alpha=0.6)
    plt.plot(df['SHIFTED_TIME'] % 3600, label='Original Sawtooth', color='green', alpha=0.3, lw=1)
    plt.title("Timeline Reconstruction: Sawtooth vs. Repaired")
    plt.xlabel("Sample Index"); plt.ylabel("Seconds"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200); plt.close()
    else: plt.show()


def analyze_duplicates_by_speed(df, lat='LATITUDE', lon='LONGITUDE', speed='GND_SPD'):
    """Table D.8 — consecutive duplicate rate by speed bin."""
    is_dup = (df[lat].eq(df[lat].shift()) & df[lon].eq(df[lon].shift())).iloc[1:]
    avg_spd = (df[speed] + df[speed].shift(1)).iloc[1:] / 2.0
    pairs = pd.DataFrame({'is_dup': is_dup, 'spd': avg_spd})
    bins = [-np.inf, 0.1, 3, 10, 30, 60, np.inf]
    labels = ['Stationary', '0–3 m/s', '3–10 m/s', '10–30 m/s', '30–60 m/s', '>60 m/s']
    pairs['bin'] = pd.cut(pairs['spd'], bins=bins, labels=labels)
    stats = pairs.groupby('bin', observed=True).agg(
        Total_Pairs=('is_dup', 'count'), Duplicate_Pairs=('is_dup', 'sum'))
    stats['Rate_%'] = (stats['Duplicate_Pairs'] / stats['Total_Pairs'] * 100).round(2)
    return stats


def clean_and_interpolate(df, min_speed=0.5, lat='LATITUDE', lon='LONGITUDE'):
    """Interpolate flying duplicates; return (df, count_fixed, rmse_metres)."""
    df = df.copy()
    is_dup = (df[lat] == df[lat].shift(1)) & (df[lon] == df[lon].shift(1))
    is_flying = df['GND_SPD'] > min_speed
    to_fix = is_dup & is_flying
    original = df[[lat, lon]].copy()
    df.loc[to_fix, [lat, lon]] = np.nan
    df[[lat, lon]] = df[[lat, lon]].interpolate(method='linear')
    errors = np.sqrt((df[lat] - original[lat])**2 + (df[lon] - original[lon])**2)
    rmse = np.sqrt((errors**2).mean()) * 111320
    return df, to_fix.sum(), rmse


# D.2.2.3  Identification of Takeoff  (Algorithm 3.2)
# ─────────────────────────────────────────────────────────────

def detect_gps_takeoff(df, t_start, t_end, tau=0.5,
                       time_col='SHIFTED_TIME', speed_col='GND_SPD'):
    """
    Algorithm 3.2 — ground-speed threshold crossing.
    Returns (index, takeoff_time) or None.
    """
    mask = (df[time_col] >= t_start) & (df[time_col] <= t_end)
    win = df.loc[mask]
    if win.empty:
        return None
    speed = win[speed_col].values
    crossings = (speed[:-1] <= tau) & (speed[1:] > tau)
    idxs = np.where(crossings)[0]
    if len(idxs):
        loc = win.index[idxs[0]]
        return loc, win.loc[loc, time_col]
    return None
