"""
Handles camera synchronisation, PSD analysis, filter design, and takeoff detection.
All functions referenced in Appendix D of the report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────
# D.2.1.3  Camera Synchronisation
# ─────────────────────────────────────────────────────────────

def calc_sync_stats(c1_data, c2_data, common_time, signal_name, fps=23.98):
    """Cross-correlate two signals and return lag / correlation statistics."""
    valid = ~(np.isnan(c1_data) | np.isnan(c2_data))
    c1_data, c2_data = c1_data[valid], c2_data[valid]

    sampling_rate = 1 / np.median(np.diff(common_time[valid]))
    c1_c = c1_data - np.mean(c1_data)
    c2_c = c2_data - np.mean(c2_data)

    corr = signal.correlate(c1_c, c2_c, mode='full')
    lags = signal.correlation_lags(len(c1_c), len(c2_c), mode='full')

    best_idx = np.argmax(np.abs(corr))
    best_lag = lags[best_idx]
    best_corr = corr[best_idx] / (len(c1_c) * np.std(c1_c) * np.std(c2_c))
    lag_ms = (best_lag / sampling_rate) * 1000
    lag_frames = int(round(lag_ms / 1000 * fps))

    return {
        'Signal': signal_name,
        'Best Lag (Samples)': int(best_lag),
        'Best Lag (ms)': round(lag_ms, 1),
        'Lag (Frames)': lag_frames,
        'Correlation': round(best_corr, 4),
        'Status': "C1 Leads" if best_lag > 0 else ("C2 Leads" if best_lag < 0 else "In Sync"),
        'Sampling Rate (Hz)': round(sampling_rate, 1),
    }


def imu_sync_pipeline(imu_c1, imu_c2, data_type='GYRO'):
    """Sync for high-frequency IMU channels — individual components + magnitude."""
    t1 = imu_c1['time'].values - imu_c1['time'].iloc[0]
    t2 = imu_c2['time'].values - imu_c2['time'].iloc[0]
    dt = min(np.median(np.diff(t1)), np.median(np.diff(t2)))
    common_time = np.arange(max(t1.min(), t2.min()), min(t1.max(), t2.max()), dt)

    results = []
    comps_c1, comps_c2 = [], []

    # Individual channels
    for axis in range(3):
        col = f'{data_type}_{axis}'
        c1_interp = interp1d(t1, imu_c1[col].values, kind='linear', bounds_error=False)(common_time)
        c2_interp = interp1d(t2, imu_c2[col].values, kind='linear', bounds_error=False)(common_time)
        comps_c1.append(c1_interp)
        comps_c2.append(c2_interp)
        results.append(calc_sync_stats(c1_interp, c2_interp, common_time, f"{data_type} {axis}"))

    # Magnitude
    mag1 = np.sqrt(sum(c**2 for c in comps_c1))
    mag2 = np.sqrt(sum(c**2 for c in comps_c2))
    results.append(calc_sync_stats(mag1, mag2, common_time, f"{data_type} MAG"))

    return results


def quaternion_to_euler(w, x, y, z):
    """Convert quaternion (w,x,y,z) → Euler angles (roll, pitch, yaw) in radians."""
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return roll, pitch, yaw


def _angular_velocity(angles, times):
    """Centred finite-difference angular velocity."""
    angles, times = np.asarray(angles), np.asarray(times)
    vel = np.zeros_like(angles)
    vel[1:-1] = (angles[2:] - angles[:-2]) / (times[2:] - times[:-2])
    if len(angles) > 1:
        vel[0]  = (angles[1] - angles[0]) / (times[1] - times[0])
        vel[-1] = (angles[-1] - angles[-2]) / (times[-1] - times[-2])
    return vel


def low_freq_sync(orient_df, time_col='time'):
    """Gravity + orientation angular-velocity sync (Table D.5, low-freq rows)."""
    times = orient_df[time_col].values
    results = []

    # Gravity channels
    for i in range(3):
        results.append(calc_sync_stats(
            orient_df[f'GRAV_{i}_c1'].values,
            orient_df[f'GRAV_{i}_c2'].values,
            times, f"GRAV Component {i}"))

    m1 = np.sqrt(sum(orient_df[f'GRAV_{i}_c1']**2 for i in range(3))).values
    m2 = np.sqrt(sum(orient_df[f'GRAV_{i}_c2']**2 for i in range(3))).values
    results.append(calc_sync_stats(m1, m2, times, "GRAV Magnitude"))

    # Orientation angular velocities
    for cam_suffix in ['_c1', '_c2']:
        cols = [f'CORI_{j}{cam_suffix}' for j in range(4)]
        r, p, y = quaternion_to_euler(*[orient_df[c].values for c in cols])
        if cam_suffix == '_c1':
            v1 = [_angular_velocity(np.degrees(np.unwrap(r)), times),
                  _angular_velocity(np.degrees(p), times),
                  _angular_velocity(np.degrees(np.unwrap(y)), times)]
        else:
            v2 = [_angular_velocity(np.degrees(np.unwrap(r)), times),
                  _angular_velocity(np.degrees(p), times),
                  _angular_velocity(np.degrees(np.unwrap(y)), times)]

    for i, name in enumerate(["Roll Velocity", "Pitch Velocity", "Yaw Velocity"]):
        results.append(calc_sync_stats(v1[i], v2[i], times, name))

    return pd.DataFrame(results)


def run_full_sync_table(imu_pairs, orient_pairs):
    rows = []
    for label, c1, c2 in imu_pairs:
        for sensor in ['GYRO', 'ACCL']:
            for res in imu_sync_pipeline(c1, c2, sensor):
                res['Flight'] = label
                rows.append(res)

    for label, odf in orient_pairs:
        lf = low_freq_sync(odf)
        lf['Flight'] = label
        rows.extend(lf.to_dict('records'))

    cols = ['Flight', 'Signal', 'Sampling Rate (Hz)',
            'Best Lag (Samples)', 'Best Lag (ms)', 'Lag (Frames)',
            'Correlation', 'Status']
    return pd.DataFrame(rows)[cols]


def sync_and_truncate(c1_df, c2_df, offset, c1_leads=True):
    """Align two camera DataFrames by a known frame offset and merge on time."""
    if c1_leads:
        s1 = c1_df.iloc[offset:].reset_index(drop=True)
        s2 = c2_df.reset_index(drop=True)
    else:
        s2 = c2_df.iloc[offset:].reset_index(drop=True)
        s1 = c1_df.reset_index(drop=True)

    n = min(len(s1), len(s2))
    s1, s2 = s1.iloc[:n].copy(), s2.iloc[:n].copy()
    s1['time'] -= s1['time'].iloc[0]
    s2['time'] -= s2['time'].iloc[0]

    return pd.merge_asof(
        s1.sort_values('time'), s2.sort_values('time'),
        on='time', direction='nearest', suffixes=('_c1', '_c2'))


# ─────────────────────────────────────────────────────────────
# D.2.1.4  Power Spectral Density Analysis
# ─────────────────────────────────────────────────────────────

def plot_psd(df, sensor, flight_label, fs=None, ax=None, save_path=None):
    """
    PSD for one sensor type on one flight.
    """
    if fs is None:
        fs = 1 / np.median(np.diff(df['time']))

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))

    for col in sorted(df.columns):
        if sensor in col:
            f_, p_ = signal.welch(df[col], fs=fs, nperseg=512)
            ax.semilogy(f_, p_, label=col)

    ax.set_title(f"{flight_label}: {sensor} PSD")
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize='small', ncol=2)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        plt.show()





# D.2.1.5  Filter Design and Application  (Table D.6 / Figure D.3)
# ─────────────────────────────────────────────────────────────

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass filter."""
    data = np.asarray(data)
    if len(data) <= 15:
        return data
    nyq = 0.5 * fs
    norm = min(cutoff / nyq, 0.99)
    b, a = signal.butter(order, norm, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def optimize_filters(df, flight_label, grid=None):
    """
    Grid-search for optimal low-pass cutoff per channel (Table D.6).

    Returns (filtered_df, results_table).
    """
    fs = 1.0 / np.median(np.diff(df['time']))
    channels = ['ACCL_0', 'ACCL_1', 'ACCL_2', 'GYRO_0', 'GYRO_1', 'GYRO_2']
    if grid is None:
        grid = np.arange(0.1, 50.1, 0.1)

    results = []
    filt_df = df[['time']].copy()

    for chan in channels:
        c1, c2 = df[f'{chan}_c1'].values, df[f'{chan}_c2'].values
        raw_corr = np.corrcoef(c1, c2)[0, 1]

        best_err, best_cutoff = np.inf, 0.1
        for cutoff in grid:
            c1_f = butter_lowpass_filter(c1, cutoff, fs)
            c2_f = butter_lowpass_filter(c2, cutoff, fs)
            err = np.sqrt(np.mean((c1_f - c2_f) ** 2))
            if err < best_err:
                best_err, best_cutoff = err, cutoff

        # D.2.1.5 override: ACCL_1 median cutoff = 0.2 Hz
        if 'ACCL_1' in chan:
            best_cutoff = 0.2

        c1_f = butter_lowpass_filter(c1, best_cutoff, fs)
        c2_f = butter_lowpass_filter(c2, best_cutoff, fs)
        best_corr = np.corrcoef(c1_f, c2_f)[0, 1]

        filt_df[f'{chan}_c1'], filt_df[f'{chan}_c2'] = c1_f, c2_f
        results.append({
            'Flight': flight_label, 'Channel': chan,
            'Raw Correl.': round(raw_corr, 3),
            'Filt. Correl.': round(best_corr, 3),
            'Cutoff (Hz)': round(best_cutoff, 1),
        })

    return filt_df, pd.DataFrame(results)


def apply_table_d6_filters(df, flight_id):
    """Apply the exact cutoffs from Table D.6 (for takeoff detection, etc.)."""
    cutoffs = {
        'F1': {'ACCL_0': 0.2, 'ACCL_1': 0.2, 'ACCL_2': 0.3,
               'GYRO_0': 0.3, 'GYRO_1': 4.1, 'GYRO_2': 0.7},
        'F2': {'ACCL_0': 0.1, 'ACCL_1': 0.2, 'ACCL_2': 0.1,
               'GYRO_0': 0.2, 'GYRO_1': 1.3, 'GYRO_2': 1.2},
    }
    fs = 1.0 / np.median(np.diff(df['time']))
    filt = df.copy()
    for chan, fc in cutoffs[flight_id].items():
        for sfx in ['_c1', '_c2']:
            filt[f'{chan}{sfx}'] = butter_lowpass_filter(df[f'{chan}{sfx}'].values, fc, fs)
    return filt


def plot_raw_vs_filtered_accl(raw_f1, filt_f1, raw_f2, filt_f2, t_range=(1000, 2000), save_path=None):
    """Reproduce Figure D.3 — raw vs filtered accelerometer comparison."""
    trace_colors = {
        'c1_ACCL_0': 'tab:blue',   'c1_ACCL_1': 'tab:orange', 'c1_ACCL_2': 'tab:green',
        'c2_ACCL_0': 'tab:red',    'c2_ACCL_1': 'tab:purple', 'c2_ACCL_2': 'tab:brown',
    }

    w1 = (raw_f1['time'] >= t_range[0]) & (raw_f1['time'] <= t_range[1])
    w2 = (raw_f2['time'] >= t_range[0]) & (raw_f2['time'] <= t_range[1])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, (raw, filt, w, label) in enumerate([
        (raw_f1, filt_f1, w1, 'Flight 1'), (raw_f2, filt_f2, w2, 'Flight 2')
    ]):
        for ax_i in range(3):
            col = f'ACCL_{ax_i}'
            c1k, c2k = f'c1_{col}', f'c2_{col}'
            axes[0, i].plot(raw.loc[w, 'time'], raw.loc[w, f'{col}_c1'],
                            color=trace_colors[c1k], alpha=0.7, label=c1k)
            axes[0, i].plot(raw.loc[w, 'time'], raw.loc[w, f'{col}_c2'],
                            color=trace_colors[c2k], alpha=0.7, label=c2k)
            axes[1, i].plot(filt.loc[w, 'time'], filt.loc[w, f'{col}_c1'],
                            color=trace_colors[c1k], label=f'Filtered {c1k}')
            axes[1, i].plot(filt.loc[w, 'time'], filt.loc[w, f'{col}_c2'],
                            color=trace_colors[c2k], label=f'Filtered {c2k}')

        axes[0, i].set_title(f"Raw {label} ACCL Channels ({t_range[0]}s–{t_range[1]}s)")
        axes[1, i].set_title(f"Filtered {label} ACCL Channels ({t_range[0]}s–{t_range[1]}s)")
        for r in range(2):
            axes[r, i].legend(loc='upper right', fontsize='small', ncol=2)

    for ax in axes.flat:
        ax.set_ylabel('Acceleration')
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# D.2.1.6  Identification of Takeoff  (Algorithm 3.1)
# ─────────────────────────────────────────────────────────────

def detect_imu_takeoff(df, threshold=0, samples=2020):
    """
    Algorithm 3.1 — sustained acceleration takeoff detection.

    Returns the sample index where sustained forward acceleration begins.
    """
    avg_acc = (df['ACCL_0_c1'] + df['ACCL_0_c2']) / 2
    is_high = (avg_acc > threshold).astype(int)
    count_high = is_high.rolling(window=samples).sum()
    hits = np.where(count_high == samples)[0]
    return hits[0] - samples + 1 if len(hits) else None
