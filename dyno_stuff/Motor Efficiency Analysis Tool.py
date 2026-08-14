#!/usr/bin/env python3
"""
motor_efficiency.py
===================

Compute electric-motor efficiency by combining dyno (mechanical output) and
MoTeC (electrical input) log data, then visualise the result as interactive
3-D plots.

This is a cleaned-up, runnable version of the "Fix 1" workflow from the
`20_March_Efficiency` notebook. The pipeline:

    1.  Load each dyno CSV and convert axle speed/torque to motor speed/torque
        using the 43/11 final-drive ratio.
    2.  Load the matching MoTeC CSV, compute battery (DC) power, and keep only
        the samples that fall inside the dyno's RPM window while the motor is
        loaded.
    3.  Pick the single continuous MoTeC segment whose RPM+torque profile best
        matches the dyno trace (`_best_segment`).
    4.  Optionally correct the dyno trace with a multiplicative scale factor
        (assumes a gear-ratio error that conserves shaft power).
    5.  Nearest-neighbour match MoTeC samples to dyno samples on RPM, compute
        mechanical output power vs. electrical input power, and hence efficiency.
    6.  Plot efficiency as an interactive 3-D scatter and as a fitted surface.

Run `python motor_efficiency.py --help` for options.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Where the raw CSVs live. Change these to match your folder layout.
DYNO_DATA_DIR = Path("Dyno_Data")
MOTEC_DATA_DIR = Path("Motec_Data/CSV Export")

# Final-drive ratio used to convert axle <-> motor quantities.
GEAR_RATIO = 43 / 11

# Column names in the MoTeC export.
MOTEC_RPM = "Car.Data.Motor.MotorRPM"
MOTEC_TORQUE = "Car.Data.Inverter.InverterCalculatedTorque"
MOTEC_VOLTAGE = "Car.Data.Inverter.InverterDCVoltage"
MOTEC_CURRENT = "Car.Data.Inverter.InverterDCCurrent"

# Column names we derive on the dyno frame.
DYNO_RPM = "Calculated Motor RPM"
DYNO_TORQUE = "Calculated Motor Torque"


# --------------------------------------------------------------------------- #
# Fix 1: helpers for aligning the dyno trace to the MoTeC trace
# --------------------------------------------------------------------------- #

def parse_torque_from_filename(filename: str):
    """Return the ``plNN`` power-level number from a filename, or None."""
    match = re.search(r"pl(\d+)", filename)
    return int(match.group(1)) if match else None


def estimate_scale_factors(
    cur_dyno_df,
    motec_seg,
    dyno_rpm_col,
    motec_rpm_col,
    dyno_torque_col,
    motec_torque_col,
    assume_gear_ratio_error=True,
):
    """
    Estimate multiplicative scale factors to align the dyno data to MoTeC.

    If ``assume_gear_ratio_error`` is True the torque scale is forced to be
    ``1 / rpm_scale`` so that shaft power is conserved (the physically correct
    behaviour for a gear-ratio error).

    Otherwise the torque gets an independent scale factor estimated from the
    90th-percentile torque ratio.
    """
    dyno_rpm_max = cur_dyno_df[dyno_rpm_col].max()
    motec_rpm_max = motec_seg[motec_rpm_col].max()

    # Scale so that dyno_corrected = dyno * scale  ~=  motec
    rpm_scale = motec_rpm_max / dyno_rpm_max

    if assume_gear_ratio_error:
        # Gear-ratio error: torque is the exact inverse of the RPM scale.
        torque_scale = 1.0 / rpm_scale
    else:
        # Independent sensor/calibration error: estimate from 90th percentile.
        dyno_tq_90 = cur_dyno_df[dyno_torque_col].quantile(0.90)
        motec_tq_90 = motec_seg[motec_torque_col].quantile(0.90)
        torque_scale = motec_tq_90 / dyno_tq_90

    print(
        f"Scale factors -> RPM: {rpm_scale:.6f},  Torque: {torque_scale:.6f}  "
        f"(RPM x Torque = {rpm_scale * torque_scale:.6f}, "
        f"should be 1.0 if gear-ratio error)"
    )
    return rpm_scale, torque_scale


def estimate_constant_offsets(
    cur_dyno_df,
    motec_df,
    dyno_rpm_col,
    motec_rpm_col,
    dyno_torque_col,
    motec_torque_col,
):
    """
    Additive-offset alternative to :func:`estimate_scale_factors`.

    Matches the peaks of both datasets, which is robust against long periods
    of idling / zero data in the MoTeC log. Not used by default (the
    multiplicative correction is preferred) but kept as an alternative.
    """
    dyno_rpm_max = cur_dyno_df[dyno_rpm_col].max()
    motec_rpm_max = motec_df[motec_rpm_col].max()
    rpm_offset = dyno_rpm_max - motec_rpm_max

    torque_offset = (
        cur_dyno_df[dyno_torque_col].quantile(0.90)
        - motec_df[motec_torque_col].quantile(0.90)
    )

    print(f"Calculated offsets -> RPM: {rpm_offset:.1f}, Torque: {torque_offset:.2f}")
    return rpm_offset, torque_offset


def _best_segment(
    motec_in_range: pd.DataFrame,
    dyno_df: pd.DataFrame,
    gap_threshold_s: float = 2.0,
    rpm_weight: float = 1.0,
    torque_weight: float = 1.0,
):
    """
    Split the MoTeC data into continuous segments and choose the segment whose
    RPM + torque profile best matches the dyno trace.

    Matching interpolates both traces onto a normalised (0-1) time base and
    minimises the combined RMSE.
    """
    df = motec_in_range.copy().reset_index(drop=True)

    # Split into continuous segments wherever the time gap exceeds the threshold.
    df["_seg"] = (df["Time"].diff() > gap_threshold_s).cumsum()

    # ---- Dyno normalised time axis ----
    dyno_time = dyno_df["Time (sec)"].values
    dyno_time = dyno_time - dyno_time.min()
    dyno_rpm = dyno_df[DYNO_RPM].values
    dyno_torque = dyno_df[DYNO_TORQUE].values
    dyno_t_norm = dyno_time / dyno_time.max()

    best_score = np.inf
    best_seg = None

    for seg, grp in df.groupby("_seg"):
        if len(grp) < 10:
            continue

        motec_time = grp["Time"].values
        motec_time = motec_time - motec_time.min()
        if motec_time.max() <= 0:
            continue
        motec_t_norm = motec_time / motec_time.max()

        # Interpolate the MoTeC segment onto the dyno's normalised timeline.
        try:
            rpm_interp = interp1d(
                motec_t_norm, grp[MOTEC_RPM].values,
                bounds_error=False, fill_value="extrapolate",
            )
            torque_interp = interp1d(
                motec_t_norm, grp[MOTEC_TORQUE].values,
                bounds_error=False, fill_value="extrapolate",
            )
            motec_rpm_resampled = rpm_interp(dyno_t_norm)
            motec_torque_resampled = torque_interp(dyno_t_norm)
        except Exception:
            continue

        rpm_rmse = np.sqrt(np.mean((motec_rpm_resampled - dyno_rpm) ** 2))
        torque_rmse = np.sqrt(np.mean((motec_torque_resampled - dyno_torque) ** 2))
        score = rpm_weight * rpm_rmse + torque_weight * torque_rmse

        print(
            f"Segment {seg}: RPM RMSE={rpm_rmse:.1f}, "
            f"Torque RMSE={torque_rmse:.1f}, Score={score:.1f}"
        )

        if score < best_score:
            best_score = score
            best_seg = seg

    print(f"Selected segment: {best_seg} (score={best_score:.1f})")
    return df[df["_seg"] == best_seg].drop(columns="_seg")


# --------------------------------------------------------------------------- #
# Fix 1: main efficiency pipeline
# --------------------------------------------------------------------------- #

def process_and_compute_efficiency(
    dyno_files,
    motec_files,
    offset,
    rpm_diff_tol=5,
    rolling_window=3,
    default_torque_threshold=0.0,
    plot=True,
):
    """
    Process dyno + MoTeC data and compute motor efficiency (the "Fix 1"
    workflow).

    Parameters
    ----------
    dyno_files : list[str]
        Dyno CSV filenames (relative to ``DYNO_DATA_DIR``).
    motec_files : list[str]
        MoTeC CSV filenames (relative to ``MOTEC_DATA_DIR``), paired with
        ``dyno_files`` by position. NOTE: the two lists are zipped, so the run
        count is the length of the *shorter* list.
    offset : list
        Per-run flags. A truthy value at index ``i`` applies the multiplicative
        scale correction to run ``i``.
    rpm_diff_tol : float
        Only keep nearest-neighbour matches within +/- this many RPM.
    rolling_window : int
        Window (in samples) for the rolling-mean smoothing of the MoTeC trace.
    default_torque_threshold : float
        Torque threshold used if a power level can't be parsed from a filename.
    plot : bool
        If True, draw the per-run alignment plot and the final 2-D scatter.

    Returns
    -------
    pandas.DataFrame
        Combined per-sample efficiency results across all runs. Empty if no
        run produced data.
    """
    all_runs = []

    for run_idx, (dyno_file, motec_file) in enumerate(zip(dyno_files, motec_files)):
        print(f"\n--- Run {run_idx}: {dyno_file}  <->  {motec_file} ---")

        # --- Load dyno and convert axle -> motor quantities ---
        cur_dyno_df = pd.read_csv(DYNO_DATA_DIR / dyno_file)
        cur_dyno_df = cur_dyno_df.dropna(subset=["Axle Speed (rpm)", "Axle Torque (Nm)"])
        cur_dyno_df[DYNO_RPM] = cur_dyno_df["Axle Speed (rpm)"] * GEAR_RATIO
        cur_dyno_df[DYNO_TORQUE] = cur_dyno_df["Axle Torque (Nm)"] / GEAR_RATIO

        # --- Load MoTeC and compute battery (DC) power ---
        motec_df = pd.read_csv(MOTEC_DATA_DIR / motec_file, low_memory=False)
        motec_df = motec_df.iloc[1:].reset_index(drop=True)          # drop units row
        motec_df = motec_df.apply(pd.to_numeric, errors="coerce")
        motec_df["Battery Power (kW)"] = (
            motec_df[MOTEC_VOLTAGE] * motec_df[MOTEC_CURRENT] / 100_000
        )

        if not (motec_df[MOTEC_CURRENT] > 0).any():
            print("  WARNING: no positive DC current detected. Skipping.")
            continue

        # --- Keep MoTeC samples inside the dyno RPM window, under load ---
        rpm_min = max(cur_dyno_df[DYNO_RPM].min(), 0)
        rpm_max = cur_dyno_df[DYNO_RPM].max()
        run_torque = parse_torque_from_filename(dyno_file)
        torque_threshold = run_torque * 0.85 if run_torque else default_torque_threshold

        motec_in_range = motec_df[
            (motec_df[MOTEC_RPM] >= rpm_min)
            & (motec_df[MOTEC_RPM] <= rpm_max)
            & (motec_df[MOTEC_TORQUE] >= torque_threshold)
            & (motec_df["Battery Power (kW)"] > 1)
        ].copy()

        # --- Choose the best-matching continuous MoTeC segment ---
        motec_seg = _best_segment(
            motec_in_range, cur_dyno_df, rpm_weight=1.0, torque_weight=0.5,
        )
        print(
            f"  Best MoTeC segment: {len(motec_seg)} rows, "
            f"duration {motec_seg['Time'].max() - motec_seg['Time'].min():.1f} s, "
            f"RPM {motec_seg[MOTEC_RPM].min():.0f}-{motec_seg[MOTEC_RPM].max():.0f}"
        )

        # --- Optional multiplicative correction of the dyno trace ---
        og_rpm = cur_dyno_df[DYNO_RPM].copy()
        og_tq = cur_dyno_df[DYNO_TORQUE].copy()

        if offset[run_idx]:
            rpm_scale, torque_scale = estimate_scale_factors(
                cur_dyno_df, motec_seg,
                dyno_rpm_col=DYNO_RPM, motec_rpm_col=MOTEC_RPM,
                dyno_torque_col=DYNO_TORQUE, motec_torque_col=MOTEC_TORQUE,
                assume_gear_ratio_error=True,  # set False for independent torque scale
            )
            cur_dyno_df[DYNO_RPM] *= rpm_scale
            cur_dyno_df[DYNO_TORQUE] *= torque_scale

            if plot:
                _plot_alignment(cur_dyno_df, motec_seg, og_rpm, og_tq)

        # --- Keep only the monotonically-increasing RPM portion, smoothed ---
        motec_mono = motec_seg[motec_seg[MOTEC_RPM].diff() > 0]
        motec_mono = motec_mono.rolling(rolling_window).mean().dropna()
        print(f"  After monotonic + rolling mean: {len(motec_mono)} rows")
        if motec_mono.empty:
            print("  No monotonic data - skipping.")
            continue

        # --- Nearest-neighbour match on RPM only ---
        # InverterCalculatedTorque saturates at a command limit (not real output
        # torque) so it cannot be used as a matching feature.
        dyno_rpm = cur_dyno_df[DYNO_RPM].to_numpy().reshape(-1, 1)
        motec_rpm = motec_mono[MOTEC_RPM].to_numpy().reshape(-1, 1)

        scaler = StandardScaler()
        dyno_scaled = scaler.fit_transform(dyno_rpm)
        motec_scaled = scaler.transform(motec_rpm)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(dyno_scaled)
        _, indices = nn.kneighbors(motec_scaled)

        matched_dyno = cur_dyno_df.iloc[indices.flatten()].reset_index(drop=True)
        matched_motec = motec_mono.reset_index(drop=True)

        # RPM proximity filter
        rpm_diff = np.abs(
            matched_dyno[DYNO_RPM].values - matched_motec[MOTEC_RPM].values
        )
        keep = rpm_diff <= rpm_diff_tol
        matched_dyno = matched_dyno[keep].reset_index(drop=True)
        matched_motec = matched_motec[keep].reset_index(drop=True)
        print(f"  After RPM tolerance (+/-{rpm_diff_tol:.0f}): {len(matched_motec)} rows")
        if matched_motec.empty:
            print("  Skipping - no rows pass RPM tolerance.")
            continue

        # --- Efficiency = mechanical out / electrical in ---
        result = matched_motec.copy()
        result["Motor RPM (dyno)"] = matched_dyno[DYNO_RPM].values
        result["Motor Torque (dyno)"] = matched_dyno[DYNO_TORQUE].values
        result["P_out_kW"] = (
            result["Motor Torque (dyno)"] * 2 * np.pi
            * result["Motor RPM (dyno)"] / 60 / 1000
        )
        result["P_in_kW"] = matched_motec["Battery Power (kW)"].values
        result["eff"] = result["P_out_kW"] / result["P_in_kW"]

        print(
            f"  P_out {result['P_out_kW'].min():.1f}-{result['P_out_kW'].max():.1f} kW  "
            f"P_in {result['P_in_kW'].min():.1f}-{result['P_in_kW'].max():.1f} kW  "
            f"eff {result['eff'].min():.3f}-{result['eff'].max():.3f}"
        )
        result["run"] = run_idx
        all_runs.append(result)

    if not all_runs:
        print("\nNo runs produced data.")
        return pd.DataFrame()

    # Combine all runs and keep physically plausible efficiencies (0 < eff < 1).
    combined_df = pd.concat(all_runs, ignore_index=True)
    combined_df = combined_df[(combined_df["eff"] > 0) & (combined_df["eff"] < 1)]

    print(f"\n{'=' * 60}")
    print(f"Total valid efficiency points : {len(combined_df)}")
    print(f"Mean efficiency               : {combined_df['eff'].mean():.3f}")
    print(
        f"Efficiency range              : {combined_df['eff'].min():.3f} - "
        f"{combined_df['eff'].max():.3f}"
    )

    if plot and not combined_df.empty:
        plt.figure(figsize=(12, 8))
        plt.scatter(combined_df["Motor RPM (dyno)"], combined_df["eff"], s=10, alpha=0.7)
        plt.xlabel("Motor RPM")
        plt.ylabel("Efficiency")
        plt.title("Efficiency vs Motor RPM")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return combined_df


def _plot_alignment(cur_dyno_df, motec_seg, og_rpm, og_tq):
    """Diagnostic plot: dyno (raw + corrected) vs MoTeC on a normalised time axis."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    dyno_t_norm = (
        (cur_dyno_df["Time (sec)"] - cur_dyno_df["Time (sec)"].min())
        / (cur_dyno_df["Time (sec)"].max() - cur_dyno_df["Time (sec)"].min())
    )
    motec_t_norm = (
        (motec_seg["Time"] - motec_seg["Time"].min())
        / (motec_seg["Time"].max() - motec_seg["Time"].min())
    )

    # RPM traces (left axis)
    ax1.plot(dyno_t_norm, cur_dyno_df[DYNO_RPM],
             label="Dyno RPM Corrected", color="blue", alpha=0.7)
    ax1.plot(dyno_t_norm, og_rpm, label="Dyno RPM Raw", color="navy", linewidth=2)
    ax1.plot(motec_t_norm, motec_seg[MOTEC_RPM].values,
             label="MoTeC RPM", color="cyan", linestyle="--")
    ax1.set_ylabel("RPM")

    # Torque traces (right axis)
    ax2.plot(dyno_t_norm, cur_dyno_df[DYNO_TORQUE],
             label="Dyno Torque Corrected", color="red", alpha=0.7)
    ax2.plot(dyno_t_norm, og_tq, label="Dyno Torque Raw", color="darkred", linewidth=2)
    ax2.plot(motec_t_norm, motec_seg[MOTEC_TORQUE].values,
             label="MoTeC Torque", color="orange", linestyle="--")
    ax2.set_ylabel("Torque (Nm)")

    ax1.set_xlabel("Normalised time (0 = start, 1 = end of each recording)")
    ax1.set_title("Dyno vs MoTeC Alignment")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Interactive plots
# --------------------------------------------------------------------------- #

def _emit(fig, show=True, save_html=None):
    """Save a Plotly figure to HTML and/or open it in the browser."""
    if save_html:
        fig.write_html(save_html)
        print(f"Saved interactive plot to {save_html}")
    if show:
        try:
            fig.show(renderer="browser")
        except Exception as exc:  # headless / no browser available
            print(f"Could not open a browser ({exc}). Use the saved HTML file instead.")


def plot_interactive_efficiency(combined_df, show=True, save_html=None):
    """Interactive 3-D scatter of efficiency vs RPM and torque, coloured by run."""
    fig = px.scatter_3d(
        combined_df,
        x=MOTEC_RPM,
        y=MOTEC_TORQUE,
        z="eff",
        color="run",
        color_continuous_scale="RdYlGn",
        title="Interactive Motor Efficiency Map",
        labels={
            "eff": "Efficiency",
            MOTEC_RPM: "RPM",
            MOTEC_TORQUE: "Torque (Nm)",
        },
    )
    fig.update_traces(marker=dict(size=3))
    _emit(fig, show=show, save_html=save_html)
    return fig


def plot_interactive_efficiency_surface(combined_df, grid_num=100, show=True, save_html=None):
    """Interactive 3-D efficiency surface fitted to the scattered data."""
    raw_rpm = combined_df[MOTEC_RPM].values
    raw_torque = combined_df[MOTEC_TORQUE].values
    raw_eff = combined_df["eff"].values

    # Uniform grid + linear interpolation of the scattered efficiency data.
    grid_rpm = np.linspace(raw_rpm.min(), raw_rpm.max(), num=grid_num)
    grid_torque = np.linspace(raw_torque.min(), raw_torque.max(), num=grid_num)
    rpm_mesh, torque_mesh = np.meshgrid(grid_rpm, grid_torque)
    eff_mesh = griddata(
        (raw_rpm, raw_torque), raw_eff, (rpm_mesh, torque_mesh), method="linear",
    )

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=grid_rpm, y=grid_torque, z=eff_mesh,
        colorscale="RdYlGn", colorbar=dict(title="Efficiency"),
        name="Fitted Surface", opacity=0.85,
    ))
    fig.add_trace(go.Scatter3d(
        x=raw_rpm, y=raw_torque, z=raw_eff,
        mode="markers",
        marker=dict(size=2, color=combined_df["run"],
                    colorscale="Viridis", opacity=0.6),
        name="Actual Data",
    ))
    fig.update_layout(
        title="Interactive Motor Efficiency Map with Surface Fit",
        scene=dict(xaxis_title="RPM", yaxis_title="Torque (Nm)", zaxis_title="Efficiency"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    _emit(fig, show=show, save_html=save_html)
    return fig


# --------------------------------------------------------------------------- #
# Default datasets (edit to match the files you have)
# --------------------------------------------------------------------------- #

DYNO_MATCHED_PERF = [
    "speedramp2_pl60_manual_dhruv.csv",
    "speedramp3_pl70_manual_dhruv.csv",
    "speedramp4_pl80_manual_dhruv.csv",
    "speedramp5_pl90_manual_dhruv.csv",
    "speedramp6_pl100_manual_dhruv.csv",
    "speedramp7_pl110_manual_dhruv.csv",
    "speedramp8_pl120_manual_dhruv.csv",
    "speedramp9_pl130_manual_dhruv.csv",
    "speedramp10_pl140_manual_dhruv.csv",
]

MOTEC_MATCHED_PERF = [
    "pl60_1.csv",
    "pl70_1.csv",
    "pl80_1.csv",
    "pl90_1.csv",
    "pl100_1.csv",
    "pl110_1.csv",
    "pl120_1.csv",
    "pl130_1.csv",
]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    global DYNO_DATA_DIR, MOTEC_DATA_DIR

    parser = argparse.ArgumentParser(
        description="Compute and plot electric-motor efficiency from dyno + MoTeC logs."
    )
    parser.add_argument("--dyno-dir", type=Path, default=DYNO_DATA_DIR,
                        help=f"Folder with dyno CSVs (default: {DYNO_DATA_DIR})")
    parser.add_argument("--motec-dir", type=Path, default=MOTEC_DATA_DIR,
                        help=f"Folder with MoTeC CSVs (default: {MOTEC_DATA_DIR})")
    parser.add_argument("--rolling-window", type=int, default=3,
                        help="Rolling-mean window in samples (default: 3)")
    parser.add_argument("--rpm-tol", type=float, default=5,
                        help="RPM tolerance for NN matching (default: 5)")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Skip the per-run matplotlib alignment/scatter plots.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open the interactive plots in a browser.")
    parser.add_argument("--save-html", action="store_true",
                        help="Save the two interactive plots as HTML files.")
    args = parser.parse_args()

    # Allow overriding the data directories from the command line.
    DYNO_DATA_DIR = args.dyno_dir
    MOTEC_DATA_DIR = args.motec_dir

    offset = [1] * len(DYNO_MATCHED_PERF)

    combined_df = process_and_compute_efficiency(
        DYNO_MATCHED_PERF,
        MOTEC_MATCHED_PERF,
        offset,
        rpm_diff_tol=args.rpm_tol,
        rolling_window=args.rolling_window,
        plot=not args.no_diagnostics,
    )

    if combined_df.empty:
        print("No data to plot.")
        return

    show = not args.no_show
    plot_interactive_efficiency(
        combined_df, show=show,
        save_html="efficiency_scatter.html" if args.save_html else None,
    )
    plot_interactive_efficiency_surface(
        combined_df, show=show,
        save_html="efficiency_surface.html" if args.save_html else None,
    )


if __name__ == "__main__":
    main()
