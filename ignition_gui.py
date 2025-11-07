"""
Ignition Data GUI – Streamlit (v4: MANUAL burn markers, hardened)

- CSV or TDMS upload
- Build time from sample rate OR use existing time column
- Trim window; plot/export use ZERO-BASED time inside the trim
- Multi-channel plot with smoothing
- Optional ignition flag (manual/threshold/existing)
- Manual burn START/STOP markers (range-slider + inputs)
- Distinct colors in Kaleido PNG export
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import traceback

# ---------------------------
# Optional dependency (TDMS)
# ---------------------------
try:
    from nptdms import TdmsFile
    _HAS_TDMS = True
except Exception:
    TdmsFile = None
    _HAS_TDMS = False


# ---------------------------
# Helpers
# ---------------------------
@dataclass
class DataSpec:
    time_col: Optional[str]
    signal_cols: List[str]


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert a time-like column to seconds since its first entry."""
    s = series.copy()
    if np.issubdtype(s.dtype, np.number):
        s = s.astype(float)
        return s - float(s.iloc[0])
    try:
        s_dt = pd.to_datetime(s, errors="raise")
        base = s_dt.iloc[0]
        return (s_dt - base).dt.total_seconds()
    except Exception:
        s_num = pd.to_numeric(s, errors="coerce").fillna(method="ffill").fillna(method="bfill")
        return s_num - float(s_num.iloc[0])


def _time_from_samplerate(n: int, fs: float, index=None) -> pd.Series:
    t = np.arange(n, dtype=float) / float(max(fs, 1e-9))
    return pd.Series(t, index=index)


def _moving_average(x: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    return x.rolling(window, center=True, min_periods=1).mean()


def _load_csv(file, sep: Optional[str], header_row: bool, skiprows: int,
              engine: str, encoding: str, on_bad_lines: str, auto_detect: bool) -> pd.DataFrame:
    """Robust CSV reader with delimiter auto-detect, skiprows, encoding, and bad-line handling."""
    if auto_detect:
        use_sep = None
        use_engine = 'python'
    else:
        use_sep = sep
        use_engine = engine

    hdr = 0 if header_row else None
    df = pd.read_csv(
        file,
        sep=use_sep,
        header=hdr,
        skiprows=skiprows if skiprows > 0 else None,
        engine=use_engine,
        encoding=encoding,
        on_bad_lines=on_bad_lines if use_engine == 'python' else None,
    )
    return df


def _load_tdms(file) -> pd.DataFrame:
    if not _HAS_TDMS:
        raise RuntimeError("nptdms not installed. pip install nptdms")
    tf = TdmsFile.read(file)
    frames = []
    for group in tf.groups():
        for ch in group.channels():
            name = f"{group.name}/{ch.name}"
            frames.append(pd.DataFrame({name: ch[:]}))
    df = pd.concat(frames, axis=1)
    return df


def _infer_time_and_signals(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    candidates = [c for c in df.columns if c.lower() in {"time", "timestamp", "t", "seconds", "time_s"}]
    time_col = candidates[0] if candidates else None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    signal_cols = [c for c in numeric_cols if c != time_col] if time_col else numeric_cols
    if not signal_cols:
        others = [c for c in df.columns if c != (time_col or "")]
        signal_cols = others[:8]
    return time_col, signal_cols


def _compute_ignition_flag(df: pd.DataFrame, t: pd.Series, method: str,
                           manual_time: Optional[float] = None,
                           threshold_channel: Optional[str] = None,
                           threshold_value: Optional[float] = None,
                           direction: str = "rising") -> Tuple[pd.Series, Optional[float]]:
    """Returns (boolean series flag, ignition_time_in_seconds or None)."""
    ignition_time = None

    if method == "manual":
        if manual_time is None:
            ignition = pd.Series(False, index=df.index)
        else:
            ignition_time = float(manual_time)
            ignition = t >= ignition_time
        return ignition, ignition_time

    if method == "existing" and "ignition_flag" in df.columns:
        ignition = df["ignition_flag"].astype(int) > 0
        idx = ignition.idxmax() if ignition.any() else None
        ignition_time = float(t.loc[idx]) if idx is not None else None
        return ignition, ignition_time

    if method == "threshold" and threshold_channel in df.columns:
        y = pd.to_numeric(df[threshold_channel], errors="coerce").fillna(method="ffill").fillna(method="bfill")
        if direction == "rising":
            mask = (y.shift(1) < threshold_value) & (y >= threshold_value)
        else:
            mask = (y.shift(1) > threshold_value) & (y <= threshold_value)
        idx = mask.idxmax() if mask.any() else None
        if idx is not None and mask.loc[idx]:
            ignition_time = float(t.loc[idx])
            ignition = t >= ignition_time
        else:
            ignition = pd.Series(False, index=df.index)
        return ignition, ignition_time

    return pd.Series(False, index=df.index), None


def _make_plot(df: pd.DataFrame, t: np.ndarray, signals: List[str], legend_map: Dict[str, str],
               window: Optional[int], ignition_time_rel: Optional[float],
               title: str, subtitle: Optional[str],
               burn_markers: Optional[Tuple[float, float]] = None) -> go.Figure:
    """Plot multiple channels against zero-based time array; show burn markers region if provided."""
    fig = go.Figure()

    for i, col in enumerate(signals):
        y = df[col]
        if window and window > 1:
            y = _moving_average(y, window)
        y_np = pd.to_numeric(y, errors="coerce").to_numpy()
        fig.add_trace(go.Scatter(x=t, y=y_np, mode='lines', name=legend_map.get(col, col)))

    if ignition_time_rel is not None:
        fig.add_vline(x=ignition_time_rel, line_width=2, line_dash="dash",
                      annotation_text="Ignition", annotation_position="top right")

    if burn_markers is not None:
        b0, b1 = burn_markers
        if np.isfinite(b0) and np.isfinite(b1) and b1 >= b0:
            fig.add_vrect(x0=b0, x1=b1, opacity=0.2, line_width=0,
                          annotation_text=f"Burn: {b1 - b0:.6f} s",
                          annotation_position="top left")
            fig.add_vline(x=b0, line_width=1, line_dash="dot")
            fig.add_vline(x=b1, line_width=1, line_dash="dot")

    full_title = title if title else "Signals vs Time"
    if subtitle:
        full_title += f"<br><sup>{subtitle}</sup>"

    # distinct colors in kaleido export
    fig.update_layout(
        title=full_title,
        xaxis_title="Time (s)",
        yaxis_title="Signal",
        legend_title="Channels",
        margin=dict(l=40, r=10, t=70, b=40),
        height=560,
        template="plotly_white",
        colorway=["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A",
                  "#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"],
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ===========================
# App
# ===========================
def run():
    st.set_page_config(page_title="Post Process GUI", layout="wide")

    # Optional logo (won't crash if missing)
    if Path("logo.png").exists():
        st.image("logo.png", width=140)

    st.title("Post Process GUI")
    st.caption("Upload CSV/TDMS → choose columns → build time from Fs if needed → trim → set burn start/stop → export")

    with st.sidebar:
        st.header("1) Load data")
        kind = st.radio("File type", ["CSV", "TDMS"], horizontal=True)
        file = st.file_uploader("Upload file", type=(['csv'] if kind == 'CSV' else ['tdms']))

        df: Optional[pd.DataFrame] = None
        if file is not None:
            if kind == "CSV":
                st.subheader("CSV options")
                auto_detect = st.checkbox("Auto-detect delimiter", value=True)
                sep = st.selectbox("Delimiter (if not auto)", options=[",", ";", "\t", " ", "|"], index=0)
                header_has_row = st.checkbox("First row is header", value=True)
                skiprows = st.number_input("Rows to skip at top (metadata)", min_value=0, value=0, step=1)
                encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
                engine = st.selectbox("Engine", ["c", "python"], index=0)
                on_bad = st.selectbox("On bad lines (python engine)", ["error", "warn", "skip"], index=2)
                try:
                    df = _load_csv(file, sep=sep, header_row=header_has_row, skiprows=skiprows,
                                   engine=engine, encoding=encoding, on_bad_lines=on_bad, auto_detect=auto_detect)
                    df = df.reset_index(drop=True)
                    # CLEAN & CONVERT (prevents channel picker crashes)
                    df.columns = df.columns.astype(str).str.strip()
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df.apply(pd.to_numeric, errors='ignore')
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
            else:
                if not _HAS_TDMS:
                    st.warning("nptdms not installed. Run: pip install nptdms")
                try:
                    df = _load_tdms(file)
                    df = df.reset_index(drop=True)
                    df.columns = df.columns.astype(str).str.strip()
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df.apply(pd.to_numeric, errors='ignore')
                except Exception as e:
                    st.error(f"Failed to read TDMS: {e}")

        st.divider()
        st.header("2) Time base")
        time_mode = st.radio("Use time from:", ["Sample rate (Fs)", "Existing time column"], index=0,
                             help="Your files are samples vs data; pick Fs to build time axis")
        fs = st.number_input("Sample rate (Hz)", min_value=1.0, value=12497.0, step=1.0)

        time_col = None
        if df is not None and time_mode == "Existing time column":
            time_guess, _ = _infer_time_and_signals(df)
            if time_guess is None:
                st.warning("No obvious time column found; falling back to sample rate.")
                time_mode = "Sample rate (Fs)"
            else:
                time_col = st.selectbox("Time column", options=list(df.columns), index=list(df.columns).index(time_guess))

        st.divider()
        st.header("3) Channels")
        if df is not None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            options = numeric_cols if numeric_cols else list(df.columns)
            if len(options) == 0:
                st.error("No columns detected. Check 'First row is header' and delimiter.")
                st.stop()
            quick = st.selectbox("Quick pick one channel", options=options, index=0)
            default_selection = [quick] if quick else []
            signal_cols = st.multiselect("Channels to plot (multi-select)", options=options,
                                         default=default_selection, help="Hold Ctrl/Cmd to select multiple")
            if len(signal_cols) == 0:
                st.warning("No channels selected. Pick at least one to plot.")
        else:
            signal_cols = []

        # Optional helper: reference channel (purely visual help)
        st.divider()
        st.header("Burn detection")
        if df is not None and signal_cols:
            burn_channel_select = st.selectbox(
                "Channel to reference while choosing burn markers",
                options=signal_cols, index=0,
                help="Markers are manual; this is just a visual reference."
            )
        else:
            burn_channel_select = None

        st.divider()
        st.header("4) Smoothing")
        window = st.number_input("Moving-average window (samples)", min_value=1, max_value=10001, value=1, step=1)

        st.divider()
        st.header("5) Ignition flag (optional)")
        ign_method = st.radio("Method", ["manual", "threshold", "existing"],
                              help="Manual: pick time; Threshold: detect crossing; Existing: use 'ignition_flag' column")
        ign_time_in = None
        thr_chan = None
        thr_val = None
        thr_dir = "rising"

        if ign_method == "manual":
            ign_time_in = st.number_input("Ignition time (s)", min_value=0.0, value=0.0)
        elif ign_method == "threshold":
            if df is not None:
                thr_chan = st.selectbox("Channel for threshold", options=signal_cols or list(df.columns))
            thr_val = st.number_input("Threshold value", value=0.0)
            thr_dir = st.radio("Direction", ["rising", "falling"], horizontal=True)

    st.divider()

    if df is None:
        st.info("Upload a file to begin.")
        st.stop()

    # Build time axis (absolute), then trim & convert to zero-based
    t_secs = _time_from_samplerate(len(df), fs, index=df.index) if time_mode == "Sample rate (Fs)" else _to_seconds(df[time_col]).set_axis(df.index)

    # Plot customization
    st.subheader("Plot customization")
    colA, colB = st.columns([2, 1])
    with colA:
        chart_title = st.text_input("Chart title", value="Signals vs Time")
    with colB:
        exp_date = st.date_input("Experiment date")

    # Legend labels
    legend_map: Dict[str, str] = {}
    with st.expander("Legend labels (rename per channel)"):
        for ch in signal_cols:
            legend_map[ch] = st.text_input(f"Label for '{ch}'", value=ch, key=f"legend_{ch}")

    # Ignition flag compute (on absolute timebase)
    ign_flag, detected_time = _compute_ignition_flag(
        df, t=t_secs, method=ign_method,
        manual_time=ign_time_in, threshold_channel=thr_chan,
        threshold_value=thr_val, direction=thr_dir,
    )
    _df = df.copy()
    _df["ignition_flag"] = ign_flag.astype(int)

    # Time window (absolute), then convert to zero-based inside cut
    st.subheader("Time window")
    left, right = st.columns([3, 2])
    with left:
        t0, t1 = float(t_secs.min()), float(t_secs.max())
        w = st.slider("Select range (s)", min_value=t0, max_value=t1, value=(t0, t1))
    with right:
        st.write("Manual cut (precise)")
        man_tmin = st.number_input("Start (s)", value=w[0], step=0.001, format="%.6f")
        man_tmax = st.number_input("End (s)", value=w[1], step=0.001, format="%.6f")

    final_tmin = float(man_tmin)
    final_tmax = float(man_tmax)

    # Cut mask & zero-based time
    t_np = t_secs.to_numpy()
    cut_mask = (t_np >= final_tmin) & (t_np <= final_tmax)
    t_cut = t_np[cut_mask]
    if t_cut.size == 0:
        st.error("Selected time window is empty. Adjust the range.")
        st.stop()
    t_rel = t_cut - float(t_cut[0])  # zero-based time within trimmed window

    # Ignition relative marker (if inside window)
    ign_rel = None
    if ign_method == "manual":
        ign_rel = float(ign_time_in - final_tmin) if ign_time_in is not None and (final_tmin <= ign_time_in <= final_tmax) else None
    else:
        if detected_time is not None and (final_tmin <= detected_time <= final_tmax):
            ign_rel = float(detected_time - final_tmin)

    # ---------------------------
    # MANUAL Burn markers
    # ---------------------------
    st.subheader("Burn markers (manual)")
    bcol1, bcol2 = st.columns([3, 2])

    default_b0 = 0.0
    default_b1 = float(t_rel[-1])

    with bcol1:
        burn_range = st.slider(
            "Drag to set burn START and STOP (s, relative to trimmed window)",
            min_value=0.0, max_value=float(t_rel[-1]),
            value=(default_b0, default_b1), step=max(float((t_rel[-1]) / 2000.0), 1e-6),
            help="Duration = STOP - START."
        )

    with bcol2:
        b0 = st.number_input("Burn START (s, relative)", value=burn_range[0], min_value=0.0, max_value=float(t_rel[-1]),
                             step=0.000001, format="%.6f")
        b1 = st.number_input("Burn STOP (s, relative)",  value=burn_range[1], min_value=0.0, max_value=float(t_rel[-1]),
                             step=0.000001, format="%.6f")
        if b1 < b0:
            b0, b1 = b1, b0

    burn_duration = float(max(0.0, b1 - b0))
    burn_markers = (float(b0), float(b1))

    # Plot (zero-based) with manual burn region
    fig = _make_plot(
        _df.loc[cut_mask, :],
        t=t_rel,
        signals=signal_cols,
        legend_map=legend_map,
        window=window,
        ignition_time_rel=ign_rel,
        title=chart_title,
        subtitle=str(exp_date) if exp_date else None,
        burn_markers=burn_markers
    )

    st.plotly_chart(fig, use_container_width=True)

    # Info + preview
    info_left, info_right = st.columns([1, 2])
    with info_left:
        st.markdown(f"**Samples (cut):** {int(cut_mask.sum())}")
        st.markdown(f"**Duration (cut):** {t_rel[-1]:.6f} s")
        if ign_rel is not None:
            st.markdown(f"**Ignition @** {ign_rel:.6f} s (relative)")
        if burn_channel_select:
            st.markdown(f"**Reference channel:** {burn_channel_select}")
        st.markdown(f"**Burn START:** {b0:.6f} s")
        st.markdown(f"**Burn STOP:** {b1:.6f} s")
        st.markdown(f"**Burn time:** {burn_duration:.6f} s")
    with info_right:
        preview_cols = signal_cols + ["ignition_flag"]
        preview_df = _df.loc[cut_mask, preview_cols].copy()
        preview_df.insert(0, "time_s_rel", t_rel)
        preview_df["burn_flag"] = ((t_rel >= b0) & (t_rel <= b1)).astype(int)
        st.dataframe(preview_df.head(50), use_container_width=True)

    # ---------------------------
    # Exports
    # ---------------------------
    st.divider()
    st.subheader("Export (uses zero-based time)")

    cut_df = _df.loc[cut_mask, signal_cols].copy()
    cut_df.insert(0, "time_s_rel", t_rel)
    cut_df["ignition_flag"] = _df.loc[cut_mask, "ignition_flag"].values
    cut_df["burn_flag"] = ((t_rel >= b0) & (t_rel <= b1)).astype(int)

    csv_bytes = cut_df.to_csv(index=False).encode("utf-8")
    base_name = (chart_title or "signals_plot").strip().replace(" ", "_")

    st.download_button("Download cut CSV", data=csv_bytes, file_name=f"{base_name}_cut.csv", mime="text/csv")

    try:
        png_bytes = fig.to_image(format="png", width=1400, height=650, scale=2)
        st.download_button("Download plot PNG", data=png_bytes, file_name=f"{base_name}.png", mime="image/png")
    except Exception as e:
        with st.expander("PNG export troubleshooting"):
            st.warning("PNG export requires 'kaleido'. Add `kaleido` to requirements.txt.")
            st.code(str(e))

    # Session meta
    import json
    meta = {
        "chart_title": chart_title,
        "experiment_date": str(exp_date) if exp_date else None,
        "time_mode": "Fs" if time_mode == "Sample rate (Fs)" else "existing_col",
        "fs_hz": float(fs) if time_mode == "Sample rate (Fs)" else None,
        "ignition_method": ign_method,
        "ignition_time_s_rel": ign_rel,
        "time_window_s_abs": [final_tmin, final_tmax],
        "duration_s_rel": float(t_rel[-1]),
        "channels": signal_cols,
        "legend_map": legend_map,
        "burn_markers_s_rel": {"start": float(b0), "stop": float(b1), "duration": burn_duration},
        "reference_channel": burn_channel_select
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    st.download_button("Download session meta (JSON)", data=meta_bytes, file_name=f"{base_name}_meta.json", mime="application/json")

    st.caption("© Spacefield – Local analysis utility. Built with Streamlit.")


# Run with error surfacing in the UI
try:
    run()
except Exception:
    st.error("Unexpected error in app:")
    st.code("".join(traceback.format_exc()))
