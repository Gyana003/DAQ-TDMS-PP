"""
Ignition Data GUI – Streamlit single-file app (v2)

New features (per your request):
- Channel picker with multi-select (plot many channels on ONE graph).
- Files are samples vs data → option to build **Time (s)** from a known sample rate.
  • default Fs = **12497 samples/sec** (editable).
- Manual time-frame cut via start/end **numeric inputs** (besides the slider).
- Customization: chart title, experiment date, and **legend labels per channel**.
- Upload CSV or TDMS; smoothing; ignition flag (manual/threshold/existing); export cut CSV & plot PNG.

How to run locally:
1) pip install streamlit pandas numpy plotly nptdms kaleido
2) Save as ignition_gui.py
3) streamlit run ignition_gui.py
"""

import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

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
    """Best-effort convert x-axis to seconds from an existing time-like column."""
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


def _load_csv(file, sep: Optional[str], header_row: bool, skiprows: int, engine: str, encoding: str, on_bad_lines: str, auto_detect: bool) -> pd.DataFrame:
    """Robust CSV reader with delimiter auto-detect, skiprows, encoding, and bad-line handling."""
    if auto_detect:
        # sep=None + engine='python' triggers csv.Sniffer auto-detect
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

    if hdr is None:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
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
    if not any(c.lower().startswith("time") for c in df.columns):
        # leave as samples; we'll generate time from Fs
        pass
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


def _make_plot(df: pd.DataFrame, t: pd.Series, signals: List[str], legend_map: Dict[str, str],
               window: Optional[int], tmin: Optional[float], tmax: Optional[float],
               ignition_time: Optional[float], title: str, subtitle: Optional[str]):
    # Use numpy-based masking to avoid index alignment issues
    t_np = t.to_numpy()
    lo = tmin if tmin is not None else float(np.nanmin(t_np))
    hi = tmax if tmax is not None else float(np.nanmax(t_np))
    cut = (t_np >= lo) & (t_np <= hi)

    fig = go.Figure()

    for col in signals:
        y = df[col]
        if window and window > 1:
            y = _moving_average(y, window)
        y_np = pd.to_numeric(y, errors="coerce").to_numpy()
        fig.add_trace(go.Scatter(x=t_np[cut], y=y_np[cut], mode='lines', name=legend_map.get(col, col)))

    if ignition_time is not None:
        fig.add_vline(x=ignition_time, line_width=2, line_dash="dash", annotation_text="Ignition", annotation_position="top right")

    full_title = title if title else "Signals vs Time"
    if subtitle:
        full_title += f"<br><sup>{subtitle}</sup>"

    fig.update_layout(
        title=full_title,
        xaxis_title="Time (s)",
        yaxis_title="Signal",
        legend_title="Channels",
        margin=dict(l=40, r=10, t=70, b=40),
        height=560,
    )

    return fig, cut


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Post Process GUI", layout="wide")
st.image("logo.png", width=140)
st.title("Post Process GUI")
st.caption("Upload CSV/TDMS → choose columns → build time from Fs if needed → add ignition flag → customize plot → cut → export")

with st.sidebar:
    st.header("1) Load data")
    kind = st.radio("File type", ["CSV", "TDMS"], horizontal=True)
    file = st.file_uploader("Upload file", type=(['csv'] if kind == 'CSV' else ['tdms']))

    df: Optional[pd.DataFrame] = None
    if file is not None:
        if kind == "CSV":
            st.subheader("CSV options")
            auto_detect = st.checkbox("Auto-detect delimiter (recommended)", value=True,
                                      help="Uses Python engine & csv.Sniffer to guess commas/semicolons/tabs/pipes")
            sep = st.selectbox("Delimiter (if not auto)", options=[",", ";", "	", " ", "|"], index=0)
            header_has_row = st.checkbox("First row is header", value=True)
            skiprows = st.number_input("Rows to skip at top (metadata)", min_value=0, value=0, step=1)
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "utf-16"], index=0)
            engine = st.selectbox("Engine", ["c", "python"], index=0,
                                   help="If tokenizing errors occur, switch to 'python'")
            on_bad = st.selectbox("On bad lines (python engine)", ["error", "warn", "skip"], index=2)
            try:
                df = _load_csv(file, sep=sep, header_row=header_has_row, skiprows=skiprows,
                               engine=engine, encoding=encoding, on_bad_lines=on_bad, auto_detect=auto_detect)
                df = df.reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        else:
            if not _HAS_TDMS:
                st.warning("nptdms not installed. Run: pip install nptdms")
            try:
                df = _load_tdms(file)
                df = df.reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to read TDMS: {e}")

    st.divider()
    st.header("2) Time base")
    time_mode = st.radio("Use time from:", ["Sample rate (Fs)", "Existing time column"], index=0, help="Your files are samples vs data; pick Fs to build time axis")
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
    st.header("3) Channels (8 available)")
    if df is not None:
        _, signals_guess = _infer_time_and_signals(df)
        # Single channel quick-pick
        quick = st.selectbox("Quick pick one channel", options=signals_guess, index=0)
        # Multi-select for plotting many on one graph
        signal_cols = st.multiselect("Channels to plot (multi-select)", options=list(df.columns), default=[quick], help="Hold Ctrl/Cmd to select multiple")
    else:
        signal_cols = []

    st.divider()
    st.header("4) Smoothing")
    window = st.number_input("Moving-average window (samples)", min_value=1, max_value=10001, value=1, step=1)

    st.divider()
    st.header("5) Ignition flag")
    ign_method = st.radio("Method", ["manual", "threshold", "existing"], help="Manual: pick time; Threshold: detect crossing; Existing: use 'ignition_flag' column")
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

# Build time axis
t_secs = _time_from_samplerate(len(df), fs, index=df.index) if time_mode == "Sample rate (Fs)" else _to_seconds(df[time_col]).set_axis(df.index)

# Legend labels customization
st.subheader("Plot customization")
colA, colB = st.columns([2, 1])
with colA:
    chart_title = st.text_input("Chart title", value="Signals vs Time")
with colB:
    exp_date = st.date_input("Experiment date")

legend_map: Dict[str, str] = {}
with st.expander("Legend labels (rename per channel)"):
    for ch in signal_cols:
        legend_map[ch] = st.text_input(f"Label for '{ch}'", value=ch, key=f"legend_{ch}")

# Compute ignition flag on chosen timebase
ign_flag, detected_time = _compute_ignition_flag(
    df, t=t_secs, method=ign_method,
    manual_time=ign_time_in, threshold_channel=thr_chan,
    threshold_value=thr_val, direction=thr_dir,
)

_df = df.copy()
_df["ignition_flag"] = ign_flag.astype(int)

# Time window controls
st.subheader("Time window")
left, right = st.columns([3, 2])
with left:
    t0, t1 = float(t_secs.min()), float(t_secs.max())
    w = st.slider("Select range (s)", min_value=t0, max_value=t1, value=(t0, t1))
with right:
    st.write("Manual cut (precise)")
    man_tmin = st.number_input("Start (s)", value=w[0], step=0.001, format="%.6f")
    man_tmax = st.number_input("End (s)", value=w[1], step=0.001, format="%.6f")

# Final time limits (manual overrides slider if changed)
final_tmin = float(man_tmin)
final_tmax = float(man_tmax)

subtitle = str(exp_date) if exp_date else None
fig, cut_mask = _make_plot(_df, t_secs, signal_cols, legend_map, window, final_tmin, final_tmax,
                           detected_time if ign_method != "manual" else ign_time_in,
                           title=chart_title, subtitle=subtitle)

st.plotly_chart(fig, use_container_width=True)

# Info + preview
info_left, info_right = st.columns([1, 2])
with info_left:
    st.markdown(f"**Samples:** {len(_df)}")
    st.markdown(f"**Duration:** {t_secs.iloc[-1]-t_secs.iloc[0]:.3f} s")
    if detected_time is not None or ign_method == "manual":
        st.markdown(f"**Ignition @** {(detected_time if ign_method != 'manual' else ign_time_in):.6f} s")
with info_right:
    st.dataframe(_df.loc[cut_mask, [c for c in [time_col] if time_col] + signal_cols + ["ignition_flag"]].head(50), use_container_width=True)

# ---------------------------
# Exports
# ---------------------------
st.divider()
st.subheader("Export")

cut_df = _df.loc[cut_mask, signal_cols].copy()
cut_df.insert(0, "time_s", t_secs[cut_mask].values)
cut_df["ignition_flag"] = _df.loc[cut_mask, "ignition_flag"].values

csv_bytes = cut_df.to_csv(index=False).encode("utf-8")
base_name = (chart_title or "signals_plot").strip().replace(" ", "_")

st.download_button("Download cut CSV", data=csv_bytes, file_name=f"{base_name}_cut.csv", mime="text/csv")

try:
    png_bytes = fig.to_image(format="png", width=1400, height=650, scale=2)
    st.download_button("Download plot PNG", data=png_bytes, file_name=f"{base_name}.png", mime="image/png")
except Exception as e:
    with st.expander("PNG export troubleshooting"):
        st.warning("PNG export requires 'kaleido'. Run: pip install -U kaleido")
        st.code(str(e))

# Session meta
import json
meta = {
    "chart_title": chart_title,
    "experiment_date": str(exp_date) if exp_date else None,
    "time_mode": "Fs" if time_mode == "Sample rate (Fs)" else "existing_col",
    "fs_hz": fs if time_mode == "Sample rate (Fs)" else None,
    "ignition_method": ign_method,
    "ignition_time_s": float(detected_time if ign_method != "manual" else (ign_time_in or np.nan)) if (detected_time is not None or ign_method == "manual") else None,
    "time_window_s": [final_tmin, final_tmax],
    "channels": signal_cols,
    "legend_map": legend_map,
}
meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
st.download_button("Download session meta (JSON)", data=meta_bytes, file_name=f"{base_name}_meta.json", mime="application/json")

st.caption("© Spacefield – Local analysis utility. Built with Streamlit.")
