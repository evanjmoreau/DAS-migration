#!/usr/bin/env python3
"""
DAS Migration Tool  —  Streamlit App
=====================================
Upload your device map, Wattch template, and source CSVs to generate
a populated Wattch data-upload CSV.

Run with:
    streamlit run das_migration_app.py
"""

import csv
import gc
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# FIELD MAPS
# Maps source column metric suffix → (wattch_internal_metric, phase_index)
# These are consistent across projects for each device type.
# ─────────────────────────────────────────────────────────────────────────────

INV_FIELD_MAP = {
    # AC output
    "AC_CURRENT_A":     ("AC_OUTPUT_CURRENT",                    1),
    "AC_CURRENT_B":     ("AC_OUTPUT_CURRENT",                    2),
    "AC_CURRENT_C":     ("AC_OUTPUT_CURRENT",                    3),
    "AC_POWER":         ("AC_OUTPUT_POWER_ACTIVE",               None),
    "AC_VOLTAGE_AB":    ("AC_OUTPUT_VOLTAGE_LL",                 1),
    "AC_VOLTAGE_BC":    ("AC_OUTPUT_VOLTAGE_LL",                 2),
    "AC_VOLTAGE_CA":    ("AC_OUTPUT_VOLTAGE_LL",                 3),
    "FREQUENCY":        ("AC_OUTPUT_FREQUENCY",                  None),
    "POWER_FACTOR":     ("AC_OUTPUT_POWER_FACTOR",               None),
    "SVA":              ("AC_OUTPUT_POWER_APPARENT",             None),
    "VAR":              ("AC_OUTPUT_POWER_REACTIVE",             None),
    # DC input
    "DC_CURRENT":       ("DC_INPUT_CURRENT",                     1),
    "DC_VOLTAGE":       ("DC_INPUT_VOLTAGE",                     1),
    # Energy
    "ENERGY_DELIVERED": ("LIFETIME_OUTPUT_ENERGY_IMPORT",        None),
    # Temperatures
    "T_INTERNAL":       ("ACTIVE_ELEMENT_TEMPERATURE",           None),
    "T_COOLER":         ("ACTIVE_ELEMENT_TEMPERATURE",           None),
    "T_MOD":            ("AMBIENT_TEMPERATURE",                  None),
    # Fault registers
    "STATUS_FAULT_00":  ("EVENT_BITFIELD",                       40),
    "STATUS_FAULT_01":  ("EVENT_BITFIELD",                       41),
    "STATUS_FAULT_02":  ("EVENT_BITFIELD",                       42),
    "STATUS_FAULT_03":  ("EVENT_BITFIELD",                       43),
    "STATUS_FAULT_04":  ("EVENT_BITFIELD",                       44),
    "STATUS_FAULT_05":  ("EVENT_BITFIELD",                       45),
    "STATUS_FAULT_06":  ("EVENT_BITFIELD",                       46),
    "STATUS_FAULT":     ("EVENT_BITFIELD",                       None),
}

MTR_FIELD_MAP = {
    "AC_CURRENT_A":     ("AC_INPUT_CURRENT",                     1),
    "AC_CURRENT_B":     ("AC_INPUT_CURRENT",                     2),
    "AC_CURRENT_C":     ("AC_INPUT_CURRENT",                     3),
    "AC_POWER":         ("AC_INPUT_POWER_ACTIVE",                None),
    "AC_VOLTAGE_AB":    ("AC_INPUT_VOLTAGE_LL",                  1),
    "AC_VOLTAGE_BC":    ("AC_INPUT_VOLTAGE_LL",                  2),
    "AC_VOLTAGE_CA":    ("AC_INPUT_VOLTAGE_LL",                  3),
    "AC_VOLTAGE_A":     ("AC_INPUT_VOLTAGE",                     1),
    "AC_VOLTAGE_B":     ("AC_INPUT_VOLTAGE",                     2),
    "AC_VOLTAGE_C":     ("AC_INPUT_VOLTAGE",                     3),
    "AC_VOLTAGE_LL":    ("AC_INPUT_VOLTAGE_LL",                  None),
    "FREQUENCY":        ("AC_INPUT_FREQUENCY",                   None),
    "POWER_FACTOR":     ("AC_INPUT_POWER_FACTOR",                None),
    "SVA":              ("AC_INPUT_POWER_APPARENT",              None),
    "VAR":              ("AC_INPUT_POWER_REACTIVE",              None),
    "ENERGY_DELIVERED": ("LIFETIME_INPUT_ENERGY_EXPORT",         None),
    "ENERGY_RECEIVED":  ("LIFETIME_INPUT_ENERGY_IMPORT",         None),
}

UPS_FIELD_MAP = {
    "DC_VOLTAGE":       ("DC_INPUT_VOLTAGE",                     1),
    "T_ASSET":          ("ACTIVE_ELEMENT_TEMPERATURE",           None),
}

# For multi-device MET files: maps column metric suffix → (wattch_metric, phase_idx)
# The Wattch device ID comes from the device map (sub-device lookup), not hardcoded here.
MET_SUFFIX_MAP = {
    # Irradiance channels
    "IRRADIANCE_GHI":       ("IRRADIANCE",                       1),
    "IRRADIANCE_POA":       ("IRRADIANCE",                       2),
    "IRRADIANCE_REAR":      ("IRRADIANCE",                       3),
    "IRRADIANCE_AUX":       ("IRRADIANCE",                       4),
    # Standalone GHI sensor (e.g. Hukseflux)
    "IRRADIANCE_GLOB":      ("GLOBAL_HORIZONTAL_IRRADIANCE",     None),
    # Temperatures
    "T_AMB":                ("AMBIENT_TEMPERATURE",              None),
    "T_MOD":                ("SURFACE_TEMPERATURE",              1),
    "T_ONBOARD":            ("SURFACE_TEMPERATURE",              2),
    # Weather station channels
    "BAROMETRIC_PRES":      ("ATMOSPHERIC_PRESSURE",             None),
    "HUMIDITY":             ("HUMIDITY",                         None),
    "WIND_DIRECTION":       ("WIND_BEARING",                     None),
    "WIND_SPEED":           ("WIND_SPEED",                       None),
    "PRECIPITATION":        ("PRECIPITATION_INTENSITY",          None),
}

RCL_FIELD_MAP = {
    # AC input currents (phases A/B/C, then neutral and ground)
    "AC_CURRENT_A":            ("AC_INPUT_CURRENT",              1),
    "AC_CURRENT_B":            ("AC_INPUT_CURRENT",              2),
    "AC_CURRENT_C":            ("AC_INPUT_CURRENT",              3),
    "AC_CURRENT_N":            ("AC_INPUT_CURRENT",              4),
    "AC_CURRENT_GND":          ("AC_INPUT_CURRENT",              5),
    # AC input voltages (phase-to-neutral; Z-terminal = zero-sequence)
    "AC_VOLTAGE_A":            ("AC_INPUT_VOLTAGE",              1),
    "AC_VOLTAGE_B":            ("AC_INPUT_VOLTAGE",              2),
    "AC_VOLTAGE_C":            ("AC_INPUT_VOLTAGE",              3),
    "AC_VOLTAGE_A_Z_TERMINAL": ("AC_INPUT_VOLTAGE",              4),
    "AC_VOLTAGE_B_Z_TERMINAL": ("AC_INPUT_VOLTAGE",              5),
    "AC_VOLTAGE_C_Z_TERMINAL": ("AC_INPUT_VOLTAGE",              6),
    # Breaker state
    "STATUS_BREAKER":          ("STATE_ENUM",                    None),
}

DEVICE_FIELD_MAPS = {
    "INV": INV_FIELD_MAP,
    "MTR": MTR_FIELD_MAP,
    "UPS": UPS_FIELD_MAP,
    "RCL": RCL_FIELD_MAP,
}

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE MAP PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_device_map(uploaded_file) -> tuple[dict, dict, set]:
    """
    Parse the device map Excel into three structures:

    simple_map         : {source_file_stem: wattch_id}
                         For single-device CSVs (INV*, MTR01, UPS01, …)
    sub_device_map     : {sub_device_id: wattch_id}
                         For sub-devices within multi-device CSVs (e.g. PYR02 → FXE1FcyR)
    multi_device_stems : set of source file stems that contain multiple devices
                         (e.g. {"MET01"})

    Device map Excel column layout (col 0 is always blank):
      col 1  source file stem for all devices; blank on multi-device continuation rows
      col 2  sub-device IDs (comma-separated); blank for simple devices
      col 3  Wattch ID

    Row types:
      col1 set,  col2 blank  → simple device  (col1 = file stem, col3 = Wattch ID)
      col1 set,  col2 set    → multi-device parent row  (col1 = file stem,
                               col2 = sub-device IDs, col3 = Wattch ID)
      col1 blank, col2 set   → multi-device continuation row (inherits parent stem)
    """
    df = pd.read_excel(uploaded_file, header=None, dtype=str).fillna("")

    simple_map         = {}
    # sub_device_map is scoped per source file stem so that MET01, MET02, MET03
    # etc. can each have sub-devices with identical names (TMP01, PYR01 …)
    # without overwriting each other.
    # Structure:  { file_stem: { sub_device_id: wattch_id } }
    sub_device_map     = {}
    multi_device_stems = set()
    current_parent     = None

    for _, row in df.iterrows():
        col1 = str(row[1]).strip() if len(row) > 1 else ""
        col2 = str(row[2]).strip() if len(row) > 2 else ""
        col3 = str(row[3]).strip() if len(row) > 3 else ""

        if not col3 or col3 == "nan":
            continue

        if col1 and col2:
            # ── Multi-device parent row ──────────────────────────────────────
            current_parent = col1
            multi_device_stems.add(col1)
            if col1 not in sub_device_map:
                sub_device_map[col1] = {}
            for sub_id in [s.strip() for s in col2.split(",") if s.strip()]:
                sub_device_map[col1][sub_id] = col3

        elif col2 and not col1:
            # ── Multi-device continuation row ────────────────────────────────
            if current_parent:
                for sub_id in [s.strip() for s in col2.split(",") if s.strip()]:
                    sub_device_map[current_parent][sub_id] = col3

        elif col1 and not col2:
            # ── Simple 1:1 device row ────────────────────────────────────────
            current_parent = None
            simple_map[col1] = col3

    return simple_map, sub_device_map, multi_device_stems

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_template(uploaded_file) -> tuple[list, dict, int]:
    """
    Parse the Wattch template CSV.

    Returns
    -------
    header_rows : list[list[str]]   The 6 header lines, preserved verbatim
    col_map     : dict  (wattch_device_id, metric_name, phase_idx) → col_index
    n_cols      : int   total column count
    """
    content = uploaded_file.read().decode("utf-8-sig")
    all_rows = list(csv.reader(io.StringIO(content)))

    device_row = all_rows[0]
    metric_row = all_rows[1]
    index_row  = all_rows[2]

    col_map = {}
    for i in range(1, len(device_row)):
        dev_id = device_row[i].strip()
        metric = metric_row[i].strip() if i < len(metric_row) else ""
        idx_s  = index_row[i].strip()  if i < len(index_row)  else ""

        if not dev_id or not metric:
            continue

        phase_idx = None
        if idx_s and idx_s.lower() not in ("", "nan"):
            try:
                phase_idx = int(float(idx_s))
            except ValueError:
                pass

        key = (dev_id, metric, phase_idx)
        if key not in col_map:
            col_map[key] = i

    return all_rows[:6], col_map, len(device_row)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def get_device_prefix(name: str) -> str | None:
    for prefix in DEVICE_FIELD_MAPS:
        if name.upper().startswith(prefix):
            return prefix
    return None

def find_unclaimed_col(
    wid: str,
    wm:  str,
    pi,
    col_map:      dict,
    claimed_cols: set,
) -> tuple:
    """
    Find the best available template column for a (device, metric, phase_idx) triple.

    Strategy (in order):
      1. Try the exact phase_idx from the field map.
      2. If not found or claimed, try idx=None  (standalone single-channel sensors).
      3. If still not found or claimed, auto-increment idx 1→9  (handles same-device
         multi-channel sensors like front/back of module on a single pyranometer).

    Returns (col_idx, actual_phase_idx) or (None, None) if nothing is available.
    This lets TMP02 claim SURFACE_TEMPERATURE idx=1 and TMP04 automatically
    fall through to idx=2 on the same device without any hardcoded mappings.
    """
    candidates = [pi]
    if pi is not None:
        candidates.append(None)          # fallback to no-index variant
    candidates += [i for i in range(1, 10) if i != pi]   # try idx 1-9

    for candidate_pi in candidates:
        cidx = col_map.get((wid, wm, candidate_pi))
        if cidx is not None and cidx not in claimed_cols:
            return cidx, candidate_pi

    return None, None

def process_files(
    source_files,
    simple_map:         dict,
    sub_device_map:     dict,
    multi_device_stems: set,
    col_map:            dict,
    timestamp_fmt:      str,
    log_lines:          list,
) -> pd.DataFrame:
    """
    Memory-efficient processing pipeline:

    1. Reads each source CSV in 50,000-row chunks — never loads a full file at once.
    2. Converts numeric values to float32 immediately (half the memory of float64).
    3. Retains only mapped columns — all unmapped source data is discarded.
    4. Merges one file at a time into a single accumulator DataFrame and calls
       gc.collect() after each file so Python can reclaim memory promptly.

    Returns a DataFrame indexed by timestamp with template column indices as
    column names and float32 values.
    """
    def log(msg): log_lines.append(msg)

    accumulator: pd.DataFrame | None = None
    total_mapped = 0

    for uploaded in source_files:
        stem     = Path(uploaded.name).stem
        is_multi = stem in multi_device_stems

        # ── Resolve device metadata before touching file bytes ────────────────
        if is_multi:
            log(f"▸  {uploaded.name}  [multi-device]")
            wattch_id = None
            field_map = None
        else:
            wattch_id = simple_map.get(stem)
            if not wattch_id:
                log(f"⚠  '{stem}' not found in device map — skipping {uploaded.name}")
                continue
            prefix    = get_device_prefix(stem)
            field_map = DEVICE_FIELD_MAPS.get(prefix)
            if not field_map:
                log(f"⚠  No field map for device type '{prefix}' ({stem}) — skipping")
                continue
            log(f"▸  {uploaded.name}  →  Wattch ID: {wattch_id}")

        # ── Resolve sub-device map once (multi-device files only) ──────────────────
        file_sub_map = sub_device_map.get(stem, {}) if is_multi else {}
        if is_multi:
            log(f"   Sub-device map entries for {stem}:")
            for sd, wid in sorted(file_sub_map.items()):
                log(f"     {sd} → {wid}")
            log("")

        # ── Read in 50k-row chunks; free raw bytes immediately ────────────────
        raw = uploaded.read()
        chunk_iter = pd.read_csv(
            io.BytesIO(raw), dtype=str, chunksize=50_000, encoding="utf-8-sig"
        )
        del raw

        col_idx_map: dict[str, int] = {}   # src_col_name → template col index
        file_chunks: list[pd.DataFrame] = []
        file_mapped = 0
        skip_file   = False

        for chunk_num, chunk in enumerate(chunk_iter):

            # Validate timestamp column
            if "Timestamp" not in chunk.columns:
                log(f"⚠  {uploaded.name}: no 'Timestamp' column — skipped")
                skip_file = True
                break
            try:
                chunk["Timestamp"] = pd.to_datetime(
                    chunk["Timestamp"], format=timestamp_fmt
                )
            except Exception as e:
                log(f"⚠  {uploaded.name}: timestamp error ({e}) — skipped")
                skip_file = True
                break

            chunk = chunk.set_index("Timestamp")

            # Build the src→template column index map on the first chunk only
            if chunk_num == 0:
                claimed_cols: set[int] = set()   # template col indices already mapped
                for src_col in chunk.columns:
                    if is_multi:
                        parts = src_col.split(".")
                        if len(parts) < 2:
                            continue
                        sub_dev, m_suffix = parts[-2], parts[-1]
                        wid = file_sub_map.get(sub_dev)
                        if not wid:
                            log(f"   ✗  {src_col}: sub-device '{sub_dev}' not found in device map for {stem} — skipped")
                            continue
                        mapping = MET_SUFFIX_MAP.get(m_suffix)
                        if not mapping:
                            log(f"   ✗  {src_col}: metric suffix '{m_suffix}' not in MET_SUFFIX_MAP — skipped")
                            continue
                        wm, pi = mapping
                        cidx, actual_pi = find_unclaimed_col(wid, wm, pi, col_map, claimed_cols)
                        if cidx is not None:
                            col_idx_map[src_col] = cidx
                            claimed_cols.add(cidx)
                            suffix_note = f" (auto idx={actual_pi})" if actual_pi != pi else ""
                            log(f"   ✓  {src_col} → {wid}/{wm} [idx={actual_pi}]{suffix_note} → col {cidx}")
                            file_mapped += 1
                        else:
                            log(f"   ✗  {src_col} → ({wid}, {wm}, idx={pi}) — no available column in template")
                    else:
                        suffix  = src_col.split(".")[-1]
                        mapping = field_map.get(suffix)
                        if not mapping:
                            continue
                        wm, pi  = mapping
                        cidx, actual_pi = find_unclaimed_col(wattch_id, wm, pi, col_map, claimed_cols)
                        if cidx is not None:
                            col_idx_map[src_col] = cidx
                            claimed_cols.add(cidx)
                            log(f"   ✓  {src_col} → col {cidx}")
                            file_mapped += 1
                        else:
                            log(f"   ✗  {src_col} → ({wattch_id}, {wm}, idx={pi}) — no available column in template")

            if not col_idx_map:
                break

            # Keep only mapped columns, rename to int col index, cast to float32
            # Group-by index to collapse any duplicate timestamps before appending
            mapped = (
                chunk[list(col_idx_map.keys())]
                .rename(columns=col_idx_map)
                .apply(pd.to_numeric, errors="coerce")
                .astype("float32")
            )
            if mapped.index.duplicated().any():
                mapped = mapped.groupby(level=0).mean()
            file_chunks.append(mapped)
            del chunk, mapped   # free immediately

        if skip_file or not file_chunks:
            log(f"     0 data rows processed\n")
            del file_chunks
            gc.collect()
            continue

        # Concatenate chunks for this file, then free chunk list
        file_df = pd.concat(file_chunks, copy=False)
        del file_chunks
        gc.collect()

        log(f"     {file_mapped} columns mapped, {len(file_df):,} rows\n")
        total_mapped += file_mapped

        # Merge into running accumulator, then free this file's DataFrame
        if accumulator is None:
            accumulator = file_df
        else:
            accumulator = accumulator.combine_first(file_df)
        del file_df
        gc.collect()

    log(f"Total columns mapped: {total_mapped}")

    if accumulator is not None:
        acc = accumulator.sort_index()
        # Report non-null value counts per column so missing data is visible
        log("\n── Column data summary (non-null row counts) ────────────────────")
        for col_idx in sorted(acc.columns):
            n_valid = int(acc[col_idx].notna().sum())
            if n_valid == 0:
                log(f"   ⚠  template col {col_idx}: 0 non-null values — column will be empty in output")
            else:
                log(f"   ✓  template col {col_idx}: {n_valid:,} non-null values")
        return acc
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_output(
    header_rows: list,
    data_df:     pd.DataFrame,
    n_cols:      int,
    output_tz:   str,
) -> bytes:
    """
    Memory-efficient output builder:

    1. Pre-formats all timestamps in one vectorised call.
    2. Builds the output as a numpy object array (one column at a time),
       avoiding row-by-row Python loops over wide DataFrames.
    3. Writes the final array to CSV in 5,000-row batches so the peak
       string buffer size is bounded regardless of total row count.
    """
    n_rows = len(data_df)

    # ── Pre-allocate output array filled with empty strings ───────────────────
    out = np.full((n_rows, n_cols), "", dtype=object)

    # ── Timestamps (vectorised strftime) ─────────────────────────────────────
    out[:, 0] = data_df.index.strftime(f"%Y-%m-%dT%H:%M:%S{output_tz}").to_numpy()

    # ── Fill data columns one at a time ──────────────────────────────────────
    for col_idx in data_df.columns:
        vals  = data_df[col_idx].to_numpy(dtype="float64")   # promote for formatting
        valid = ~np.isnan(vals)
        v     = vals[valid]
        # Format: integers where value is whole, 6 sig-fig decimal otherwise
        is_int = np.abs(v - np.round(v)) < 1e-5
        formatted          = np.empty(len(v), dtype=object)
        formatted[is_int]  = np.round(v[is_int]).astype("int64").astype(str)
        formatted[~is_int] = [f"{x:.6g}" for x in v[~is_int]]
        out[valid, col_idx] = formatted

    del data_df
    gc.collect()

    # ── Write to CSV in 5,000-row batches ────────────────────────────────────
    BATCH = 5_000
    parts: list[bytes] = []

    hdr_buf = io.StringIO()
    csv.writer(hdr_buf).writerows(header_rows)
    parts.append(hdr_buf.getvalue().encode("utf-8"))
    del hdr_buf

    for start in range(0, n_rows, BATCH):
        batch_buf = io.StringIO()
        csv.writer(batch_buf).writerows(out[start : start + BATCH].tolist())
        parts.append(batch_buf.getvalue().encode("utf-8"))
        del batch_buf

    del out
    gc.collect()

    return b"".join(parts)

def build_split_zip(
    header_rows: list,
    data_df:     pd.DataFrame,
    n_cols:      int,
    output_tz:   str,
    period:      str,           # "monthly" | "quarterly"
) -> tuple[bytes, list[str]]:
    """
    Split data_df by calendar period, write one CSV per period (each with the
    full template header), and bundle them into an in-memory zip archive.

    Returns (zip_bytes, list_of_filenames).
    """
    # ── Group timestamps by period ────────────────────────────────────────────
    if period == "monthly":
        groups = data_df.groupby(data_df.index.to_period("M"))
        fmt    = lambda p: f"wattch_upload_{p.year}_{p.month:02d}.csv"
    else:   # quarterly
        groups = data_df.groupby(data_df.index.to_period("Q"))
        fmt    = lambda p: f"wattch_upload_{p.year}_Q{p.quarter}.csv"

    zip_buf   = io.BytesIO()
    filenames = []

    BATCH = 5_000
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for period_label, chunk_df in groups:
            fname = fmt(period_label)
            filenames.append(fname)

            n_rows = len(chunk_df)
            out    = np.full((n_rows, n_cols), "", dtype=object)
            out[:, 0] = chunk_df.index.strftime(
                f"%Y-%m-%dT%H:%M:%S{output_tz}"
            ).to_numpy()

            for col_idx in chunk_df.columns:
                vals  = chunk_df[col_idx].to_numpy(dtype="float64")
                valid = ~np.isnan(vals)
                v     = vals[valid]
                if len(v) == 0:
                    continue
                is_int             = np.abs(v - np.round(v)) < 1e-5
                formatted          = np.empty(len(v), dtype=object)
                formatted[is_int]  = np.round(v[is_int]).astype("int64").astype(str)
                formatted[~is_int] = [f"{x:.6g}" for x in v[~is_int]]
                out[valid, col_idx] = formatted

            # Write CSV for this period into the zip
            csv_buf = io.StringIO()
            w = csv.writer(csv_buf)
            w.writerows(header_rows)
            for start in range(0, n_rows, BATCH):
                w.writerows(out[start : start + BATCH].tolist())

            zf.writestr(fname, csv_buf.getvalue())
            del csv_buf, out, chunk_df
            gc.collect()

    return zip_buf.getvalue(), filenames

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DAS Migration Tool",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ DAS Migration Tool")
st.markdown(
    "Transform historical solar DAS data into the Wattch upload template format. "
    "Upload your files below, configure the settings, then click **Run Migration**."
)

st.divider()

# ── File uploads ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1 · Device Map")
    device_map_file = st.file_uploader(
        "Excel file mapping device names to Wattch IDs",
        type=["xlsx"],
        key="device_map",
    )

with col2:
    st.subheader("2 · Wattch Template")
    template_file = st.file_uploader(
        "Wattch data upload template CSV",
        type=["csv"],
        key="template",
    )

with col3:
    st.subheader("3 · Source Data")
    source_files = st.file_uploader(
        "Equipment CSV files (select all at once)",
        type=["csv"],
        accept_multiple_files=True,
        key="sources",
    )

st.divider()

# ── Device map preview ────────────────────────────────────────────────────────
if device_map_file:
    device_map_file.seek(0)
    try:
        _simple, _sub, _multi = parse_device_map(device_map_file)
        device_map_file.seek(0)
        with st.expander(
            f"Device map preview — {len(_simple)} simple devices, "
            f"{sum(len(v) for v in _sub.values())} sub-devices across "
            f"{len(_sub)} multi-device file(s): {_multi or 'none'}"
        ):
            if _simple:
                st.markdown("**Simple devices (1 file → 1 Wattch ID)**")
                st.dataframe(
                    pd.DataFrame(_simple.items(), columns=["Source file", "Wattch ID"]),
                    hide_index=True, use_container_width=True,
                )
            if _sub:
                st.markdown("**Sub-devices (multi-device files)**")
                rows = [
                    {"File": stem, "Sub-device ID": sd, "Wattch ID": wid}
                    for stem, mapping in _sub.items()
                    for sd, wid in mapping.items()
                ]
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True, use_container_width=True,
                )
    except Exception as e:
        st.warning(f"Could not preview device map: {e}")

st.divider()

# ── Settings ──────────────────────────────────────────────────────────────────
st.subheader("Settings")
cfg1, cfg2 = st.columns(2)

with cfg1:
    timestamp_fmt = st.text_input(
        "Source timestamp format",
        value="%m/%d/%y %H:%M:%S",
        help="Python strptime format string, e.g. %m/%d/%y %H:%M:%S for 06/10/25 07:00:00",
    )

with cfg2:
    output_tz = st.text_input(
        "Output UTC offset",
        value="-05:00",
        help="ISO-8601 offset appended to output timestamps, e.g. -05:00 or +00:00",
    )

cfg3, _ = st.columns([1, 2])
with cfg3:
    split_mode = st.selectbox(
        "Output file splitting",
        options=["Single file", "Split by month", "Split by quarter"],
        index=0,
        help="Split into multiple CSVs if your upload portal has a file-size limit.",
    )

st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
ready = device_map_file and template_file and source_files
run_btn = st.button("▶  Run Migration", type="primary", disabled=not ready)

if not ready:
    missing = []
    if not device_map_file: missing.append("device map")
    if not template_file:   missing.append("template")
    if not source_files:    missing.append("source data files")
    st.info(f"Upload {' and '.join(missing)} to continue.")

if run_btn:
    log_lines = []
    output_bytes = None
    n_timestamps = 0
    n_values = 0

    with st.spinner("Processing…"):
        try:
            # Parse device map
            simple_map, sub_device_map, multi_device_stems = parse_device_map(
                device_map_file
            )
            log_lines.append(
                f"Device map: {len(simple_map)} simple devices, "
                f"{sum(len(v) for v in sub_device_map.values())} sub-devices, "
                f"multi-device files: {multi_device_stems or 'none'}\n"
            )

            # Parse template
            template_file.seek(0)
            header_rows, col_map, n_cols = parse_template(template_file)
            log_lines.append(
                f"Template: {len(col_map)} unique (device, metric, index) "
                f"columns across {n_cols - 1} data columns\n"
            )

            # Process source files
            data_df = process_files(
                source_files,
                simple_map,
                sub_device_map,
                multi_device_stems,
                col_map,
                timestamp_fmt,
                log_lines,
            )

            if data_df.empty:
                st.error(
                    "No columns were successfully mapped. "
                    "Check that your device map and source files match."
                )
            else:
                n_timestamps  = len(data_df)
                n_values      = int(data_df.notna().sum().sum())
                n_mapped_cols = len(data_df.columns)

                if split_mode == "Single file":
                    output_bytes = build_output(header_rows, data_df, n_cols, output_tz)
                else:
                    period = "monthly" if split_mode == "Split by month" else "quarterly"
                    output_bytes, split_filenames = build_split_zip(
                        header_rows, data_df, n_cols, output_tz, period
                    )

        except Exception as e:
            st.error(f"Error during processing: {e}")
            log_lines.append(f"\nFATAL ERROR: {e}")

    # Results
    if output_bytes:
        st.success(
            f"✅  Done — {n_timestamps:,} timestamps, "
            f"{n_values:,} data values mapped across "
            f"{n_mapped_cols} template columns"
        )
        if split_mode == "Single file":
            st.download_button(
                label="⬇  Download output CSV",
                data=output_bytes,
                file_name="wattch_upload_output.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.caption(f"Split into {len(split_filenames)} files: "
                       f"{', '.join(split_filenames)}")
            st.download_button(
                label=f"⬇  Download {len(split_filenames)}-file zip",
                data=output_bytes,
                file_name="wattch_upload_split.zip",
                mime="application/zip",
                type="primary",
            )

    with st.expander("Processing log", expanded=not output_bytes):
        st.code("\n".join(log_lines), language=None)
