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
import io
import logging
from pathlib import Path

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
}

DEVICE_FIELD_MAPS = {
    "INV": INV_FIELD_MAP,
    "MTR": MTR_FIELD_MAP,
    "UPS": UPS_FIELD_MAP,
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
            for sub_id in [s.strip() for s in col2.split(",") if s.strip()]:
                sub_device_map[sub_id] = col3

        elif col2 and not col1:
            # ── Multi-device continuation row ────────────────────────────────
            for sub_id in [s.strip() for s in col2.split(",") if s.strip()]:
                sub_device_map[sub_id] = col3

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


def process_files(
    source_files,
    simple_map:        dict,
    sub_device_map:    dict,
    multi_device_stems: set,
    col_map:           dict,
    n_cols:            int,
    timestamp_fmt:     str,
    output_tz:         str,
    log_lines:         list,
) -> dict[int, pd.Series]:
    """
    Iterate over uploaded source CSVs and map every column to a template
    column index. Returns {col_idx: pd.Series[timestamp → value]}.
    """
    def log(msg): log_lines.append(msg)

    col_series: dict[int, pd.Series] = {}
    total_mapped = 0

    for uploaded in source_files:
        stem = Path(uploaded.name).stem   # e.g. "INV2-01", "MET01"
        content = uploaded.read().decode("utf-8-sig")
        src = pd.read_csv(io.StringIO(content), dtype=str)

        if "Timestamp" not in src.columns:
            log(f"⚠  {uploaded.name}: no 'Timestamp' column found — skipped")
            continue

        try:
            src["Timestamp"] = pd.to_datetime(src["Timestamp"], format=timestamp_fmt)
        except Exception as e:
            log(f"⚠  {uploaded.name}: timestamp parse error ({e}) — skipped")
            continue

        src = src.set_index("Timestamp")
        file_mapped = 0

        # ── Multi-device file (e.g. MET01) ───────────────────────────────────
        if stem in multi_device_stems:
            log(f"▸  {uploaded.name}  [multi-device]")

            for src_col in src.columns:
                parts = src_col.split(".")
                if len(parts) < 2:
                    continue

                sub_device    = parts[-2]   # e.g. "PYR02"
                metric_suffix = parts[-1]   # e.g. "IRRADIANCE_GHI"

                wattch_id = sub_device_map.get(sub_device)
                if not wattch_id:
                    log(f"   –  {src_col}: sub-device '{sub_device}' not in device map")
                    continue

                mapping = MET_SUFFIX_MAP.get(metric_suffix)
                if not mapping:
                    log(f"   –  {src_col}: no MET suffix map for '{metric_suffix}'")
                    continue

                wattch_metric, phase_idx = mapping
                col_idx = col_map.get((wattch_id, wattch_metric, phase_idx))
                if col_idx is None:
                    log(f"   –  {src_col}: no template column for "
                        f"({wattch_id}, {wattch_metric}, idx={phase_idx})")
                    continue

                series = pd.to_numeric(src[src_col], errors="coerce")
                col_series[col_idx] = (
                    col_series[col_idx].combine_first(series)
                    if col_idx in col_series else series
                )
                log(f"   ✓  {src_col} → {wattch_id} / {wattch_metric} "
                    f"[idx={phase_idx}] → template col {col_idx}")
                file_mapped += 1

        # ── Simple single-device file ─────────────────────────────────────────
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

            for src_col in src.columns:
                suffix  = src_col.split(".")[-1]
                mapping = field_map.get(suffix)
                if not mapping:
                    continue

                wattch_metric, phase_idx = mapping
                col_idx = col_map.get((wattch_id, wattch_metric, phase_idx))
                if col_idx is None:
                    continue

                series = pd.to_numeric(src[src_col], errors="coerce")
                col_series[col_idx] = (
                    col_series[col_idx].combine_first(series)
                    if col_idx in col_series else series
                )
                log(f"   ✓  {src_col} → template col {col_idx}")
                file_mapped += 1

        log(f"     {file_mapped} columns mapped\n")
        total_mapped += file_mapped

    log(f"Total columns mapped: {total_mapped}")
    return col_series


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_output(
    header_rows: list,
    col_series:  dict,
    n_cols:      int,
    output_tz:   str,
) -> bytes:
    """Assemble the output CSV as bytes ready for download."""
    data_df = pd.DataFrame(col_series).sort_index()

    buf = io.StringIO()
    writer = csv.writer(buf)

    for row in header_rows:
        writer.writerow(row)

    for ts, row_data in data_df.iterrows():
        out_row = [""] * n_cols
        out_row[0] = ts.strftime(f"%Y-%m-%dT%H:%M:%S{output_tz}")

        for col_idx, val in row_data.items():
            if pd.notna(val):
                out_row[col_idx] = (
                    int(val) if float(val) == int(val) else round(val, 6)
                )

        writer.writerow(out_row)

    return buf.getvalue().encode("utf-8")


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
            f"{len(_sub)} sub-devices, multi-device files: {_multi or 'none'}"
        ):
            if _simple:
                st.markdown("**Simple devices (1 file → 1 Wattch ID)**")
                st.dataframe(
                    pd.DataFrame(_simple.items(), columns=["Source file", "Wattch ID"]),
                    hide_index=True, use_container_width=True,
                )
            if _sub:
                st.markdown("**Sub-devices (multi-device file)**")
                st.dataframe(
                    pd.DataFrame(_sub.items(), columns=["Sub-device ID", "Wattch ID"]),
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
                f"{len(sub_device_map)} sub-devices, "
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
            col_series = process_files(
                source_files,
                simple_map,
                sub_device_map,
                multi_device_stems,
                col_map,
                n_cols,
                timestamp_fmt,
                output_tz,
                log_lines,
            )

            if not col_series:
                st.error(
                    "No columns were successfully mapped. "
                    "Check that your device map and source files match."
                )
            else:
                data_df = pd.DataFrame(col_series).sort_index()
                n_timestamps = len(data_df)
                n_values = int(data_df.notna().sum().sum())

                output_bytes = build_output(header_rows, col_series, n_cols, output_tz)

        except Exception as e:
            st.error(f"Error during processing: {e}")
            log_lines.append(f"\nFATAL ERROR: {e}")

    # Results
    if output_bytes:
        st.success(
            f"✅  Done — {n_timestamps:,} timestamps, "
            f"{n_values:,} data values mapped across "
            f"{len(col_series)} template columns"
        )
        st.download_button(
            label="⬇  Download output CSV",
            data=output_bytes,
            file_name="wattch_upload_output.csv",
            mime="text/csv",
            type="primary",
        )

    with st.expander("Processing log", expanded=not output_bytes):
        st.code("\n".join(log_lines), language=None)
