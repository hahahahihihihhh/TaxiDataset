# -*- coding: utf-8 -*-
"""
region_inout_show.py
--------------------
Draw region-level inflow/outflow heatmaps for a given timestamp from a *.grid file.

The *.grid file is a CSV with columns:
    dyna_id,type,time,row_id,column_id,inflow,outflow

Example:
    python region_inout_show.py --grid_path ./T_DRIVE20150206.grid --time "2015-02-01T08:00:00Z"
    python region_inout_show.py --grid_path ./T_DRIVE20150206.grid --time "2015-02-01 08:00:00" --origin upper

If you have a setting.json (same format as your original script), you can omit --grid_path:
    python region_inout_show.py --dataset TDRIVE --time "2015-02-01T08:00:00Z"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

# Make matplotlib safe on servers (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


setting_file = "setting.json"
out_dir_path = "./graph/en"


@dataclass
class GridInfo:
    n_rows: int
    n_cols: int
    matched_cells: int
    min_time_seen: Optional[str]
    max_time_seen: Optional[str]


def _normalize_time_to_z(t: str) -> str:
    """
    Normalize user input time to 'YYYY-MM-DDTHH:MM:SSZ' to match grid file.
    Accepts:
      - 2015-02-01T08:00:00Z
      - 2015-02-01T08:00:00
      - 2015-02-01 08:00:00
    """
    t = (t or "").strip()
    if not t:
        raise ValueError("Empty time string.")

    # already Z format
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", t):
        return t

    # Try a few common patterns
    patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(t, p)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # Last resort: attempt fromisoformat (without Z)
    try:
        dt = datetime.fromisoformat(t.replace("Z", ""))
        return dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        raise ValueError(f"Unsupported time format: {t}") from e


def _safe_slug(s: str) -> str:
    """Make a safe filename slug."""
    return re.sub(r"[^0-9A-Za-z_-]+", "-", s).strip("-")


def _resolve_grid_path(dataset: str, setting_path: Path) -> Optional[Path]:
    """
    Resolve grid path from setting.json if it exists.
    Expected structure:
      settings[dataset]["paths"]["raw_data_grid"] = {"dir": "...", "file": "..."}
    """
    if not setting_path.exists():
        return None

    try:
        settings = json.loads(setting_path.read_text(encoding="utf-8"))
        cfg = settings[dataset]["paths"]["raw_data_grid"]
        p = Path(cfg["dir"]) / f"{cfg['file']}.grid"
        return p
    except Exception:
        return None


def build_matrices_for_time(
    grid_path: Path,
    target_time: str,
) -> Tuple[np.ndarray, np.ndarray, GridInfo]:
    """
    Stream the CSV to avoid loading it fully into memory.
    Returns:
      inflow_mat (n_rows, n_cols)
      outflow_mat (n_rows, n_cols)
      GridInfo summary
    """
    max_r = -1
    max_c = -1

    inflow: Dict[Tuple[int, int], float] = {}
    outflow: Dict[Tuple[int, int], float] = {}

    matched = 0

    # Track a little time range for debugging when target_time not found
    first_times: list[str] = []
    last_times = deque(maxlen=5)

    # Because file may be large, read with csv.DictReader
    with grid_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time", "row_id", "column_id", "inflow", "outflow"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Invalid grid header. Need columns {sorted(required)}, got {reader.fieldnames}"
            )

        for i, row in enumerate(reader, start=1):

            t = row["time"]
            if len(first_times) < 5 and (not first_times or t != first_times[-1]):
                first_times.append(t)
            if not last_times or t != last_times[-1]:
                last_times.append(t)

            r = int(row["row_id"])
            c = int(row["column_id"])

            if t == target_time:
                # In case same (r,c) repeats, last one wins
                inflow[(r, c)] = float(row["inflow"])
                outflow[(r, c)] = float(row["outflow"])
                matched += 1

                # If not inferring from all, infer from selected time only
                if r > max_r:
                    max_r = r
                if c > max_c:
                    max_c = c

    if max_r < 0 or max_c < 0:
        raise RuntimeError("Failed to infer grid shape (empty file?).")

    n_rows, n_cols = max_r + 1, max_c + 1

    inflow_mat = np.zeros((n_rows, n_cols), dtype=np.float32)
    outflow_mat = np.zeros((n_rows, n_cols), dtype=np.float32)

    for (r, c), v in inflow.items():
        inflow_mat[r, c] = v
    for (r, c), v in outflow.items():
        outflow_mat[r, c] = v

    min_time = first_times[0] if first_times else None
    max_time = last_times[-1] if last_times else None

    info = GridInfo(
        n_rows=n_rows,
        n_cols=n_cols,
        matched_cells=matched,
        min_time_seen=min_time,
        max_time_seen=max_time,
    )

    # If nothing matched, raise with helpful hint
    if matched == 0:
        hint = (
            f"Target time not found: {target_time}\n"
            f"First times seen (sample): {first_times}\n"
            f"Last times seen (sample): {list(last_times)}\n"
            "Tip: check you are using a time that exists in the file."
        )
        raise ValueError(hint)

    return inflow_mat, outflow_mat, info


def plot_heatmap(
    mat: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 180,
) -> None:
    """
    Draw a single heatmap and save as PNG.
    """

    plt.figure(figsize=(10, 8), dpi=dpi)
    im = plt.imshow(mat, origin="lower", interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    # plt.xlabel("column_id")
    # plt.ylabel("row_id")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(f"{out_path}.svg", bbox_inches="tight")
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Draw inflow/outflow heatmaps from a *.grid file.")
    parser.add_argument("--dataset", default="TDRIVE", help="Dataset key in setting.json (default: TDRIVE).")
    parser.add_argument("--time", default="2015-04-25T07:00:00Z", help="Target time (e.g. 2015-02-01T08:00:00Z). If empty, use the first time in file.")
    args = parser.parse_args()
    setting_path = Path(setting_file)
    grid_path = _resolve_grid_path(args.dataset, setting_path)
    if not grid_path or not grid_path.exists():
        raise FileNotFoundError(
            f"Grid file not found.\n"
            f"- Provided --grid_path: {args.grid_path or '(empty)'}\n"
            f"- setting.json resolved: {grid_path}\n"
            f"Please pass --grid_path explicitly or ensure setting.json is correct."
        )

    # Determine target time: if empty, take the first time in file (fast sample)
    target_time = _normalize_time_to_z(args.time)

    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    inflow_mat, outflow_mat, info = build_matrices_for_time(
        grid_path,
        target_time
    )

    time_slug = _safe_slug(target_time)
    base = _safe_slug(args.dataset)

    inflow_path = out_dir / f"{base}_inflow_{time_slug}"
    outflow_path = out_dir / f"{base}_outflow_{time_slug}"

    plot_heatmap(
        inflow_mat,
        title = "",
        # title=f"{args.dataset} Inflow Heatmap @ {target_time}",
        out_path=inflow_path,
    )
    plot_heatmap(
        outflow_mat,
        title="",
        # title=f"{args.dataset} Outflow Heatmap @ {target_time}",
        out_path=outflow_path,
    )

    print("Done.")
    print(f"Grid: {grid_path}")
    print(f"Time: {target_time}")
    print(f"Grid size: rows={info.n_rows}, cols={info.n_cols}")
    print(f"Matched cells: {info.matched_cells}")
    print(f"Saved: {inflow_path}")
    print(f"Saved: {outflow_path}")


if __name__ == "__main__":
    main()