import csv
import os
import math
import json
import ast
import numpy as np
import pandas as pd


dataset = "TDRIVE"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

raw_data_grid_dir = cfg["paths"]["raw_data_grid"]["dir"]
raw_data_grid_file = cfg["paths"]["raw_data_grid"]["file"]
raw_data_dyna_dir = cfg["paths"]["raw_data_dyna"]["dir"]
raw_data_dyna_file = cfg["paths"]["raw_data_dyna"]["file"]


def _polygon_centroid(coord_str: str):
    try:
        coords = json.loads(coord_str)
    except Exception:
        coords = ast.literal_eval(coord_str)

    ring = coords[0] if isinstance(coords, list) and len(coords) > 0 else None
    if not ring:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))]


def write_rel_full_matrix(rel_out: str, H: int, W: int, neighbor_mode: str = "4", weight_mode: str = "link"):
    """
    写出 N*N 的 rel（N=H*W）
    - 相邻(4/8邻接) => link_weight=1（或 dist）
    - 不相邻 => link_weight=0
    - 自环 => 0
    以边表形式存储：rel_id,type,origin_id,destination_id,link_weight
    """
    N = H * W
    use8 = neighbor_mode in ("8", "queen")
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if use8:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    with open(rel_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_id", "type", "origin_id", "destination_id", "link_weight"])
        rel_id = 0

        for src in range(N):
            r, c = divmod(src, W)

            # 当前 src 的邻居（只存邻居，非邻居默认为0）
            neigh = {}
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    dst = rr * W + cc
                    neigh[dst] = float(math.sqrt(dr * dr + dc * dc)) if weight_mode == "dist" else 1.0

            # 写满 N 个目的点
            for dst in range(N):
                if dst == src:
                    wt = 0.0
                else:
                    wt = neigh.get(dst, 0.0)
                w.writerow([rel_id, "geo", src, dst, wt])
                rel_id += 1


def convert_tdrive_grid_to_network_keep_june(
    geo_in: str,
    grid_in: str,
    out_dir: str,
    out_name: str,
    neighbor_mode: str = "4",
    weight_mode: str = "link"
):
    os.makedirs(out_dir, exist_ok=True)

    geo_df = pd.read_csv(geo_in)
    grid_df = pd.read_csv(grid_in)

    # ===== 1) 输出网络 .geo（Point + centroid）=====
    nodes = geo_df[["geo_id"]].copy()
    nodes["type"] = "Point"
    nodes["coordinates"] = geo_df["coordinates"].apply(
        lambda s: json.dumps(_polygon_centroid(s)) if _polygon_centroid(s) is not None else ""
    )
    geo_out = os.path.join(out_dir, f"{out_name}.geo")
    nodes.to_csv(geo_out, index=False, columns=["geo_id", "type", "coordinates"])

    # (row,col) -> geo_id；
    H = int(geo_df["row_id"].max()) + 1
    W = int(geo_df["column_id"].max()) + 1

    rc2id = {(int(r.row_id), int(r.column_id)): int(r.geo_id) for r in geo_df.itertuples(index=False)}

    # ===== 2) 输出网络 .rel（N*N，全连接矩阵；不相邻权重=0）=====
    rel_out = os.path.join(out_dir, f"{out_name}.rel")
    write_rel_full_matrix(rel_out, H=H, W=W, neighbor_mode=neighbor_mode, weight_mode=weight_mode)

    # ===== 3) 输出网络 .dyna（只保留 6 月份）=====
    t = pd.to_datetime(grid_df["time"], utc=True, errors="coerce")
    june_df = grid_df.loc[t.dt.month == 6].copy()

    june_df["entity_id"] = [
        rc2id[(int(r), int(c))] for r, c in zip(june_df["row_id"], june_df["column_id"])
    ]

    dyna = june_df.drop(columns=["row_id", "column_id", "dyna_id"], errors="ignore").reset_index(drop=True)
    dyna.insert(0, "dyna_id", np.arange(len(dyna)))

    dyna_out = os.path.join(out_dir, f"{out_name}.dyna")
    dyna.to_csv(dyna_out, index=False, columns=["dyna_id", "type", "time", "entity_id", "inflow", "outflow"])

    return {
        "H": H, "W": W, "N": H * W,
        "geo_out": geo_out,
        "rel_out": rel_out,
        "dyna_out": dyna_out,
        "n_rel_rows": (H * W) * (H * W),
        "n_records_june": len(dyna),
        "time_range_june": (june_df["time"].min(), june_df["time"].max())
    }


def main():
    # ===== 示例调用 =====
    result = convert_tdrive_grid_to_network_keep_june(
        geo_in=f"{raw_data_grid_dir}/{raw_data_grid_file}.geo",
        grid_in=f"{raw_data_grid_dir}/{raw_data_grid_file}.grid",
        out_dir=raw_data_dyna_dir,
        out_name=raw_data_dyna_file
    )
    print(result)


if __name__ == "__main__":
    main()