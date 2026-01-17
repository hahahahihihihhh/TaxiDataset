import os.path

from utils.setting import METEOROLOGICAL_VARS2LABELS, METEOROLOGICAL_VARS_DIVIDE
from utils.utils import ensure_dir
import json
import os.path
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from matplotlib import colors as mcolors


dataset = "NYCTAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

prefix_path_weather = cfg["paths"]["prefix_path_weather"]
prefix_path_meteor_distr = cfg["paths"]["prefix_path_meteor_distr"]
start_time, end_time = cfg["start_time"], cfg["end_time"]
poi_filter_file = cfg["paths"]["poi_filter_file"]
weather_all_grids_file = cfg["paths"]["weather_all_grids_file"]
lon_min = cfg["grid"]["lon_min"]
lon_max = cfg["grid"]["lon_max"]
lat_min = cfg["grid"]["lat_min"]
lat_max = cfg["grid"]["lat_max"]
H = cfg["grid"]["H"]
W = cfg["grid"]["W"]


def weather_horiz_distr_count(time):
    time_str = str(time).split(":")[0]
    df = pd.read_csv(os.path.join(prefix_path_weather, "weather_kg.csv"))
    ensure_dir(prefix_path_meteor_distr)
    df["time"] = pd.to_datetime(df["time"])
    # pick a representative time with richer spatial variation
    time_sel = pd.Timestamp(time)
    sub = df[df["time"] == time_sel].copy()
    sub.to_csv("test.csv")
    # 1) one-row-per-grid snapshot (wide table)
    sub.pivot(index="grid_id", columns="relation", values="value").reset_index()
    # 2) distribution stats per variable (counts + percentage)
    print(time)
    dist_list = []
    for rel, g in sub.groupby("relation"):
        print(f"【{rel}】")
        counts = g["value"].value_counts()
        pct = (counts / counts.sum() * 100).round(2)
        tdf = pd.DataFrame({
            "time": time_sel,
            "relation": rel,
            "value": counts.index,
            "count": counts.values,
            "pct_%": pct.values
        })
        dist_list.append(tdf)
    print("---------------------------------")
    meteor_df = pd.concat(dist_list, ignore_index=True)
    meteor_df.to_csv(os.path.join(prefix_path_meteor_distr,
                                  f"meteor_distr_{time_str}.csv"), index=False)


def plot_meteor_heatmap_from_df(df_result, meteor_var="temperature_2m"):
    """
    根据 df_result (包含 latitude/longitude/time/met_var) 绘制热力图：
    - temperature_2m：按 1°C 分级
    - 其他变量：自动等距分箱（n_bins 个区间）
    说明：df_result 中的经纬度是网格中心点，且不重复（理想情况下为 H*W 条）。
    """
    # --- 0) 基本检查 ---
    required_cols = {"latitude", "longitude", "time", meteor_var}
    missing = required_cols - set(df_result.columns)
    if missing:
        raise ValueError(f"df_result missing columns: {missing}")
    # --- 1) 构建网格矩阵 ---
    lats = np.sort(df_result["latitude"].unique())   # 南->北
    lons = np.sort(df_result["longitude"].unique())  # 西->东
    if len(lats) != H or len(lons) != W:
        print(f"[WARN] unique lats={len(lats)} (expect {H}), unique lons={len(lons)} (expect {W})")
    lat2row = {v: i for i, v in enumerate(lats)}
    lon2col = {v: i for i, v in enumerate(lons)}
    grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
    for lat, lon, val in zip(df_result["latitude"], df_result["longitude"], df_result[meteor_var]):
        # val 可能是字符串/空，转 float 更稳
        try:
            v = float(val)
        except Exception:
            v = np.nan
        grid[lat2row[lat], lon2col[lon]] = v
    # --- 2) 生成分级边界（温度 1°C，其它自动）---
    vals = grid[np.isfinite(grid)]
    if vals.size == 0:
        t0 = df_result["time"].iloc[0] if len(df_result) else ""
        raise ValueError(f"No valid values for '{meteor_var}' at time={t0}")
    divide = METEOROLOGICAL_VARS_DIVIDE[meteor_var]
    bounds = list(divide.keys())
    if meteor_var == "wind_speed_10m":
        bounds = [0] + bounds
    labels = [divide[k] for k in sorted(divide.keys())]
    bounds = np.asarray(bounds, dtype=float)
    mids = 0.5 * (bounds[1:] + bounds[:-1])
    cbar_ticks = mids
    cbar_ticklabels = labels
    # --- 4) 画图（分箱上色）---
    fig, ax = plt.subplots(figsize=(10, 5))
    # 用 BoundaryNorm 做“分段”映射；colors 数量 = 区间数
    n_colors = max(len(bounds) - 1, 2)
    cmap = plt.get_cmap("Blues", n_colors)
    norm = mcolors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    # --- 5) 色条 ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_ticklabels is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels, fontsize=10)
    # ✅ 去掉红框中箭头指的“刻度线”（major + minor）
    cbar.ax.tick_params(which="major", length=0)  # 不画刻度线
    # cbar.minorticks_off()  # 关闭 minor ticks（保险）
    # --- 6) 标题 / 坐标 ---
    t0 = df_result["time"].iloc[0]
    met_label = METEOROLOGICAL_VARS2LABELS[meteor_var]
    # ax.set_title(f"{met_label} heatmap @ {t0}", fontsize=12.5)
    ax.set_xlabel("longitude", fontsize=12.5)
    ax.set_ylabel("latitude", fontsize=12.5)
    # 可选：把经纬度当做刻度（网格大时不建议全显示）
    xticks = np.arange(0, len(lons), step=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{lons[i]:.2f}" for i in xticks], fontsize=10)
    ax.set_yticks(np.arange(len(lats)))
    ax.set_yticklabels([f"{y:.2f}" for y in lats], fontsize=10)
    plt.tight_layout()
    plt.savefig("./graph/en/meteorological_heatmap_t1.pdf")
    plt.savefig("./graph/en/meteorological_heatmap_t1.svg")
    plt.show()
    return


def weather_horiz_distr_show(time, meteorological_var):
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    df_weather = pd.read_csv(weather_all_grids_file_path)
    df_result = df_weather[df_weather["time"] == time][["latitude", "longitude", "time", meteorological_var]]
    plot_meteor_heatmap_from_df(df_result, meteorological_var)


def plot_meteorological_var_daily(df_day, meteorological_var, date=None):
    # 1) 确保 time 是 datetime
    df_day = df_day.copy()
    df_day["time"] = pd.to_datetime(df_day["time"], errors="coerce")
    # 2) 丢掉无法解析的时间行
    if df_day.empty:
        raise ValueError("df_day has no valid datetime in column 'time' after parsing.")
    # 3) 按小时聚合
    df_hourly = (
        df_day
        .groupby(df_day["time"].dt.hour)[meteorological_var]
        .mean()
        .reset_index()
        .rename(columns={"time": "hour"})
        .sort_values("hour")
    )
    # 4) 画图
    plt.figure(figsize=(10, 5))
    plt.plot(df_hourly["hour"], df_hourly[meteorological_var], marker="o")
    xticks = np.arange(0, 25, 2)
    plt.xticks(xticks, [f"{h}:00" for h in xticks])
    plt.xlabel("Time", fontsize=15)
    meteorological_label = METEOROLOGICAL_VARS2LABELS[meteorological_var]
    plt.yticks(range(0, 14, 2), fontsize=10)
    plt.ylabel(meteorological_label, fontsize=15)
    if date is None:
        # 从数据里推断日期（取第一条）
        date = df_day["time"].dt.date.iloc[0]
    # plt.title(f"{meteorological_label} daily variation @ {date}", fontsize=15)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("./graph/en/meteorological_var_daily.pdf")
    plt.savefig("./graph/en/meteorological_var_daily.svg")
    plt.show()
    return df_hourly


def weather_time_distr_show(date, meteorological_var):
    grid_id = 0
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    df_weather = pd.read_csv(weather_all_grids_file_path)
    df_result = df_weather.loc[df_weather["time"].str.startswith(date) & (df_weather["grid_id"] == grid_id),
                                ["time", meteorological_var]]
    print(df_result)
    plot_meteorological_var_daily(df_result, meteorological_var)


def analyze_weather_distr():
    # times = pd.date_range(
    #     start=start_time,
    #     end=end_time,
    #     freq="D"  # 每天 +1
    # )
    # for time in times:
    #     weather_horiz_distr_count(time)
    for t in pd.date_range("2014-03-01T03:00", "2014-03-01T03:00", freq="H"):
        t_str = t.strftime("%Y-%m-%dT%H:%M")
        weather_horiz_distr_show(t_str, "wind_speed_10m")
    # weather_time_distr_show("2014-03-01", "wind_speed_10m")


if __name__ == "__main__":
    analyze_weather_distr()
