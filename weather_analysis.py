import os.path
from utils.utils import ensure_dir
import json
import os.path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import os
import pandas as pd


dataset = "NYCBike"
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


def plot_heatmap_from_df(df_result, met_var="temperature_2m", n_bins=10):
    """
    根据 df_result (包含 latitude/longitude/time/met_var) 绘制热力图：
    - temperature_2m：按 1°C 分级
    - 其他变量：自动等距分箱（n_bins 个区间）
    说明：df_result 中的经纬度是网格中心点，且不重复（理想情况下为 H*W 条）。
    """
    # --- 0) 基本检查 ---
    required_cols = {"latitude", "longitude", "time", met_var}
    missing = required_cols - set(df_result.columns)
    if missing:
        raise ValueError(f"df_result missing columns: {missing}")

    # --- 1) 构建网格矩阵 ---
    lats = np.sort(df_result["latitude"].unique())   # 南->北
    lons = np.sort(df_result["longitude"].unique())  # 西->东

    if (len(lats) != H) or (len(lons) != W):
        print(f"[WARN] unique lats={len(lats)} (expect {H}), unique lons={len(lons)} (expect {W})")

    lat2row = {v: i for i, v in enumerate(lats)}
    lon2col = {v: i for i, v in enumerate(lons)}

    grid = np.full((len(lats), len(lons)), np.nan, dtype=float)
    for lat, lon, val in zip(df_result["latitude"], df_result["longitude"], df_result[met_var]):
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
        raise ValueError(f"No valid values for '{met_var}' at time={t0}")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    # 若所有值相同，给一个极小范围，避免 boundaries 退化
    if np.isclose(vmin, vmax):
        eps = 1e-6
        vmin, vmax = vmin - eps, vmax + eps

    if met_var == "temperature_2m":
        lo = np.floor(vmin)
        hi = np.ceil(vmax)
        if lo == hi:
            hi = lo + 1  # 至少一个区间
        bounds = np.arange(lo, hi + 1, 1.0)  # 1°C
    else:
        # 等距自动分箱：n_bins 个区间 => n_bins+1 个边界
        bounds = np.linspace(vmin, vmax, int(n_bins) + 1)

    # 保护：BoundaryNorm 要求至少两个边界
    if bounds is None or len(bounds) < 2:
        t0 = df_result["time"].iloc[0] if len(df_result) else ""
        raise ValueError(f"Invalid bounds for '{met_var}' at time={t0}: {bounds}")

    norm = BoundaryNorm(bounds, ncolors=256, clip=True)

    # --- 3) 绘图 ---
    t = df_result["time"].iloc[0] if len(df_result) else ""

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        grid,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        aspect="auto",
        cmap="Reds",
        norm=norm
    )

    # colorbar：温度按整数刻度，其它给少一些刻度避免太密
    if met_var == "temperature_2m":
        ticks = bounds
        cbar = plt.colorbar(im, ticks=ticks)
        cbar.set_label("Temperature (°C)")
        plt.title(f"{met_var} {dataset} (1°C bins) @ {t}")
    else:
        # 其它变量：显示少量刻度（5~7 个）更清晰
        tick_num = min(7, len(bounds))
        tick_idx = np.linspace(0, len(bounds) - 1, tick_num).astype(int)
        ticks = bounds[tick_idx]
        cbar = plt.colorbar(im, ticks=ticks)
        cbar.set_label(met_var)
        plt.title(f"{dataset} @ {met_var} @ {t}")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def weather_horiz_distr_show(time, meteorological_var):
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    df_weather = pd.read_csv(weather_all_grids_file_path)
    df_result = df_weather[df_weather["time"] == time][["latitude", "longitude", "time", meteorological_var]]
    plot_heatmap_from_df(df_result, meteorological_var)


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
    plt.figure(figsize=(8, 4))
    plt.plot(df_hourly["hour"], df_hourly[meteorological_var], marker="o")
    plt.xticks(range(0, 24))
    plt.xlabel("Hour of Day")
    plt.ylabel(meteorological_var)
    if date is None:
        # 从数据里推断日期（取第一条）
        date = df_day["time"].dt.date.iloc[0]
    plt.title(f"{meteorological_var} daily variation on {date}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return df_hourly


def weather_time_distr_show(date, meteorological_var):
    grid_id = 2
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    df_weather = pd.read_csv(weather_all_grids_file_path)
    df_result = df_weather.loc[df_weather["time"].str.startswith(date) & (df_weather["grid_id"] == grid_id),
                                ["time", meteorological_var]]
    print(df_result)
    plot_meteorological_var_daily(df_result, meteorological_var)


def analyze_weather_horiz_distr():
    # times = pd.date_range(
    #     start=start_time,
    #     end=end_time,
    #     freq="D"  # 每天 +1
    # )
    # for time in times:
    #     weather_horiz_distr_count(time)
    # weather_horiz_distr_show("2014-04-15T00:00", "wind_speed_10m")
    weather_time_distr_show("2014-04-01", "wind_speed_10m")


if __name__ == "__main__":
    analyze_weather_horiz_distr()
