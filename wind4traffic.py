import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================================================
# Config
# =========================================================
dataset = "NYCTAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

raw_data_dyna_dir, raw_data_dyna_file = cfg["paths"]["raw_data_dyna"]["dir"], cfg["paths"]["raw_data_dyna"]["file"]
DYNA_PATH = f"{raw_data_dyna_dir}/{raw_data_dyna_file}.dyna"

prefix_path_weather = cfg["paths"]["prefix_path_weather"]
WEATHER_PATH = f"{prefix_path_weather}/weather_kg.csv"

OUT_DIR = "./graph/en"
os.makedirs(OUT_DIR, exist_ok=True)

hours_full = np.arange(12, 24)  # 12..23，显示到 24:00
chunksize = 500_000
usecols = ["grid_id", "relation", "value", "time"]


# =========================================================
# Wind helpers
# =========================================================
beaufort_order = [
    "calm", "light_air", "light_breeze", "gentle_breeze", "moderate_breeze",
    "fresh_breeze", "strong_breeze", "near_gale", "gale", "strong_gale",
    "storm", "violent_storm", "hurricane"
]
rank = {k: i for i, k in enumerate(beaufort_order)}

def wind_rank(level: str) -> int:
    if level is None:
        return -1
    return rank.get(str(level).strip(), -1)

def disp(level: str) -> str:
    return str(level).replace("_", " ").strip()

def choose_stronger(prev_level: str | None, new_level: str) -> str:
    """同一 grid-hour 多条风记录时，取更强风级"""
    if prev_level is None:
        return new_level
    return new_level if wind_rank(new_level) > wind_rank(prev_level) else prev_level

def spearman_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman 相关（不会触发 divide-by-zero warning）
    若 x 或 y 无变化，则返回 np.nan
    """
    if len(x) < 3:
        return np.nan
    # 转秩
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 1e-12:
        return np.nan
    return float((rx * ry).sum() / denom)


# =========================================================
# 0) Load traffic
# =========================================================
dyna = pd.read_csv(DYNA_PATH)
dyna["time"] = pd.to_datetime(dyna["time"], utc=True, errors="coerce")
dyna = dyna.dropna(subset=["time"])

dyna["date"] = dyna["time"].dt.date
dyna["hour"] = dyna["time"].dt.hour
dyna["entity_id"] = pd.to_numeric(dyna["entity_id"], errors="coerce").astype("Int64")
dyna["total_flow"] = dyna["inflow"].astype(float) + dyna["outflow"].astype(float)

traffic_dates = set(dyna["date"].unique())


# =========================================================
# 1) 先选一批“风力混合”的候选日期（扩大范围，避免找不到）
#    这里用 12-23 全段来找混合日（只是候选日筛选，不用于证明）
# =========================================================
WINDY_THRESHOLD = "moderate_breeze"
WINDY_TH_RANK = wind_rank(WINDY_THRESHOLD)

windy_total, windy_cnt = {}, {}

for chunk in pd.read_csv(WEATHER_PATH, usecols=usecols, chunksize=chunksize):
    chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["time"])
    chunk["date"] = chunk["time"].dt.date
    chunk = chunk[chunk["date"].isin(traffic_dates)]
    if chunk.empty:
        continue

    chunk["hour"] = chunk["time"].dt.hour
    chunk = chunk[(chunk["hour"] >= 12) & (chunk["hour"] <= 23)]
    if chunk.empty:
        continue

    c_wind = chunk[chunk["relation"] == "wind_speed_10m"]
    if c_wind.empty:
        continue

    dates = c_wind["date"].to_numpy()
    flags = np.array([1 if wind_rank(v) >= WINDY_TH_RANK else 0 for v in c_wind["value"].astype(str)], dtype=np.int32)

    for d, v in zip(dates, flags):
        windy_total[d] = windy_total.get(d, 0) + int(v)
        windy_cnt[d] = windy_cnt.get(d, 0) + 1

common_dates = sorted(set(windy_cnt.keys()) & traffic_dates)
daily = pd.DataFrame({
    "date": common_dates,
    "windy_ratio": [windy_total[d] / windy_cnt[d] for d in common_dates],
})

# 混合性得分：越接近 0.5 越好
daily["mix_score"] = 1.0 - np.abs(daily["windy_ratio"] - 0.5) * 2.0
daily = daily.sort_values("mix_score", ascending=False)

# 取更多候选日，避免“找不到”
TOP_K_DAYS = 200
candidate_days = set(daily.head(TOP_K_DAYS)["date"].tolist())


# =========================================================
# 2) 构建候选日逐小时 wind_level
#    wind_level_by_day[day][(grid_id, hour)] = level(str)
# =========================================================
wind_level_by_day = {d: {} for d in candidate_days}

for chunk in pd.read_csv(WEATHER_PATH, usecols=usecols, chunksize=chunksize):
    chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["time"])
    chunk["date"] = chunk["time"].dt.date
    chunk = chunk[chunk["date"].isin(candidate_days)]
    if chunk.empty:
        continue

    chunk["hour"] = chunk["time"].dt.hour
    chunk = chunk[(chunk["hour"] >= 12) & (chunk["hour"] <= 23)]
    chunk = chunk[chunk["relation"] == "wind_speed_10m"]
    if chunk.empty:
        continue

    chunk["grid_id"] = pd.to_numeric(chunk["grid_id"], errors="coerce").astype("Int64")
    chunk = chunk.dropna(subset=["grid_id"])

    for row in chunk.itertuples(index=False):
        day = row.date
        gid = int(row.grid_id)
        hr = int(row.hour)
        level = str(row.value).strip()
        key = (gid, hr)
        prev = wind_level_by_day[day].get(key, None)
        wind_level_by_day[day][key] = choose_stronger(prev, level)


# =========================================================
# 3) 选择“更可信”的例子：
#    - 在非高峰窗口内：风级变化 & 流量变化呈负相关（阈值自动放宽）
#    - 全时段（12-24）不出现非常尖的“晚高峰尖峰”（避免被解释为高峰主导）
# =========================================================
OFFPEAK_WINDOWS = [
    np.arange(12, 16),  # 12-16
    np.arange(20, 24),  # 20-24
]

# 非极端 + 去除晚高峰尖峰（全时段）
FULL_MAX_OVER_MEAN = 1.65   # max <= mean * 1.65（越小越不“高峰尖”）
FULL_MIN_MEAN = 30.0        # 平均流量别太小（避免噪声）

# offpeak 内对比要求（会逐步放宽）
CORR_THRESH_LIST = [-0.50, -0.35, -0.25, -0.15]  # 自动放宽
MIN_DISTINCT_LEVELS = 2      # offpeak 内至少出现 2 种风级
MIN_VAR_RANK = 1             # offpeak 内 max_rank - min_rank >= 1
MIN_DROP_PCT = 6.0           # windy(>=阈值) 相对 non-windy 的下降至少 6%（也可调）
MAX_DROP_PCT = 70.0          # 下降别太夸张（避免极端）

def build_day_grid_table(day):
    """返回该天(12-23) 每网格每小时的 total_flow 表"""
    td = dyna[(dyna["date"] == day) & (dyna["hour"] >= 12) & (dyna["hour"] <= 23)].copy()
    if td.empty:
        return None
    tg = td.groupby(["entity_id", "hour"])["total_flow"].sum().reset_index()
    tg = tg.dropna(subset=["entity_id"])
    tg["grid_id"] = tg["entity_id"].astype(int)
    return tg

best = None
best_score = -np.inf

# 为了效率，先按 mix_score 高的日期优先
day_list = daily[daily["date"].isin(candidate_days)]["date"].tolist()

for corr_th in CORR_THRESH_LIST:
    for win in OFFPEAK_WINDOWS:
        win_set = set(win.tolist())

        for day in day_list:
            tg = build_day_grid_table(day)
            if tg is None:
                continue

            # 先为所有行附上 wind_level
            tg["wind_level"] = tg.apply(
                lambda r: wind_level_by_day[day].get((int(r["grid_id"]), int(r["hour"])), "unknown"),
                axis=1
            )
            tg["wrank"] = tg["wind_level"].apply(wind_rank).astype(int)
            # 剔除 unknown
            tg = tg[tg["wrank"] >= 0]
            if tg.empty:
                continue

            # 预取全时段序列，用于“去晚高峰尖峰”
            # （只对候选 grid 检查，避免太慢：先 groupby 迭代）
            for gid, g in tg.groupby("grid_id"):
                # full 12-23
                y_full = g.set_index("hour")["total_flow"].reindex(hours_full).to_numpy(dtype=float)
                if np.all(~np.isfinite(y_full)):
                    continue
                mean_full = float(np.nanmean(y_full))
                if not np.isfinite(mean_full) or mean_full < FULL_MIN_MEAN:
                    continue
                max_full = float(np.nanmax(y_full))
                if max_full > mean_full * FULL_MAX_OVER_MEAN:
                    # 太像“高峰尖峰”，跳过
                    continue

                # offpeak window slice
                go = g[g["hour"].isin(win_set)].copy()
                if len(go) < 3:
                    continue

                # offpeak 内风级变化要求
                wr = go["wrank"].to_numpy(dtype=float)
                fl = go["total_flow"].to_numpy(dtype=float)
                if np.any(~np.isfinite(fl)):
                    continue

                if (wr.max() - wr.min()) < MIN_VAR_RANK:
                    continue

                lv_set = set(go["wind_level"].tolist())
                if len(lv_set) < MIN_DISTINCT_LEVELS:
                    continue

                # offpeak 内相关性（负相关）——如果无变化会返回 nan
                corr = spearman_corr_safe(wr, fl)
                if not np.isfinite(corr) or corr > corr_th:
                    continue

                # offpeak 内“阈值以上/以下”均值下降（辅助约束）
                windy = go[go["wrank"] >= WINDY_TH_RANK]["total_flow"].astype(float)
                nowindy = go[go["wrank"] < WINDY_TH_RANK]["total_flow"].astype(float)
                if len(windy) < 1 or len(nowindy) < 1:
                    continue
                mean_w = float(windy.mean())
                mean_n = float(nowindy.mean())
                if mean_n <= 1e-6 or mean_w <= 1e-6:
                    continue

                drop_pct = (mean_n - mean_w) / mean_n * 100.0
                if not (MIN_DROP_PCT <= drop_pct <= MAX_DROP_PCT):
                    continue

                # 综合评分：相关性越负越好 + drop 越明显越好 + 全时段均值更高更稳
                score = (-corr) * 10.0 + drop_pct + 0.01 * mean_full

                if score > best_score:
                    best_score = score
                    best = {
                        "day": day,
                        "gid": int(gid),
                        "corr": float(corr),
                        "corr_th": float(corr_th),
                        "window": win,
                        "mean_full": mean_full,
                        "drop_pct": float(drop_pct),
                        "mean_nowindy": mean_n,
                        "mean_windy": mean_w,
                    }

    if best is not None:
        # 找到就停止放宽
        break

if best is None:
    raise RuntimeError(
        "仍未找到满足条件的样例。\n"
        "你可以：\n"
        "1) 把 FULL_MAX_OVER_MEAN 从 1.65 放宽到 1.9（允许轻微峰）；\n"
        "2) 把 MIN_DROP_PCT 从 6 降到 3；\n"
        "3) 或把 TOP_K_DAYS 再增大到 400。"
    )


# =========================================================
# 4) Build series & plot (12-24)
# =========================================================
day = best["day"]
gid = best["gid"]

td = dyna[
    (dyna["date"] == day)
    & (dyna["hour"] >= 12) & (dyna["hour"] <= 23)
    & (dyna["entity_id"].astype("Int64") == gid)
].copy()

series = td.groupby("hour")["total_flow"].sum().reindex(hours_full, fill_value=np.nan)
y = series.to_numpy(dtype=float)

wind_levels = [wind_level_by_day[day].get((gid, int(h)), "unknown") for h in hours_full]

# merge consecutive same wind level into segments [start, end)
segments = []
seg_start = int(hours_full[0])
cur = wind_levels[0]
for i in range(1, len(hours_full)):
    if wind_levels[i] != cur:
        segments.append((seg_start, int(hours_full[i]), cur))
        seg_start = int(hours_full[i])
        cur = wind_levels[i]
segments.append((seg_start, int(hours_full[-1]) + 1, cur))

# alpha mapping by wind strength (rank)
uniq_levels = sorted(set(wind_levels), key=wind_rank)
alpha_vals = np.linspace(0.06, 0.18, num=len(uniq_levels))
alpha_map = dict(zip(uniq_levels, alpha_vals))

plt.figure(figsize=(7.2, 4.2))
plt.plot(hours_full, y, marker="o", linewidth=1.6, markersize=3, label="Traffic flow")

ax = plt.gca()
for s, e, lv in segments:
    ax.axvspan(s, e, alpha=float(alpha_map[lv]))

plt.xlabel("Time")
plt.ylabel("Traffic Flow (inflow + outflow)")
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

xticks = np.arange(12, 25, 2)
plt.xticks(xticks, [f"{h:02d}:00" for h in xticks])
plt.xlim(12, 24)

# legend: 右上角，只显示当天出现的风力等级
handles, labels = ax.get_legend_handles_labels()
for lv in uniq_levels:
    handles.append(Patch(alpha=float(alpha_map[lv]), label=disp(lv)))
    labels.append(disp(lv))
ax.legend(handles, labels, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wind4traffic.pdf"))
plt.savefig(os.path.join(OUT_DIR, "wind4traffic.svg"))
plt.show()

print("✅ 已自动换成更难被“晚高峰”解释的例子：")
print(f"  日期：{best['day']}")
print(f"  网格：grid_id={best['gid']}")
w = best["window"]
print(f"  非高峰窗口：{w[0]:02d}:00–{(w[-1]+1):02d}:00")
print(f"  Spearman(wind_rank, flow) = {best['corr']:.3f}（阈值使用 {best['corr_th']:.2f}）")
print(f"  offpeak内 drop = {best['drop_pct']:.1f}%  (no-windy mean={best['mean_nowindy']:.1f}, windy mean={best['mean_windy']:.1f})")
print(f"  全时段 mean={best['mean_full']:.1f}，且 max/mean <= {FULL_MAX_OVER_MEAN}（降低晚高峰尖峰干扰）")
