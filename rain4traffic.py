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

hours = np.arange(12, 24)  # 12..23（图上显示到 24:00）
chunksize = 500_000
usecols = ["grid_id", "relation", "value", "time"]


# =========================================================
# Helpers: normalize / rank precipitation levels
# =========================================================
def norm_level(v) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s if s else "unknown"

def disp_level(s: str) -> str:
    # legend 显示用：下划线换空格
    return str(s).replace("_", " ")

def precip_rank(level: str) -> int:
    """
    越大表示雨越大；用于：
    - 同一小时多条记录时选“更强”的
    - legend 排序（从弱到强）
    """
    s = norm_level(level)

    # 常见精确匹配（按需增补）
    fixed = {
        "no_rain": 0,
        "drizzle": 1,
        "light_drizzle": 1,
        "light_rain": 2,
        "moderate_rain": 3,
        "heavy_rain": 4,
        "torrential_rain": 5,
        "extreme_rain": 6,
        "violent_rain": 6,
    }
    if s in fixed:
        return fixed[s]

    # 关键词兜底（适配你数据里可能出现的其它写法）
    if "no_rain" in s or (("no" in s) and ("rain" in s)):
        return 0
    if "drizzle" in s:
        return 1
    if "light" in s and "rain" in s:
        return 2
    if "moderate" in s and "rain" in s:
        return 3
    if "heavy" in s and "rain" in s:
        return 4
    if "torrential" in s or "extreme" in s or "violent" in s:
        return 6

    # 如果 value 是数值（例如 mm/h），也做个分箱兜底
    try:
        x = float(s)
        if x <= 0:
            return 0
        elif x <= 0.5:
            return 2
        elif x <= 2:
            return 3
        elif x <= 10:
            return 4
        else:
            return 6
    except Exception:
        return -1  # unknown


def choose_stronger(prev_level: str | None, new_level: str) -> str:
    if prev_level is None:
        return new_level
    return new_level if precip_rank(new_level) > precip_rank(prev_level) else prev_level


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
# 1) Find candidate days: mixed rain (12-23)
# =========================================================
rain_total, rain_cnt = {}, {}

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

    c_prec = chunk[chunk["relation"] == "precipitation"]
    if c_prec.empty:
        continue

    dates = c_prec["date"].to_numpy()
    vals = c_prec["value"].map(norm_level).to_numpy()
    is_rain = np.array([1 if v != "no_rain" else 0 for v in vals], dtype=np.int32)

    for d, v in zip(dates, is_rain):
        rain_total[d] = rain_total.get(d, 0) + int(v)
        rain_cnt[d] = rain_cnt.get(d, 0) + 1

common_dates = sorted(set(rain_cnt.keys()) & traffic_dates)
daily = pd.DataFrame({
    "date": common_dates,
    "rain_ratio": [rain_total[d] / rain_cnt[d] for d in common_dates],
})

# 混合天：既有 no_rain 也有 rain（比例不极端）
mixed = daily[(daily["rain_ratio"] > 0.10) & (daily["rain_ratio"] < 0.90)].copy()
candidates = (mixed if not mixed.empty else daily.copy()).sort_values("rain_ratio", ascending=False).head(10)
candidate_days = set(candidates["date"].tolist())


# =========================================================
# 2) Build per-hour rain_level for candidate days
#    rain_level_by_day[day][(grid_id, hour)] = level(str)
# =========================================================
rain_level_by_day = {d: {} for d in candidate_days}

for chunk in pd.read_csv(WEATHER_PATH, usecols=usecols, chunksize=chunksize):
    chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["time"])

    chunk["date"] = chunk["time"].dt.date
    chunk = chunk[chunk["date"].isin(candidate_days)]
    if chunk.empty:
        continue

    chunk["hour"] = chunk["time"].dt.hour
    chunk = chunk[(chunk["hour"] >= 12) & (chunk["hour"] <= 23)]
    chunk = chunk[chunk["relation"] == "precipitation"]
    if chunk.empty:
        continue

    chunk["grid_id"] = pd.to_numeric(chunk["grid_id"], errors="coerce").astype("Int64")
    chunk = chunk.dropna(subset=["grid_id"])

    # 同一小时多条：选更强 rain_level
    for row in chunk.itertuples(index=False):
        day = row.date
        gid = int(row.grid_id)
        hr = int(row.hour)
        level = norm_level(row.value)
        key = (gid, hr)
        prev = rain_level_by_day[day].get(key, None)
        rain_level_by_day[day][key] = choose_stronger(prev, level)


# =========================================================
# 3) Choose non-extreme (day, grid): compare rain vs no_rain means
# =========================================================
MIN_NORAIN_MEAN = 30.0
MIN_RAIN_MEAN = 8.0
MIN_POINTS = 10
MIN_RAIN_HOURS = 2
MIN_NORAIN_HOURS = 2
MAX_DROP_PCT = 80.0
MIN_DROP_PCT = 10.0

best = None
best_score = -np.inf

for day in candidates["date"].tolist():
    td = dyna[(dyna["date"] == day) & (dyna["hour"] >= 12) & (dyna["hour"] <= 23)].copy()
    if td.empty:
        continue

    tg = td.groupby(["entity_id", "hour"])["total_flow"].sum().reset_index()
    tg = tg.dropna(subset=["entity_id"])
    tg["grid_id"] = tg["entity_id"].astype(int)

    # attach rain flag (derived from rain_level)
    rl = []
    rf = []
    for r in tg.itertuples(index=False):
        lv = rain_level_by_day[day].get((int(r.grid_id), int(r.hour)), "unknown")
        rl.append(lv)
        rf.append(0 if lv == "no_rain" else 1)
    tg["rain_level"] = rl
    tg["rain"] = np.array(rf, dtype=int)

    for gid, g in tg.groupby("grid_id"):
        if len(g) < MIN_POINTS:
            continue

        r = g[g["rain"] == 1]["total_flow"]
        n = g[g["rain"] == 0]["total_flow"]
        if len(r) < MIN_RAIN_HOURS or len(n) < MIN_NORAIN_HOURS:
            continue

        mean_r = r.mean()
        mean_n = n.mean()
        if mean_n < MIN_NORAIN_MEAN or mean_r < MIN_RAIN_MEAN:
            continue

        drop_pct = (mean_n - mean_r) / mean_n * 100.0
        if not (MIN_DROP_PCT <= drop_pct <= MAX_DROP_PCT):
            continue

        # 评分：drop 更大 + 雨小时更多 + baseline 更高
        score = drop_pct + 2.0 * (len(r) / 12.0) + 0.02 * mean_n
        if score > best_score:
            best_score = score
            best = (day, int(gid), float(drop_pct), float(mean_n), float(mean_r))

if best is None:
    raise RuntimeError("未找到满足“非极端 + 可对比（雨/无雨混合）”条件的样例。可放宽阈值或扩大候选天数。")

day, gid, drop_pct, mean_n, mean_r = best


# =========================================================
# 4) Build series for plotting
# =========================================================
td = dyna[
    (dyna["date"] == day)
    & (dyna["hour"] >= 12) & (dyna["hour"] <= 23)
    & (dyna["entity_id"].astype("Int64") == gid)
].copy()

series = td.groupby("hour")["total_flow"].sum().reindex(hours, fill_value=np.nan)
y = series.to_numpy()

rain_levels = [rain_level_by_day[day].get((gid, int(h)), "unknown") for h in hours]


# =========================================================
# 5) Plot: background indicates rain_level (like your wind plot)
# =========================================================
plt.figure(figsize=(7.2, 4.2))
plt.plot(hours, y, marker="o", linewidth=1.6, markersize=3)

ax = plt.gca()

# --- merge consecutive same rain_level into segments [start, end) ---
segments = []
seg_start = int(hours[0])
cur = rain_levels[0]
for i in range(1, len(hours)):
    if rain_levels[i] != cur:
        segments.append((seg_start, int(hours[i]), cur))
        seg_start = int(hours[i])
        cur = rain_levels[i]
segments.append((seg_start, int(hours[-1]) + 1, cur))  # end at 24:00

# --- alpha mapping: weak -> light, strong -> darker ---
uniq_levels = sorted(set(rain_levels), key=precip_rank)  # 从弱到强
alpha_vals = np.linspace(0.04, 0.12, num=len(uniq_levels))  # 可调范围
alpha_map = dict(zip(uniq_levels, alpha_vals))

# --- draw background (use same base color as default C0, only alpha differs) ---
for s, e, lv in segments:
    ax.axvspan(s, e, facecolor="C0", alpha=float(alpha_map[lv]), edgecolor="none")

plt.xlabel("Time")
plt.ylabel("Traffic Flow(inflow + outflow)")
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

xticks = np.arange(12, 25, 2)
plt.xticks(xticks, [f"{h:02d}:00" for h in xticks])
plt.xlim(12, 24)

# --- legend: rain_level patches, placed at upper right (你要求的右上角) ---
handles, labels = ax.get_legend_handles_labels()
for lv in uniq_levels:
    handles.append(Patch(facecolor="C0", alpha=float(alpha_map[lv]), label=disp_level(lv)))
    labels.append(disp_level(lv))

ax.legend(handles, labels, loc="upper right")

# (可选) 注释信息：你要的话可以打开
# note = (
#     f"rain_ratio: {np.mean([1 if lv!='no_rain' else 0 for lv in rain_levels]):.1%}\n"
#     f"no_rain mean: {mean_n:.1f}; rain mean: {mean_r:.1f}\n"
#     f"drop: {drop_pct:.1f}%"
# )
# ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, va="bottom",
#         bbox=dict(boxstyle="round,pad=0.4", alpha=0.15))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rain4traffic.pdf"))
plt.savefig(os.path.join(OUT_DIR, "rain4traffic.svg"))
plt.show()

print("非极端样例选择结果（rain_level 背景标注）：")
print(f"  日期：{day}")
print(f"  网格区域：grid_id={gid}")
print(f"  无雨均值={mean_n:.2f}，有雨均值={mean_r:.2f}，下降={drop_pct:.1f}%")
print("  12–24点每小时 rain_level：")
for h, lv in zip(hours, rain_levels):
    print(f"    {h:02d}:00  {lv}")
