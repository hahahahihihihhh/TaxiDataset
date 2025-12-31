import gc
import os
import re
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyrosm import OSM
from requests.adapters import HTTPAdapter
import time
from copy import deepcopy
from urllib3 import Retry
from .setting import CATES_LABELS, CATES, IGNORE_VALUE_SET
from .setting import EXPAND_KEYS


def _build_session():
    s = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=1.0,  # 1s, 2s, 4s...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SESSION = _build_session()
def fetch_open_meteo_hourly(longitude, latitude, start_date, end_date, hourly_vars,
                           timezone="America/New_York", timeout=60):
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "longitude": longitude,
        "latitude": latitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }
    # 关键：Connection: close 避免复用连接导致的 EOF
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Connection": "close",
    }
    resp = _SESSION.get(BASE_URL, params=params, headers=headers, timeout=timeout)
    # 如果是 429，尊重一下限流，稍微等一等
    if resp.status_code == 429:
        time.sleep(3)
    resp.raise_for_status()
    return resp.json()


def merge_open_meteo_hourly_responses_in_order(responses):
    """
    Merge multiple Open-Meteo responses (resp0, resp1, resp2, ...)
    assuming:
      1) hourly['time'] in each resp is already sorted
      2) times across responses are in increasing order and do NOT overlap (no duplicates)

    Returns a merged dict in Open-Meteo-like format.
    """
    if not responses:
        raise ValueError("No valid responses to merge.")

    base = deepcopy(responses[0])
    if "hourly" not in base or "hourly_units" not in base:
        raise ValueError("Response missing 'hourly' or 'hourly_units'.")
    units = base["hourly_units"]
    hourly_vars = [k for k in units.keys() if k != "time"]  # 用 units 确定变量集合更稳
    # 初始化合并容器
    merged_hourly = {"time": []}
    for v in hourly_vars:
        merged_hourly[v] = []
    # 拼接
    for r in responses:
        # 基础一致性检查（可按需删减）
        if r.get("latitude") != base.get("latitude") or r.get("longitude") != base.get("longitude"):
            raise ValueError("Latitude/Longitude mismatch among responses.")
        if r.get("timezone") != base.get("timezone"):
            raise ValueError("Timezone mismatch among responses.")
        if r.get("hourly_units") != units:
            raise ValueError("hourly_units mismatch among responses (hourly vars may differ).")
        h = r.get("hourly", {})
        times = h.get("time", [])
        merged_hourly["time"].extend(times)
        for v in hourly_vars:
            vals = h.get(v, [])
            if len(vals) != len(times):
                raise ValueError(f"Length mismatch in response for '{v}': {len(vals)} vs time {len(times)}")
            merged_hourly[v].extend(vals)
    base["hourly"] = merged_hourly
    return base


def norm_osm_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []

    # bool / np.bool_ 不在这里处理
    if isinstance(v, (bool, np.bool_)):
        return []

    s = str(v).strip().lower()
    if not s or s in IGNORE_VALUE_SET:
        return []

    parts = s.split(";")
    out = []
    for p in parts:
        p = p.strip().lower()
        p = p.replace(" ", "_")
        p = re.sub(r"_+", "_", p).strip("_")
        if not p or p in IGNORE_VALUE_SET:
            continue
        out.append(p)
    return out


def build_subtype_to_cate(
    cates_labels,
    cate_priority = None
):
    """
    返回:
      - mapping: subtype -> cate
      - conflicts: subtype -> [other_cates...]
    """
    # 默认优先级：按字典顺序；你也可以显式传入
    if cate_priority is None:
        cate_priority = list(cates_labels.keys())
    rank = {c: i for i, c in enumerate(cate_priority)}

    mapping = {}
    conflicts = defaultdict(list)

    for cate, labels in cates_labels.items():
        for sub in labels:
            sub = str(sub).strip().lower().replace(" ", "_")
            if not sub:
                continue

            if sub not in mapping:
                mapping[sub] = cate
            else:
                # conflicts 里同时记录“已有 cate”和“新 cate”
                if mapping[sub] not in conflicts[sub]:
                    conflicts[sub].append(mapping[sub])
                if cate not in conflicts[sub]:
                    conflicts[sub].append(cate)
                if rank.get(cate, 10 ** 9) < rank.get(mapping[sub], 10 ** 9):
                    mapping[sub] = cate

    return mapping, dict(conflicts)


def detect_bool_expand_keys(gdf, keys):
    """
    如果某列非空值全部是 bool，则认为它是展开列（True/False）
    """
    bool_keys = set()
    for k in keys:
        if k not in gdf.columns:
            continue
        ser = gdf[k].dropna()
        if len(ser) == 0:
            continue
        # dtype 是 bool，或值全为 bool / np.bool_
        if pd.api.types.is_bool_dtype(ser) or ser.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            bool_keys.add(k)
    return bool_keys


def infer_cate_row(row, expand_keys, subtype_to_cate, bool_keys):
    for k in expand_keys:
        if k not in row.index or pd.isna(row[k]):
            continue
        if k in bool_keys:
            if bool(row[k]) and k in subtype_to_cate:
                return subtype_to_cate[k]
            continue
        for sub in norm_osm_value(row[k]):
            if sub in subtype_to_cate:
                return subtype_to_cate[sub]
    return "others"


def build_poi_filter_csv(osm_path, save_path):
    # ---- 0) 构建 SUBTYPE_TO_CATE（可调整优先级） ----
    SUBTYPE_TO_CATE, conflicts = build_subtype_to_cate(CATES_LABELS, cate_priority=CATES)

    out_cols = ["name", "poi_id", "osm_way_id", "building", "amenity",
                "centroid", "area", "area_ft2", "lat", "lng", "cate"]
    osm = OSM(osm_path)
    print("loading osm ...")
    pois = osm.get_pois()
    print("getting pois ...")
    if pois is None or len(pois) == 0:
        pd.DataFrame(columns=out_cols).to_csv(save_path, index=False, encoding="utf-8")
        return
    print("filtering2 ...")
    cond = pd.Series(False, index=pois.index)
    for k in EXPAND_KEYS:
        if k in pois.columns:
            cond |= pois[k].notna()
    gdf = pois[cond].copy()
    if len(gdf) == 0:
        pd.DataFrame(columns=out_cols).to_csv(save_path, index=False, encoding="utf-8")
        return
    # ---- 3) 面积 & 代表点：只拿 geometry 去做 CRS（避免把整表带过去）----
    geom_only = gdf[["geometry"]].copy()
    geom_only = geom_only.set_geometry("geometry")
    gdf_3857 = geom_only.to_crs(epsg=3857)
    geom_type = gdf_3857.geometry.geom_type
    is_poly = geom_type.isin(["Polygon", "MultiPolygon"])
    area_m2 = pd.Series(0.0, index=gdf_3857.index)
    area_m2[is_poly] = gdf_3857.loc[is_poly, "geometry"].area
    rep_3857 = gdf_3857.geometry.representative_point()
    rep_wgs84 = gpd.GeoSeries(rep_3857, crs="EPSG:3857").to_crs(epsg=4326)
    # 释放中间大对象
    del geom_only, gdf_3857, rep_3857
    gc.collect()
    # ---- 4) 自动识别 bool 展开列 keys ----
    bool_keys = detect_bool_expand_keys(gdf, EXPAND_KEYS)
    print("infering...")
    # ---- 5) cate 推断（保持你的逻辑；若仍内存/速度不够再做分块/向量化）----
    cate_series = gdf.apply(
        lambda r: infer_cate_row(r, EXPAND_KEYS, SUBTYPE_TO_CATE, bool_keys),
        axis=1
    )
    # ---- 6) 输出组装 ----
    out = pd.DataFrame(index=gdf.index, columns=out_cols)
    out["name"] = gdf["name"] if "name" in gdf.columns else pd.NA
    out["poi_id"] = gdf["id"] if "id" in gdf.columns else pd.NA
    if "osm_type" in gdf.columns and "id" in gdf.columns:
        out["osm_way_id"] = gdf["id"].where(gdf["osm_type"].astype(str).str.lower().eq("way"), pd.NA)
    else:
        out["osm_way_id"] = pd.NA
    out["building"] = gdf["building"] if "building" in gdf.columns else pd.NA
    out["amenity"] = gdf["amenity"] if "amenity" in gdf.columns else pd.NA
    out["centroid"] = rep_wgs84.to_wkt()
    out["area"] = area_m2.round(1)
    out["area_ft2"] = (area_m2 * 10.763910416709722).round(1)
    out["lat"] = rep_wgs84.y
    out["lng"] = rep_wgs84.x
    out["cate"] = cate_series
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8")
    # 最后释放
    del gdf, out, rep_wgs84, cate_series, area_m2
    gc.collect()


def make_default_value():
    return {poi_type: 0 for poi_type in CATES}


def discretize_value(value, divide_dict):
    """
    根据上界字典进行分级
    divide_dict: {upper_bound: label}
    """
    for upper, label in divide_dict.items():
        if value <= upper:
            return label
    return None


def ensure_dir(path: str):
    """
    确保目录存在（若传入的是文件路径，则创建其父目录）

    Parameters
    ----------
    path : str
        目录路径 或 文件路径
    """
    # 如果是文件路径，取父目录
    dir_path = path if os.path.isdir(path) or not os.path.splitext(path)[1] \
        else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)