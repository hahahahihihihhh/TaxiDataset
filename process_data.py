import json
import os.path
import time
from collections import defaultdict
from utils.setting import METEOROLOGICAL_VARS, METEOROLOGICAL_VARS_DIVIDE
from utils.utils import fetch_open_meteo_hourly, merge_open_meteo_hourly_responses_in_order, build_poi_filter_csv, \
    make_default_value, discretize_value
import os
import pandas as pd


dataset = "NYCTAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

prefix_path_poi = cfg["paths"]["prefix_path_poi"]
prefix_path_weather = cfg["paths"]["prefix_path_weather"]
osm_file_path = cfg["paths"]["osm_file_path"]
poi_filter_file = cfg["paths"]["poi_filter_file"]
weather_data_file = cfg["paths"]["weather_data_file"]
weather_all_grids_file = cfg["paths"]["weather_all_grids_file"]
lon_min = cfg["grid"]["lon_min"]
lon_max = cfg["grid"]["lon_max"]
lat_min = cfg["grid"]["lat_min"]
lat_max = cfg["grid"]["lat_max"]
H = cfg["grid"]["H"]
W = cfg["grid"]["W"]
time_splits = cfg["time_splits"]
lon_unit = (lon_max - lon_min) / W
lat_unit = (lat_max - lat_min) / H


def coord_to_grid(lon, lat):
    # 不在网格覆盖范围内的直接返回 NaN
    if not (lon_min <= lon < lon_max and lat_min <= lat < lat_max):
        return -1
    col = int((lon - lon_min) // lon_unit)  # 0 ~ 7
    row = int((lat - lat_min) // lat_unit)  # 0 ~ 15
    grid_id = row * W + col
    return grid_id


def grid_to_center_coord(grid_id):
    # 非法 grid_id
    if grid_id < 0 or grid_id >= H * W:
        return None  # 或者返回 (-1, -1)
    row = grid_id // W
    col = grid_id % W
    lon_center = lon_min + (col + 0.5) * lon_unit
    lat_center = lat_min + (row + 0.5) * lat_unit
    return lon_center, lat_center


def gen_poi_kg():
    poi_filter_path = f"{prefix_path_poi}/{poi_filter_file}"
    build_poi_filter_csv(osm_file_path, poi_filter_path)
    poi_filter = pd.read_csv(poi_filter_path)
    poi_filter["grid_id"] = poi_filter.apply(
        lambda r: coord_to_grid(r["lng"], r["lat"]),
        axis=1
    )
    grid_id_poi_distr = defaultdict(make_default_value)
    for _, row in poi_filter.iterrows():
        grid_id_poi_distr[coord_to_grid(row["lng"], row["lat"])][row["cate"]] += 1
    poi_kg = []
    for grid_id, poi_dict in grid_id_poi_distr.items():
        for poi_type, num in poi_dict.items():
            if grid_id != -1 and num > 0:  # 只保留有数据的
                poi_kg.append((grid_id, num, poi_type))
    poi_kg_df = pd.DataFrame(poi_kg, columns=["grid_id", "num", "poi_type"])
    poi_kg_df.to_csv(os.path.join(prefix_path_poi, "poi_kg.csv"), index=False, encoding="utf-8")


def flatten_to_rows(weather_data, grid_id, lon_center, lat_center, met_vars, all_weather_data):
    """
    将 Open-Meteo hourly 响应展开为行记录（按小时一行）
    输出列：grid_id, latitude, longitude, time, <met_vars...>
    """
    lon = lon_center
    lat = lat_center
    hourly = weather_data.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    n = len(times)
    # 取出各变量数组，长度不足则补 None，缺变量也补 None
    series = {}
    for v in met_vars:
        arr = hourly.get(v, None)
        if arr is None:
            series[v] = [None] * n
        else:
            # 防御：万一长度和 time 不一致
            if len(arr) < n:
                series[v] = list(arr) + [None] * (n - len(arr))
            else:
                series[v] = list(arr[:n])
    for i in range(n):
        all_weather_data["grid_id"].append(grid_id)
        all_weather_data["latitude"].append(lat)
        all_weather_data["longitude"].append(lon)
        all_weather_data["time"].append(times[i])
        for v in met_vars:
            all_weather_data[v].append(series[v][i])


def load_weather():
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    os.makedirs(os.path.dirname(weather_all_grids_file_path), exist_ok=True)
    weather_fields = ["grid_id", "latitude", "longitude", "time"] + list(METEOROLOGICAL_VARS)
    all_weather_data = {field: [] for field in weather_fields}
    for grid_id in range(0, H * W):
        lon_center, lat_center = grid_to_center_coord(grid_id)
        # weatherbit
        # resp = requests.get(
        #     "https://api.weatherbit.io/v2.0/history/hourly",
        #     params = {
        #         "lat": lat_center, "lon": lon_center,
        #         "start_date": start_date,
        #         "end_date": end_date,
        #         "tz": "local",  # 本地时间
        #         "key": "837b6655d5324622947fbf962073e22f"
        #     }
        # )
        weather_data_file_path = f"{prefix_path_weather}/{weather_data_file.format(grid_id)}"
        if not os.path.exists(weather_data_file_path):
            all_resp = []
            for time_split in time_splits:
                resp = fetch_open_meteo_hourly(lon_center, lat_center, time_split[0], time_split[1], METEOROLOGICAL_VARS)
                all_resp.append(resp)
            weather_data = merge_open_meteo_hourly_responses_in_order(all_resp)
            with open(weather_data_file_path, "w", encoding="utf-8") as file:
                json.dump(weather_data, file, ensure_ascii=False, indent=2)
            time.sleep(10)
        with open(weather_data_file_path, "r", encoding="utf-8") as f:
            weather_data = json.load(f)
        # 提取latitude，longitude，time，METEOROLOGICAL_VARS中的变量，以及grid_id，保存为csv文件（按照grid_id展开）
        flatten_to_rows(weather_data, grid_id, lon_center, lat_center, METEOROLOGICAL_VARS, all_weather_data)
        print("calculating....{}/{}".format(grid_id, H * W))
    pd.DataFrame(all_weather_data).to_csv(weather_all_grids_file_path, index=False, encoding="utf-8")


def gen_weather_kg():
    load_weather()
    exit(0)
    weather_kg = []
    for grid_id in range(0, H * W):
        weather_path = os.path.join(
            prefix_path_weather,
            weather_data_file.format(grid_id)
        )
        with open(weather_path, "r", encoding="utf-8") as file:
            weather_data = json.load(file)
        hourly = weather_data.get("hourly", {})
        times = hourly.get("time", [])
        # 遍历每一个时间点
        for idx, t in enumerate(times):
            for var, divide_rule in METEOROLOGICAL_VARS_DIVIDE.items():
                if var not in hourly:
                    continue
                value = hourly[var][idx]
                # 缺失值跳过
                if value is None:
                    continue
                label = discretize_value(value, divide_rule)
                if label is None:
                    continue
                # 四元组：(head, relation, tail, time)
                weather_kg.append(
                    (grid_id, var, label, t)
                )
        print(f"processed weather grid... {grid_id}/{H * W}")
    # 保存为 CSV
    weather_kg_df = pd.DataFrame(
        weather_kg,
        columns=["grid_id", "relation", "value", "time"]
    )
    weather_kg_df.to_csv(
        os.path.join(prefix_path_weather, "weather_kg.csv"),
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    # gen_poi_kg()
    gen_weather_kg()
    # load_weather()