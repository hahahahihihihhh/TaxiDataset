import json
import os.path
import time
from collections import defaultdict
from utils.setting import METEOROLOGICAL_VARS, METEOROLOGICAL_VARS_DIVIDE
from utils.utils import fetch_open_meteo_hourly, merge_open_meteo_hourly_responses_in_order, build_poi_filter_csv, \
    make_default_value, discretize_value
import os
import pandas as pd


with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings["NYCBike"]

prefix_path_osm = cfg["paths"]["prefix_path_osm"]
prefix_path_poi = cfg["paths"]["prefix_path_poi"]
prefix_path_weather = cfg["paths"]["prefix_path_weather"]
osm_file = cfg["paths"]["osm_file"]
poi_filter_file = cfg["paths"]["poi_filter_file"]
weather_data_file = cfg["paths"]["weather_data_file"]
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
    osm_path = f"{prefix_path_osm}/{osm_file}"
    poi_filter_path = f"{prefix_path_poi}/{poi_filter_file}"
    build_poi_filter_csv(osm_path, poi_filter_path)
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


def load_weather():
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
        all_resp = []
        for time_split in time_splits:
            resp = fetch_open_meteo_hourly(lon_center, lat_center, time_split[0], time_split[1], METEOROLOGICAL_VARS)
            all_resp.append(resp)
        weather_data = merge_open_meteo_hourly_responses_in_order(all_resp)
        with open(f"{prefix_path_weather}/{weather_data_file.format(grid_id)}", "w", encoding="utf-8") as file:
            json.dump(weather_data, file, ensure_ascii=False, indent=2)
        print("calculating....{}/{}".format(grid_id, H * W))
        time.sleep(10)


def gen_weather_kg():
    # load_weather()
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


