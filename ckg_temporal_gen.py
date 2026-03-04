import json
import math
import os
from datetime import datetime, timedelta
import pandas as pd
from utils.utils import dict_to_pickle


dataset = "NYCTAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]
prefix_path_weather = cfg['paths']['prefix_path_weather']
prefix_path_adj = cfg['paths']['prefix_path_adj']
prefix_path_poi = cfg['paths']['prefix_path_poi']
prefix_path_ckg = cfg['paths']['prefix_path_ckg']
weather_all_grids_file = cfg['paths']['weather_all_grids_file']
start_str, end_str = cfg["start_time"], cfg["end_time"]
H = cfg["grid"]["H"]
W = cfg["grid"]["W"]


def _day_of_week(dt: datetime) -> int:
    """Monday=1 ... Sunday=7."""
    return dt.weekday() + 1


def _rest_work(dt: datetime) -> str:
    """'w' for weekday, 'r' for weekend."""
    return 'w' if dt.weekday() < 5 else 'r'


def _time_of_day(dt: datetime) -> int:
    """10-min slots with ':05' offset: 00:05->1 ... 23:55->144."""
    minutes = dt.hour * 60 + dt.minute
    return int((minutes - 5) // 10) + 1


def generate_ckg_temp():
    # init
    start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
    cur_time = start_time
    ckg_temp = {}
    while cur_time <= end_time:
        ckg_temp[cur_time] = {f"{_grid_id}": {} for _grid_id in range(H * W)}
        cur_time += timedelta(hours=1)
    weather_all_grids_file_path = os.path.join(prefix_path_weather, weather_all_grids_file)
    weather_df = pd.read_csv(weather_all_grids_file_path)
    for id, _ in enumerate(weather_df.values):
        time = datetime.fromisoformat(weather_df['time'][id])
        grid_id = weather_df['grid_id'][id]
        temperature = float(weather_df['temperature_2m'][id])
        windspeed = float(weather_df['wind_speed_10m'][id])
        precipitation = float(weather_df['precipitation'][id])
        # humidity = float(weather_df['relative_humidity_2m'][id])
        temp_item = {
            'time_of_day': _time_of_day(cur_time),
            'hour_of_day': time.hour,
            'day_of_week': _day_of_week(cur_time),
            'rest_work': _rest_work(cur_time),
            'temperature': temperature,
            'rainfall': precipitation,
            # 'humidity': humidity,
            'windspeed': windspeed,
        }
        ckg_temp[time][f"{grid_id}"] = temp_item
    print(ckg_temp)
    dict_to_pickle(ckg_temp, f"{prefix_path_ckg}/kg_temporal_pickle_dict.pickle")


def area_spat_gen():
    adj_kg_path = f"{prefix_path_adj}/adj_kg.csv"
    adj_kg = pd.read_csv(adj_kg_path)
    area_spat = {f"area_{_}": [] for _ in range(H * W)}
    for id, _ in enumerate(adj_kg.values):
        origin_area, dest_area = f"area_{adj_kg['origin'][id]}", f"area_{adj_kg['destination'][id]}"
        touch_num = len(area_spat[f"area_{adj_kg['origin'][id]}"])
        area_spat[origin_area].append([origin_area, f"TouchedByArea[{touch_num + 1}]", dest_area])
    # print(area_spat)
    return area_spat


def area_attr_gen():
    area_attr = {f"area_{_}": [] for _ in range(H * W)}
    for id in range(H * W):
        cur_area = f"area_{id}"
        area_attr[cur_area].append([cur_area, 'hasTouchedFreeSpeedByArea[0]', '0'])
    # return area_attr
    adj_kg_path = f"{prefix_path_adj}/adj_kg.csv"
    adj_kg = pd.read_csv(adj_kg_path)
    area_spat = {f"area_{_}": [] for _ in range(H * W)}
    for id, _ in enumerate(adj_kg.values):
        origin_area, dest_area = f"area_{adj_kg['origin'][id]}", f"area_{adj_kg['destination'][id]}"
        touch_num = len(area_spat[f"area_{adj_kg['origin'][id]}"])
        area_spat[origin_area].append([origin_area, f"TouchedByArea[{touch_num + 1}]", dest_area])
        area_attr[origin_area].append([origin_area, f"hasTouchedFreeSpeedByArea[{touch_num + 1}]", '0'])
    # print(area_attr)
    return area_attr


def poi_spat_gen():
    poi_kg_path = f"{prefix_path_poi}/poi_kg.csv"
    poi_kg = pd.read_csv(poi_kg_path)
    poi_attr = {f"area_{_}": [] for _ in range(H * W)}
    for id, _ in enumerate(poi_kg.values):
        cur_area = f"area_{poi_kg['grid_id'][id]}"
        poi_type = poi_kg['poi_type'][id]
        poi_attr[cur_area].append([poi_type, "locateInBuffer[0]", cur_area])
    print(poi_attr)
    return poi_attr


def poi_attr_gen():
    poi_kg_path = f"{prefix_path_poi}/poi_kg.csv"
    poi_kg = pd.read_csv(poi_kg_path)
    poi_spat = {f"area_{_}": [] for _ in range(H * W)}
    for id, _ in enumerate(poi_kg.values):
        cur_area = f"area_{poi_kg['grid_id'][id]}"
        poi_type = poi_kg['poi_type'][id]
        poi_num = int(poi_kg['num'][id])
        poi_spat[cur_area].append([cur_area, f"hasPoi[{poi_type}]InBuffer[0]", f"PoiNum[{poi_num}]"])
    return poi_spat


def link_bool_gen():
    adj_kg_path = f"{prefix_path_adj}/adj_kg.csv"
    adj_kg = pd.read_csv(adj_kg_path)
    N = H * W
    MAX_DEGREE = 6
    dist = [[math.inf for _ in range(N)] for _ in range(N)]
    for _ in range(N):
        dist[_][_] = 0
    for id, _ in enumerate(adj_kg.values):
        origin_id = int(adj_kg['origin'][id])
        dest_id = int(adj_kg['destination'][id])
        dist[origin_id][dest_id] = 1
    for i in range(N):
        for j in range(N):
            for k in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    link_bool = {f'degree[{_ + 1}]': [] for _ in range(MAX_DEGREE)}
    for _degree in range(MAX_DEGREE):
        for i in range(N):
            for j in range(N):
                if dist[i][j] == _degree + 1:
                    link_bool[f'degree[{_degree + 1}]'].append([f"area_{i}", f"spatiallyLinkDegree[{_degree + 1}]", f"area_{j}"])
    return link_bool


def link_num_gen():
    adj_kg_path = f"{prefix_path_adj}/adj_kg.csv"
    adj_kg = pd.read_csv(adj_kg_path)
    N = H * W
    MAX_DEGREE = 6
    edge = [[] for _ in range(N)]
    for id, _ in enumerate(adj_kg.values):
        origin_id = int(adj_kg['origin'][id])
        dest_id = int(adj_kg['destination'][id])
        edge[origin_id].append(dest_id)
    link_count_mat = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(MAX_DEGREE)]
    def dfs(u, step, vis):
        vis[u] = True
        for v in edge[u]:
            if not vis[v] and step + 1 <= MAX_DEGREE:
                link_count_mat[step][u][v] += 1
                dfs(v, step + 1, vis)
    for i in range(N):
        vis = [0 for _ in range(N)]
        dfs(i, 0, vis)
    link_num = {f'degree[{_ + 1}]': [] for _ in range(MAX_DEGREE)}
    for _degree in range(MAX_DEGREE):
        for i in range(N):
            for j in range(N):
                if link_count_mat[_degree][i][j]:
                    link_num[f'degree[{_degree + 1}]'].append([f"area_{i}", f"spatiallyLinkDegree[{_degree + 1}]", f"area_{j}", f"num:{link_count_mat[_degree][i][j]}"])
    return link_num


def generate_ckg_spat():
    ckg_spat = {
        'area_spat': area_spat_gen(),
        'area_attr': area_attr_gen(),
        'poi_spat': poi_spat_gen(),
        'poi_attr': poi_attr_gen(),
        'link_bool': link_bool_gen(),
        'link_num': link_num_gen()
    }
    # print(ckg_spat)
    dict_to_pickle(ckg_spat, f"{prefix_path_ckg}/kg_spatial_pickle_dict.pickle")


def main():
    # generate_ckg_temp()
    generate_ckg_spat()


if __name__ == '__main__':
    main()