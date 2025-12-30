import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from pyrosm import OSM


# ===== 读取配置 =====
dataset = "NYCTAXI"
with open("setting.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]

osm_basemap_cache = cfg["paths"]["osm_basemap_cache"]
osm_grid_file = cfg["paths"]["osm_grid_file"]
assist_data_path = cfg["paths"]["assist_data_path"]
osm_file = cfg["paths"]["osm_file_path"]
lon_min = cfg["grid"]["lon_min"]
lon_max = cfg["grid"]["lon_max"]
lat_min = cfg["grid"]["lat_min"]
lat_max = cfg["grid"]["lat_max"]
H = cfg["grid"]["H"]
W = cfg["grid"]["W"]


def extract_osm_basemap_bbox(osm_pbf_path, bbox, cache_path=None):
    """
    从 OSM PBF 中抽取 bbox 内的道路网络作为底图。
    bbox = (west, south, east, north)
    cache_path: 若提供，会缓存为 GeoPackage，后续直接读取加速
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[cache] loading basemap from: {cache_path}")
        return gpd.read_file(cache_path)

    print(f"[osm] loading pbf with bbox={bbox} ...")
    osm = OSM(osm_pbf_path, bounding_box=bbox)

    print("[osm] extracting driving network ...")
    roads = osm.get_network(network_type="driving")   # 速度/信息量平衡最好
    if roads is None or len(roads) == 0:
        raise RuntimeError("No roads extracted. Check bbox or PBF coverage.")

    gdf_roads = gpd.GeoDataFrame(roads, geometry="geometry", crs="EPSG:4326")

    # 可选：让底图更“干净”，只留 geometry（减少存储体积）
    gdf_roads = gdf_roads[["geometry"]].copy()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        # GeoPackage 单文件，最适合缓存
        gdf_roads.to_file(cache_path, driver="GPKG")
        print(f"[cache] saved basemap to: {cache_path}")

    return gdf_roads


def plot_osm_basemap_with_grid(
    osm_pbf_path,
    basemap_cache,
    save_path,
    buffer_deg=0.002,
    figsize=(9, 9),
    dpi=300
):
    # --- 1) 计算 bbox（加 buffer，让底图四周留点空间更好看） ---
    west  = lon_min - buffer_deg
    south = lat_min - buffer_deg
    east  = lon_max + buffer_deg
    north = lat_max + buffer_deg
    bbox = [west, south, east, north]

    # --- 2) 抽取底图（bbox 内道路）+ 缓存 ---
    gdf_roads = extract_osm_basemap_bbox(
        osm_pbf_path=osm_pbf_path,
        bbox=bbox,
        cache_path=basemap_cache
    )

    # --- 3) 准备网格线 ---
    lon_unit = (lon_max - lon_min) / W
    lat_unit = (lat_max - lat_min) / H

    fig, ax = plt.subplots(figsize=figsize)

    # --- 4) 画 OSM 底图（道路） ---
    # 线很多时 linewidth 设小一点会更清爽
    gdf_roads.plot(ax=ax, linewidth=0.3, alpha=0.6)

    # --- 5) 画网格外边界（红色） ---
    ax.plot([lon_min, lon_max], [lat_min, lat_min], color="red", linewidth=2)
    ax.plot([lon_min, lon_max], [lat_max, lat_max], color="red", linewidth=2)
    ax.plot([lon_min, lon_min], [lat_min, lat_max], color="red", linewidth=2)
    ax.plot([lon_max, lon_max], [lat_min, lat_max], color="red", linewidth=2)

    # --- 6) 画内部网格线（红色细线） ---
    for i in range(1, W):
        x = lon_min + i * lon_unit
        ax.plot([x, x], [lat_min, lat_max], color="red", linewidth=0.5, alpha=0.7)

    for j in range(1, H):
        y = lat_min + j * lat_unit
        ax.plot([lon_min, lon_max], [y, y], color="red", linewidth=0.5, alpha=0.7)

    # --- 7) 视图范围：用 bbox（带 buffer） ---
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Grid ({H}×{W}) overlay on OSM basemap")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_osm_basemap_with_grid(
        osm_pbf_path=osm_file,
        basemap_cache=osm_basemap_cache,
        save_path=f"{assist_data_path}/{osm_grid_file}",
        buffer_deg=0,
        figsize=(9, 9),
        dpi=300
    )
