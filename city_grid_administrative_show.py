# import json
# import os
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from pyrosm import OSM
#
#
# # ===== 读取配置 =====
# dataset = "TDRIVE"
# with open("setting.json", "r", encoding="utf-8") as f:
#     settings = json.load(f)
# cfg = settings[dataset]
#
# osm_basemap_cache = cfg["paths"]["osm_basemap_cache"]
# osm_grid_file = cfg["paths"]["osm_grid_file"]
# assist_data_path = cfg["paths"]["assist_data_path"]
# osm_file = cfg["paths"]["osm_file_path"]
# lon_min = cfg["grid"]["lon_min"]
# lon_max = cfg["grid"]["lon_max"]
# lat_min = cfg["grid"]["lat_min"]
# lat_max = cfg["grid"]["lat_max"]
# H = cfg["grid"]["H"]
# W = cfg["grid"]["W"]
#
#
# def extract_osm_admin_boundaries_bbox(
#     osm_pbf_path,
#     bbox,
#     cache_path=None,
#     boundary_type="administrative",
#     name=None,
#     admin_levels=None,
#     extra_attributes=("name", "admin_level")
# ):
#     """
#     从 OSM PBF 中抽取 bbox 内的行政区划边界作为底图。
#     bbox = (west, south, east, north)
#     - boundary_type: 默认 "administrative"
#     - name: 可选，按名称过滤（例如 "New York", "Manhattan" 等，取决于数据里实际 name）
#     - admin_levels: 可选，例如 ["2","4","6","8"]，过滤行政级别
#     - cache_path: 若提供，缓存为 GeoPackage
#     """
#     # if cache_path and os.path.exists(cache_path):
#     #     print(f"[cache] loading boundaries from: {cache_path}")
#     #     return gpd.read_file(cache_path)
#
#     print(f"[osm] loading pbf with bbox={bbox} ...")
#     osm = OSM(osm_pbf_path, bounding_box=bbox)
#
#     # custom_filter 用于进一步过滤（比如 admin_level）
#     custom_filter = None
#     if admin_levels is not None:
#         custom_filter = {"admin_level": [str(x) for x in admin_levels]}
#
#     print("[osm] extracting administrative boundaries ...")
#     boundaries = osm.get_boundaries(
#         boundary_type=boundary_type,
#         name=name,
#         custom_filter=custom_filter,
#         extra_attributes=list(extra_attributes) if extra_attributes else None
#     )
#
#     if boundaries is None or len(boundaries) == 0:
#         raise RuntimeError("No administrative boundaries extracted. Try widening bbox or adjusting name/admin_levels.")
#
#     gdf = gpd.GeoDataFrame(boundaries, geometry="geometry", crs="EPSG:4326")
#
#     # 只保留必要字段，减少体积（你也可以保留 name/admin_level 方便调试）
#     keep_cols = ["geometry"]
#     for c in ["name", "admin_level"]:
#         if c in gdf.columns:
#             keep_cols.append(c)
#     gdf = gdf[keep_cols].copy()
#
#     if cache_path:
#         os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
#         gdf.to_file(cache_path, driver="GPKG")
#         print(f"[cache] saved boundaries to: {cache_path}")
#
#     return gdf
#
#
# def plot_osm_basemap_with_grid(
#     osm_pbf_path,
#     basemap_cache,
#     save_path,
#     buffer_deg=0.002,
#     figsize=(9, 9),
#     dpi=300,
#     boundary_name=None,
#     admin_levels=None
# ):
#     # --- 1) 计算 bbox（加 buffer） ---
#     west  = lon_min - buffer_deg
#     south = lat_min - buffer_deg
#     east  = lon_max + buffer_deg
#     north = lat_max + buffer_deg
#     bbox = [west, south, east, north]
#
#     # --- 2) 抽取行政区划边界 + 缓存 ---
#     gdf_boundaries = extract_osm_admin_boundaries_bbox(
#         osm_pbf_path=osm_pbf_path,
#         bbox=bbox,
#         cache_path=basemap_cache,
#         boundary_type="administrative",
#         name=boundary_name,        # 例如 "New York", "Manhattan"（看 OSM 数据里有什么）
#         admin_levels=admin_levels, # 例如 ["4","6","8"]
#         extra_attributes=("name", "admin_level")
#     )
#
#     # --- 3) 准备网格线 ---
#     lon_unit = (lon_max - lon_min) / W
#     lat_unit = (lat_max - lat_min) / H
#
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # --- 4) 画行政区划边界底图 ---
#     # 边界建议用线框，不要填充
#     gdf_boundaries.boundary.plot(ax=ax, linewidth=1.0, alpha=0.8)
#
#     # --- 5) 画网格外边界（红色） ---
#     ax.plot([lon_min, lon_max], [lat_min, lat_min], color="red", linewidth=2)
#     ax.plot([lon_min, lon_max], [lat_max, lat_max], color="red", linewidth=2)
#     ax.plot([lon_min, lon_min], [lat_min, lat_max], color="red", linewidth=2)
#     ax.plot([lon_max, lon_max], [lat_min, lat_max], color="red", linewidth=2)
#
#     # --- 6) 画内部网格线（红色细线） ---
#     for i in range(1, W):
#         x = lon_min + i * lon_unit
#         ax.plot([x, x], [lat_min, lat_max], color="red", linewidth=0.5, alpha=0.7)
#
#     for j in range(1, H):
#         y = lat_min + j * lat_unit
#         ax.plot([lon_min, lon_max], [y, y], color="red", linewidth=0.5, alpha=0.7)
#
#     # --- 7) 视图范围：用 bbox（带 buffer） ---
#     ax.set_xlim(west, east)
#     ax.set_ylim(south, north)
#     ax.set_aspect("equal", adjustable="box")
#
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     ax.set_title(f"Grid ({H}×{W}) overlay on Administrative Boundaries")
#
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
#         print(f"[save] {save_path}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     plot_osm_basemap_with_grid(
#         osm_pbf_path=osm_file,
#         basemap_cache=osm_basemap_cache,
#         save_path=f"{assist_data_path}/{osm_grid_file}",
#         buffer_deg=0,
#         figsize=(9, 9),
#         dpi=300,
#         boundary_name=None,      # 例如 "New York" / "Manhattan"（可选）
#         admin_levels=["6"]  # 可按需调整；None 表示不过滤
#     )


import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
from pyrosm import OSM

# ===== 读取配置 =====
dataset = "TDRIVE"
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


def extract_osm_admin_boundaries_bbox(
        osm_pbf_path,
        bbox,
        cache_path=None,
        boundary_type="administrative",
        name=None,
        admin_levels=None,
        extra_attributes=("name", "admin_level")
):
    """
    从 OSM PBF 中抽取 bbox 内的北京区级行政边界（适配参考图风格）
    bbox = (west, south, east, north)
    """
    # 开启缓存，避免重复解析OSM文件
    if cache_path and os.path.exists(cache_path):
        print(f"[cache] loading boundaries from: {cache_path}")
        return gpd.read_file(cache_path)

    print(f"[osm] loading pbf with bbox={bbox} ...")
    osm = OSM(osm_pbf_path, bounding_box=bbox)

    # 过滤行政级别
    custom_filter = None
    if admin_levels is not None:
        custom_filter = {"admin_level": [str(x) for x in admin_levels]}

    print("[osm] extracting administrative boundaries ...")
    boundaries = osm.get_boundaries(
        boundary_type=boundary_type,
        name=name,
        custom_filter=custom_filter,
        extra_attributes=list(extra_attributes) if extra_attributes else None
    )

    if boundaries is None or len(boundaries) == 0:
        raise RuntimeError("No administrative boundaries extracted. Try widening bbox or adjusting name/admin_levels.")

    gdf = gpd.GeoDataFrame(boundaries, geometry="geometry", crs="EPSG:4326")

    # 保留必要字段
    keep_cols = ["geometry"]
    for c in ["name", "admin_level"]:
        if c in gdf.columns:
            keep_cols.append(c)
    gdf = gdf[keep_cols].copy()

    # ========== 核心过滤：只保留北京区级边界 ==========
    # 1. 只保留6级（区级）
    if "admin_level" in gdf.columns:
        gdf = gdf[gdf["admin_level"] == "6"].copy()
    # 2. 只保留北京的区（名称含“区”）
    if "name" in gdf.columns:
        gdf = gdf[gdf["name"].str.contains("区", na=False)].copy()
    # 3. 只保留面状几何（过滤掉线、点等多余元素）
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if len(gdf) == 0:
        raise RuntimeError("No Beijing district-level boundaries found! Check OSM data or bbox.")

    # 缓存结果
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        gdf.to_file(cache_path, driver="GPKG")
        print(f"[cache] saved Beijing district boundaries to: {cache_path}")

    return gdf


def plot_osm_basemap_with_grid(
        osm_pbf_path,
        basemap_cache,
        save_path,
        buffer_deg=0.002,
        figsize=(9, 9),
        dpi=300,
        boundary_name=None,
        admin_levels=None
):
    # --- 1) 计算带缓冲的bbox ---
    west = lon_min - buffer_deg
    south = lat_min - buffer_deg
    east = lon_max + buffer_deg
    north = lat_max + buffer_deg
    bbox = [west, south, east, north]

    # --- 2) 抽取北京区级边界 ---
    gdf_boundaries = extract_osm_admin_boundaries_bbox(
        osm_pbf_path=osm_pbf_path,
        bbox=bbox,
        cache_path=basemap_cache,
        boundary_type="administrative",
        name=boundary_name,
        admin_levels=admin_levels,
        extra_attributes=("name", "admin_level")
    )

    # --- 3) 计算网格单位 ---
    lon_unit = (lon_max - lon_min) / W
    lat_unit = (lat_max - lat_min) / H

    # --- 4) 绘图（核心：参考图风格配置） ---
    # 创建画布，设置白色背景
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # 4.1 绘制行政区填充（灰色底色，参考图风格）
    gdf_boundaries.plot(
        ax=ax,
        facecolor="#E0E0E0",  # 浅灰色填充
        edgecolor="#909090",  # 深一点的灰色轮廓
        linewidth=0.8,  # 轮廓线宽
        alpha=1.0,
        zorder=1  # 层级低于网格
    )

    # 4.2 绘制网格外边界（白色粗线）
    ax.plot([lon_min, lon_max], [lat_min, lat_min], color="white", linewidth=1.2, zorder=2)
    ax.plot([lon_min, lon_max], [lat_max, lat_max], color="white", linewidth=1.2, zorder=2)
    ax.plot([lon_min, lon_min], [lat_min, lat_max], color="white", linewidth=1.2, zorder=2)
    ax.plot([lon_max, lon_max], [lat_min, lat_max], color="white", linewidth=1.2, zorder=2)

    # 4.3 绘制内部网格线（白色细线）
    for i in range(1, W):
        x = lon_min + i * lon_unit
        ax.plot([x, x], [lat_min, lat_max], color="white", linewidth=0.6, alpha=0.8, zorder=2)
    for j in range(1, H):
        y = lat_min + j * lat_unit
        ax.plot([lon_min, lon_max], [y, y], color="white", linewidth=0.6, alpha=0.8, zorder=2)

    # --- 5) 样式优化（参考图简洁风格） ---
    # 限定视图范围（无多余空白）
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # 等比例显示
    ax.set_aspect("equal", adjustable="box")
    # 隐藏坐标轴、刻度、边框（核心：简洁风格）
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 移除标题（参考图无标题，如需保留可取消下面注释）
    # ax.set_title(f"Beijing Grid ({H}×{W})", fontsize=12, pad=10)

    # --- 6) 保存图片 ---
    plt.tight_layout(pad=0)  # 无边距
    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,  # 无额外边距
            facecolor="white",
            edgecolor="none"
        )
        print(f"[save] {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_osm_basemap_with_grid(
        osm_pbf_path=osm_file,
        basemap_cache=osm_basemap_cache,
        save_path=f"{assist_data_path}/{osm_grid_file}",
        buffer_deg=0.005,  # 轻微缓冲，确保边界完整
        figsize=(9, 9),
        dpi=300,
        boundary_name="北京市",  # 限定北京范围
        admin_levels=["6"]  # 北京区级
    )