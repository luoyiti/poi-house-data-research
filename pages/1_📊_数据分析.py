import streamlit as st
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster, FeatureGroupSubGroup
import json
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import sys
sys.path.append('..')  # 添加上级目录到路径
from tools.data_tool import *


# 北京市中心坐标
beijing_center = [39.9042, 116.4074]
# 读取数据
with open("./data/house_data.json", 'r') as f:
    data = json.load(f)

houses = json_data_to_vector(data)

# 读取数据
with open("./data/poi_data.json", 'r') as f:
    data = json.load(f)

pois = json_data_to_vector(data)


del data

# 创建地图
m = folium.Map(
    location=beijing_center,
    zoom_start=11,
    tiles='OpenStreetMap'
)

# 创建主聚类
marker_cluster = MarkerCluster().add_to(m)

# 创建特征组
house_group = folium.FeatureGroup(name='🏠 房屋', show=True).add_to(m)
poi_group = folium.FeatureGroup(name='📍 POI', show=True).add_to(m)

# 添加房屋标记 - 橙红色正方形
for idx, row in houses.iterrows():
    folium.RegularPolygonMarker(
        location=[row['lat'], row['lon']],
        popup=f"<b>🏠 房屋</b><br>小区: {row.get('community_name', idx)}<br>价格: {row.get('price_per_meter', 'N/A')}元/㎡",
        tooltip=f"房屋 #{idx}",
        number_of_sides=4,  # 正方形
        radius=8,
        color='#FF4500',  # 橙红色边框
        fill=True,
        fillColor='#FF6347',  # 番茄红填充
        fillOpacity=0.8,
        weight=2,
        rotation=45  # 旋转45度变成菱形
    ).add_to(house_group)

# 添加POI标记 - 青色三角形
for idx, row in pois.iterrows():
    poi_name = row.get('tags', {}).get('name', f'POI {idx}') if isinstance(row.get('tags'), dict) else f'POI {idx}'
    
    folium.RegularPolygonMarker(
        location=[row['lat'], row['lon']],
        popup=f"<b>📍 POI</b><br>名称: {poi_name}",
        tooltip=poi_name,
        number_of_sides=3,  # 三角形
        radius=8,
        color='#008B8B',  # 深青色边框
        fill=True,
        fillColor='#20B2AA',  # 浅海绿色填充
        fillOpacity=0.8,
        weight=2,
        rotation=0
    ).add_to(poi_group)

# 添加图层控制
folium.LayerControl(collapsed=False).add_to(m)

st.markdown("""
## POI-房屋位置节点分布图
""")

map_data = st_folium(m, width=700, height=500)

st.markdown("""
## 房屋价格分布图
""")






import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import contextily as ctx
from io import StringIO

st.set_page_config(page_title="北京市房价分布地图（Matplotlib + Contextily）", layout="wide")

# ===== 侧边栏：数据源 =====
st.sidebar.header("数据源")
use_upload = st.sidebar.toggle("使用上传文件（否则读取默认相对路径）", value=False)

houses_file = None
pois_file = None

if use_upload:
    houses_file = st.sidebar.file_uploader("上传 house_data.json", type=["json"])
    pois_file = st.sidebar.file_uploader("上传 poi_data.json", type=["json"])
else:
    st.sidebar.info("将从默认相对路径 ./data/house_data.json 与 ./data/poi_data.json 读取")

# ===== 字体与渲染设置 =====
plt.rcParams["axes.unicode_minus"] = False
# 尝试设置中文字体（macOS 上常见），并提供回退
try:
    plt.rcParams["font.family"] = "Heiti TC"
except Exception:
    pass  # 使用默认字体

@st.cache_data
def load_json_from_bytes(b: bytes):
    return json.loads(b.decode("utf-8"))

@st.cache_data
def load_json_from_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== 读数据 =====
with st.status("读取数据中...", expanded=False) as s:
    try:
        if use_upload:
            if houses_file is None or pois_file is None:
                s.update(label="等待上传 JSON 文件...", state="running")
                st.stop()
            houses = load_json_from_bytes(houses_file.read())
            pois = load_json_from_bytes(pois_file.read())
        else:
            houses = load_json_from_path("./data/house_data.json")
            pois = load_json_from_path("./data/poi_data.json")
        s.update(label="数据读取完成 ✅", state="complete")
    except Exception as e:
        s.update(label=f"数据读取失败：{e}", state="error")
        st.error(f"数据读取失败：{e}")
        st.stop()

# ===== 提取房屋数据 =====
lons_h, lats_h, prices = [], [], []
for house in houses:
    try:
        lons_h.append(float(house["lon"]))
        lats_h.append(float(house["lat"]))
        prices.append(float(house["price_per_meter"]))
    except (ValueError, KeyError, TypeError):
        continue

if len(prices) == 0:
    st.warning("房屋数据为空或未解析到有效坐标/价格。")
    st.stop()

lons_h = np.array(lons_h)
lats_h = np.array(lats_h)
prices = np.array(prices)

# ===== 提取 POI 数据 =====
lons_p, lats_p = [], []
for poi in pois:
    try:
        lons_p.append(float(poi["lon"]))
        lats_p.append(float(poi["lat"]))
    except (ValueError, KeyError, TypeError):
        continue
lons_p = np.array(lons_p) if len(lons_p) else np.array([])
lats_p = np.array(lats_p) if len(lats_p) else np.array([])

# ===== 侧边栏交互控件 =====
st.sidebar.header("可视化设置")
poi_alpha = st.sidebar.slider("POI 透明度", 0.0, 1.0, 0.8, 0.05)
poi_size = st.sidebar.slider("POI 点大小", 1, 40, 20)
house_size = st.sidebar.slider("房屋点大小", 10, 200, 80)
cmap_choice = st.sidebar.selectbox("房价颜色映射", ["RdYlGn_r", "viridis", "plasma", "turbo"], index=0)
show_stats = st.sidebar.toggle("显示统计信息框", value=False)
use_container_width = st.sidebar.toggle("铺满容器宽度", value=True)

# ===== 画图 =====
fig, ax = plt.subplots(figsize=(18, 16))

# 先画房屋（保留句柄用于 colorbar）
scatter_price = ax.scatter(
    lons_h, lats_h,
    c=prices, cmap=cmap_choice,
    s=house_size, alpha=0.8,
    edgecolors="white", linewidth=0.8, zorder=5
)

# 再画 POI（如果有）
if lons_p.size > 0:
    ax.scatter(
        lons_p, lats_p,
        c="#87CEEB",
        s=poi_size, alpha=poi_alpha,
        edgecolors="white", linewidth=0.8, zorder=4
    )

# 添加底图（Web Mercator），这里我们传入数据是经纬度（EPSG:4326），让 contextily 负责重投影
map_loaded = False
try:
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=ctx.providers.OpenStreetMap.Mapnik,
        attribution=False, alpha=0.6, zorder=1
    )
    map_loaded = True
except Exception as e:
    st.warning(f"地图底图加载失败：{e}\n将使用备用浅色背景与网格线。")
    ax.set_facecolor("#E8F4F8")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.8, color="#CCCCCC")
    map_loaded = False

# 颜色条绑定到房价散点
cbar = plt.colorbar(scatter_price, ax=ax, label="房价 (元/平方米)", pad=0.02, shrink=0.75)
cbar.ax.tick_params(labelsize=12)

# 标题与坐标轴
ax.set_title("北京市房价分布地图\n(基于真实地图底图)", fontsize=20, fontweight="bold", pad=25)
ax.set_xlabel("经度 (Longitude)", fontsize=14, fontweight="bold")
ax.set_ylabel("纬度 (Latitude)", fontsize=14, fontweight="bold")

# 坐标范围（用所有点的范围，如果没有 POI，就用房屋范围）
all_lons = np.concatenate([lons_h, lons_p]) if lons_p.size else lons_h
all_lats = np.concatenate([lats_h, lats_p]) if lats_p.size else lats_h
ax.set_xlim(all_lons.min() - 0.05, all_lons.max() + 0.05)
ax.set_ylim(all_lats.min() - 0.05, all_lats.max() + 0.05)
ax.tick_params(labelsize=12)

# 统计信息（可选）
if show_stats:
    price_stats = (
        "📊 统计信息\n"
        f"平均价格: ¥{prices.mean():.0f}/㎡\n"
        f"中位数价格: ¥{np.median(prices):.0f}/㎡\n"
        f"最高价格: ¥{prices.max():.0f}/㎡\n"
        f"最低价格: ¥{prices.min():.0f}/㎡\n"
        f"标准差: ¥{prices.std():.0f}/㎡\n"
        f"样本数量: {len(prices):,} 套"
    )
    ax.text(
        0.02, 0.98, price_stats, transform=ax.transAxes,
        fontsize=11, va="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white",
                  alpha=0.95, edgecolor="darkblue", linewidth=2),
        zorder=10,
    )

# 数据来源
source_text = "地图来源: OpenStreetMap © OSM Contributors" if map_loaded else "备用地图模式"
ax.text(
    0.98, 0.02, source_text, transform=ax.transAxes,
    fontsize=9, ha="right", style="italic",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    zorder=10
)

plt.tight_layout()

# 在 Streamlit 中渲染
st.pyplot(fig, use_container_width=use_container_width)


