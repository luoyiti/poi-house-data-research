import json
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import webbrowser

# 读取数据
with open("data/house_data.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 根据实际字段名提取经纬度（请根据实际情况修改字段名）
# 常见字段名: 'lon', 'lat', 'longitude', 'latitude', '经度', '纬度'
# 如果数据中有 Coordinates 字段格式为 "lat,lon"，需要分割
if 'Coordinates' in df.columns:
    coords = df['Coordinates'].str.split(',', expand=True)
    df['lat'] = pd.to_numeric(coords[0])
    df['lon'] = pd.to_numeric(coords[1])
elif 'lon' in df.columns and 'lat' in df.columns:
    df['lat'] = pd.to_numeric(df['lat'])
    df['lon'] = pd.to_numeric(df['lon'])
else:
    # 请根据实际字段名修改
    print("请检查数据中的经纬度字段名")
    print("现有字段:", df.columns.tolist())

# 过滤掉无效坐标
df = df.dropna(subset=['lat', 'lon'])
df = df[(df['lat'] > 0) & (df['lon'] > 0)]

print(f"有效数据点数量: {len(df)}")

# 北京市中心坐标
beijing_center = [39.9042, 116.4074]

# 方法1: 散点图

# 创建地图
m = folium.Map(
    location=beijing_center,
    zoom_start=11,
    tiles='OpenStreetMap'
)

# 添加聚类标记
marker_cluster = MarkerCluster().add_to(m)

for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=3,
        popup=f"房屋 {idx}",
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.6
    ).add_to(marker_cluster)
output_file = 'image/scatter_graph_beijing_houses_map.html'
m.save(output_file)

# 方法2: 热力图

# 创建地图
m = folium.Map(
    location=beijing_center,
    zoom_start=11,
    tiles='OpenStreetMap'
)

heat_data = [[row['lat'], row['lon']] for idx, row in df.iterrows()]
HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

# 保存地图
output_file = 'image/heat_map_beijing_houses_map.html'
m.save(output_file)

webbrowser.open(output_file)