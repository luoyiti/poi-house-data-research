"""
构建房源特征并保存
"""

import pandas as pd
import numpy as np
import json
from geopy.distance import distance as geo_distance
import os

# ========================= 配置 =========================
DATA_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\merged_house_data.json"
OUTPUT_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\house_predicting_features.csv"

# POI 权重设置
poi_type_weights = {
    'amenity': 0.9,
    'shop': 0.9,
    'tourism': 0.6,
    'leisure': 0.7,
    'healthcare': 1.0,
    'public_transport': 1.0,
    'railway': 0.9,
    'aeroway': -0.5,
    'sport': 0.7,
    'education': 1.0,
    'office': 0.6
}

def calc_total_poi_score(house, max_distance_km=1.0, alpha=0.5):
    """计算房源POI总显著性分数"""
    total_score = 0
    house_lat, house_lon = house.get('lat'), house.get('lon')
    if house_lat is None or house_lon is None:
        return 0
    for poi in house.get('pois', []):
        category = poi.get('first_tag')
        if category not in poi_type_weights:
            continue
        poi_lat, poi_lon = poi.get('lat'), poi.get('lon')
        if poi_lat is None or poi_lon is None:
            continue
        dist = geo_distance((house_lat, house_lon), (poi_lat, poi_lon)).km
        if dist > max_distance_km:
            continue
        total_score += poi_type_weights[category] * np.exp(-alpha * dist)
    return total_score

# ========================= 特征列表 =========================
feature_list = [
    'id', 'Size_num', 'ConstructionYear',
    'Bedroom', 'Washroom', 'Livingroom', 'Kitchen',
    'BuildingStructure', 'Elevator',
    'Floor_level',  'StairsUnitRatio_num',
    'Only_South', 'South_North',
    'TradeYear', 'TradeMonth',
    'poi_total_score',
    'price_per_meter'  # 可以作为目标
]

# ========================= 读取数据 =========================
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    houses = json.load(f)

# ========================= 构建特征 =========================
data = []
for house in houses:
    house['poi_total_score'] = calc_total_poi_score(house)
    row = {f: house.get(f, np.nan) for f in feature_list}
    data.append(row)

df_features = pd.DataFrame(data)
df_features.dropna(inplace=True)  # 去掉缺失值

# ========================= 保存特征文件 =========================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_features.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f" 特征文件已保存: {OUTPUT_FILE}")
print(f"样本数: {len(df_features):,}, 特征数: {len(df_features.columns)}")

