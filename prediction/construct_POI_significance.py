"""
构建房源特征并保存（先验+数据驱动权重+房源级动态加权版）
因为完全依靠动态加权，结果完全和现实不符，比如education很低，因此使用先验，然后利用自动加权进行调整。
"""
import pandas as pd
import numpy as np
import json
from geopy.distance import distance as geo_distance
import os
from sklearn.ensemble import RandomForestRegressor

# ========================= 配置 =========================
DATA_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\merged_house_data.json"
OUTPUT_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\house_predicting_features.csv"
WEIGHTS_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\poi_weights.txt"

# ========================= 读取数据 =========================
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    houses = json.load(f)

# ========================= POI类别初始化 =========================
poi_categories = [
    'amenity', 'shop', 'tourism', 'leisure', 'healthcare',
    'public_transport', 'railway', 'aeroway', 'sport',
    'education', 'office'
]

# ========================= 构建POI原始分数 =========================
max_distance_km = 1.0
alpha = 0.5

for house in houses:
    house_lat, house_lon = house.get('lat'), house.get('lon')
    if house_lat is None or house_lon is None:
        continue
    house['poi_scores'] = {cat: 0 for cat in poi_categories}
    for poi in house.get('pois', []):
        cat = poi.get('first_tag')
        if cat not in poi_categories:
            continue
        poi_lat, poi_lon = poi.get('lat'), poi.get('lon')
        if poi_lat is None or poi_lon is None:
            continue
        dist = geo_distance((house_lat, house_lon), (poi_lat, poi_lon)).km
        if dist > max_distance_km:
            continue
        house['poi_scores'][cat] += np.exp(-alpha * dist)

# ========================= 构建DataFrame =========================
feature_list = [
    'id', 'Size_num', 'ConstructionYear',
    'Bedroom', 'Washroom', 'Livingroom', 'Kitchen',
    'BuildingStructure', 'Elevator',
    'Floor_level', 'StairsUnitRatio_num',
    'Only_South', 'South_North',
    'TradeYear', 'TradeMonth',
    'price_per_meter'  # 目标
]

data = []
for house in houses:
    row = {f: house.get(f, np.nan) for f in feature_list}
    row.update(house.get('poi_scores', {}))
    data.append(row)

df = pd.DataFrame(data)
df.dropna(inplace=True)

# ========================= 先验权重 =========================
poi_prior_weights = {
    'amenity': 0.8,
    'shop': 0.7,
    'tourism': 0.6,
    'leisure': 0.6,
    'healthcare': 1.0,
    'public_transport': 1.0,
    'railway': 0.8,
    'aeroway': 0.3,
    'sport': 0.5,
    'education': 1.0,
    'office': 0.6
}

# ========================= 数据驱动权重 =========================
poi_corr = df[poi_categories].corrwith(df['price_per_meter']).abs().fillna(0)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(df[poi_categories], df['price_per_meter'])
poi_rf = pd.Series(rf.feature_importances_, index=poi_categories)

# 综合全局权重
alpha_prior = 0.5
alpha_data = 0.5
poi_global_weights = alpha_prior * pd.Series(poi_prior_weights) + alpha_data * (poi_corr + poi_rf)/2

# 归一化
# poi_global_weights /= poi_global_weights.sum()
print("POI全局权重:\n", poi_global_weights)

# ========================= 保存权重到TXT =========================
os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
    f.write("POI全局权重（先验+数据驱动）:\n")
    for cat, w in poi_global_weights.items():
        f.write(f"{cat}: {w:.4f}\n")
print(f"POI权重已保存到: {WEIGHTS_FILE}")

# ========================= 房源级POI总分 =========================
def calc_total_poi_score_dynamic(house, global_weights, max_distance_km=1.0, alpha=0.5):
    """每个房源动态计算POI总分"""
    total_score = 0
    house_lat, house_lon = house.get('lat'), house.get('lon')
    if house_lat is None or house_lon is None:
        return 0
    local_scores = {}
    for poi in house.get('pois', []):
        cat = poi.get('first_tag')
        if cat not in global_weights:
            continue
        poi_lat, poi_lon = poi.get('lat'), poi.get('lon')
        if poi_lat is None or poi_lon is None:
            continue
        dist = geo_distance((house_lat, house_lon), (poi_lat, poi_lon)).km
        if dist > max_distance_km:
            continue
        local_scores[cat] = local_scores.get(cat, 0) + np.exp(-alpha * dist)
    sum_local = sum(local_scores.values())
    if sum_local > 0:
        for cat, score in local_scores.items():
            total_score += (score / sum_local) * global_weights[cat]
    return total_score

df['poi_total_score'] = df.apply(
    lambda row: calc_total_poi_score_dynamic(houses[df.index.get_loc(row.name)], poi_global_weights),
    axis=1
)

# ========================= 保存特征文件 =========================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"特征文件已保存: {OUTPUT_FILE}")
print(f"样本数: {len(df):,}, 特征数: {len(df.columns)}")
