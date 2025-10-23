import json
import pandas as pd
import numpy as np
import sys

# 保证输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 文件路径
json_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\house_poi_data_more_precise.json"
excel_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\housefeatures.xlsx"

with open(json_path, 'r', encoding='utf-8') as f:
    house_dict = json.load(f)

house_df = pd.read_excel(excel_path)

# 生成坐标列
if 'Coordinates' not in house_df.columns:
    house_df['Coordinates'] = house_df['lon'].astype(str) + ',' + house_df['lat'].astype(str)

house_df.columns = [c.strip() for c in house_df.columns]

house_infos = []

for house in house_dict:
    house_info = {
        "id": house.get('id'),
        "price_per_meter": house.get('price_per_meter'),
        "lon": str(house.get('lon')),
        "lat": str(house.get('lat')),
    }

    coordinate = f"{house_info['lon']},{house_info['lat']}"
    house_mes = house_df[house_df['Coordinates'] == coordinate]

    def get_value_safe(df, col):
        return df[col].values[0] if col in df.columns and not df.empty else None

    # 基本行政区
    house_info['county'] = get_value_safe(house_mes, 'County')
    house_info['town'] = get_value_safe(house_mes, 'Town')
    house_info['community_name'] = get_value_safe(house_mes, 'CommunityName')

    # 所有特征
    house_info['Size_num'] = get_value_safe(house_mes, 'Size_num')
    house_info['ConstructionYear'] = get_value_safe(house_mes, 'ConstructionYear')
    house_info['lng'] = get_value_safe(house_mes, 'lng')
    house_info['lat'] = get_value_safe(house_mes, 'lat')
    house_info['Bedroom'] = get_value_safe(house_mes, 'Bedroom')
    house_info['Washroom'] = get_value_safe(house_mes, 'Washroom')
    house_info['Livingroom'] = get_value_safe(house_mes, 'Livingroom')
    house_info['Kitchen'] = get_value_safe(house_mes, 'Kitchen')
    house_info['BuildingStructure'] = get_value_safe(house_mes, 'BuildingStructure')
    house_info['Elevator'] = get_value_safe(house_mes, 'Elevator')
    house_info['Floor_level'] = get_value_safe(house_mes, 'Floor_level')
    house_info['StairsUnitRatio_num'] = get_value_safe(house_mes, 'StairsUnitRatio_num')
    house_info['Only_South'] = get_value_safe(house_mes, 'Only_South')
    house_info['South_North'] = get_value_safe(house_mes, 'South_North')
    house_info['TradeYear'] = get_value_safe(house_mes, 'TradeYear')
    house_info['TradeMonth'] = get_value_safe(house_mes, 'TradeMonth')

    house_info["pois"] = house.get("pois", [])

    house_infos.append(house_info)

# 转换 numpy 类型
def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

house_infos = convert_types(house_infos)

print(f"共整合 {len(house_infos)} 套房源数据")
print("示例一条：")
print(json.dumps(house_infos[0], ensure_ascii=False, indent=2))

output_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\merged_house_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(house_infos, f, ensure_ascii=False, indent=2)

print(f"\n已保存至：{output_path}")
