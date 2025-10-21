import pandas as pd

def json_data_to_vector(df):
    
    df = pd.DataFrame(df)

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
    
    return df