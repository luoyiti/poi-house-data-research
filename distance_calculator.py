import math

def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的曼哈顿距离(米)
    
    参数:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
    
    返回:
        曼哈顿距离(米)
    """
    # 地球半径(米)
    R = 6371000
    
    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    # 计算纬度差对应的距离
    lat_distance = abs(lat2_rad - lat1_rad) * R
    
    # 计算经度差对应的距离(考虑纬度的影响)
    avg_lat = (lat1_rad + lat2_rad) / 2
    lon_distance = abs(lon2_rad - lon1_rad) * R * math.cos(avg_lat)
    
    # 曼哈顿距离 = 纬度距离 + 经度距离
    manhattan_dist = lat_distance + lon_distance
    
    return manhattan_dist

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的直线距离(米) - 使用Haversine公式
    
    参数:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
    
    返回:
        直线距离(米)
    """
    # 地球半径(米)
    R = 6371000
    
    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine公式
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * \
        math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 计算距离
    distance = R * c
    
    return distance