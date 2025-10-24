import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx  


# ================== 字体配置（中文支持） ==================
plt.rcParams['font.family'] = 'Heiti TC'  # macOS
plt.rcParams['axes.unicode_minus'] = False

# ================== 路径配置 ==================
pred_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\models\predictions_20251023_224221.csv"
json_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\house_data.json"
save_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\beijing_predicted_price_map.png"

# ================== 1. 加载数据 ==================
# 预测结果
pred_df = pd.read_csv(pred_path, encoding='utf-8-sig')

# 自动识别预测列名（如 y_pred 或 predicted_price）
price_col = None
for c in pred_df.columns:
    if 'pred' in c.lower() or 'forecast' in c.lower():
        price_col = c
        break
if price_col is None:
    raise ValueError("❌ 未找到预测价格列，请检查CSV列名（例如 y_pred 或 predicted_price）。")

# 房屋经纬度
with open(json_path, 'r', encoding='utf-8') as f:
    house_data = json.load(f)
if isinstance(house_data, dict) and "houses" in house_data:
    house_data = house_data["houses"]
house_df = pd.DataFrame(house_data)

# 合并
merged = pd.merge(pred_df, house_df, on="id", how="inner")

# 转换预测房价为数值
merged["predicted_price"] = pd.to_numeric(merged[price_col], errors="coerce")

# 去除无效数据
merged = merged.dropna(subset=["predicted_price", "lon", "lat"])

# ================== 2. 提取经纬度和房价 ==================
lons = merged["lon"].astype(float).to_numpy()
lats = merged["lat"].astype(float).to_numpy()
prices = merged["predicted_price"].astype(float).to_numpy()

# ================== 3. 绘图 ==================
fig, ax = plt.subplots(figsize=(18, 16))

# 散点图（房价）
scatter = ax.scatter(
    lons, lats,
    c=prices,
    cmap='RdYlGn_r',  # 红-黄-绿反转，红=高价
    s=80, alpha=0.85,
    edgecolors='white', linewidth=0.8,
    zorder=5
)

# ================== 4. 添加地图底图 ==================
map_loaded = False
try:
    ctx.add_basemap(
        ax,
        crs='EPSG:4326',
        source=ctx.providers.OpenStreetMap.Mapnik,
        attribution=False,
        alpha=0.6,
        zorder=1
    )
    map_loaded = True
    print("✓ 地图底图加载成功")
except Exception as e:
    print(f"⚠ 地图底图加载失败: {e}")
    print("  使用备用浅色背景...")
    ax.set_facecolor('#E8F4F8')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='#CCCCCC')

# ================== 5. 美化图表 ==================
cbar = plt.colorbar(scatter, ax=ax, label='预测房价 (元/平方米)', pad=0.02, shrink=0.75)
cbar.ax.tick_params(labelsize=12)

ax.set_title('北京市预测房价分布图\n(叠加真实地图底图)', fontsize=20, fontweight='bold', pad=25)
ax.set_xlabel('经度 (Longitude)', fontsize=14, fontweight='bold')
ax.set_ylabel('纬度 (Latitude)', fontsize=14, fontweight='bold')

# 经纬度范围（自动加缓冲）
ax.set_xlim(lons.min() - 0.05, lons.max() + 0.05)
ax.set_ylim(lats.min() - 0.05, lats.max() + 0.05)
ax.tick_params(labelsize=12)

# 数据来源说明
source_text = '地图来源: OpenStreetMap © OSM Contributors' if map_loaded else '备用地图模式'
ax.text(0.98, 0.02, source_text, transform=ax.transAxes,
        fontsize=9, horizontalalignment='right', style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        zorder=10)

plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

# ================== 6. 输出信息 ==================
print(f"\n✓ 共绘制 {len(prices):,} 个房源点")
print(f"✓ 预测价格范围: ¥{prices.min():.0f} - ¥{prices.max():.0f} 元/㎡")
print(f"✓ 平均预测价格: ¥{prices.mean():.0f}/㎡")
print(f"✓ 地图已保存至: {save_path}")
