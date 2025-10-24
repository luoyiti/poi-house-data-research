import pandas as pd
import matplotlib.pyplot as plt

# ===================== 读取预测结果 =====================
file_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\models\predictions_20251023_224221.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

# ===================== 设置中文字体 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示

# ===================== 绘制散点图 =====================
plt.figure(figsize=(10, 6))
plt.scatter(df['y_true'], df['y_pred'], alpha=0.5, color='blue')

# 添加对角线
plt.plot([df['y_true'].min(), df['y_true'].max()],
         [df['y_true'].min(), df['y_true'].max()],
         'r--', lw=2, label='y = x')

# 添加标题和坐标轴
plt.title('真实房价 vs 预测房价散点图')
plt.xlabel('真实房价 (元/平方米)')
plt.ylabel('预测房价 (元/平方米)')
plt.legend()

# 调整布局，防止标题或坐标轴被截断
plt.tight_layout()

# 显示图表
plt.show()
