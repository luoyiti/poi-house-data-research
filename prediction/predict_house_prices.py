"""
房价预测多模型训练与评估
使用之前生成的房源特征文件（包含POI显著性等）
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ========================= 配置 =========================
DATA_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\final_house_features.csv"
MODEL_DIR = os.path.join(os.path.dirname(DATA_FILE), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

# ========================= 加载数据 =========================
df = pd.read_csv(DATA_FILE)
print(f" 数据加载成功，样本数: {len(df):,}, 特征数: {len(df.columns)}")

# ========================= 特征与目标 =========================
# ========================= 特征与目标 =========================
target = 'price_per_meter'

# 指定使用的特征列
feature_cols = [
    'Building_Hardware_Score',
    'Size_num',
    'Bedroom',
    'Washroom',
    'Livingroom',
    'Kitchen',
    'poi_total_score'
]

# 定义特征矩阵和目标变量
X = df[feature_cols]
y = df[target]





# ========================= 数据集划分 =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"训练集: {len(X_train):,} 样本, 测试集: {len(X_test):,} 样本")

# ========================= 模型定义 =========================
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'Lasso': Lasso(alpha=0.01, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15,
                                          random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                  learning_rate=0.1, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, n_jobs=-1)
}

# ========================= 模型训练与评估 =========================
def evaluate_regression(y_true, y_pred):
    """计算回归指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


overall_results = {}
best_model = None
best_r2 = -np.inf

for model_name, model in tqdm(models.items(), desc="训练模型"):
    print(f"\n训练 {model_name} ...")
    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        
        metrics = evaluate_regression(y_test, y_pred_test)
        overall_results[model_name] = metrics
        
        print(f"  测试集指标: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")
        
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = (model_name, model)
            
    except Exception as e:
        print(f"模型训练失败: {e}")
        overall_results[model_name] = None

# ========================= 生成评估报告 =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = os.path.join(MODEL_DIR, f"regression_report_{timestamp}.txt")

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("房价预测多模型训练评估报告\n")
    f.write(f"生成时间: {datetime.now()}\n")
    f.write("="*80 + "\n\n")
    
    f.write("【模型性能】\n")
    for model_name, metrics in overall_results.items():
        if metrics:
            f.write(f"{model_name}: {metrics}\n")
    
    if best_model:
        f.write(f"\n【最佳模型】{best_model[0]} - R2={best_r2:.4f}\n")

print(f"\n 报告已生成: {report_file}")

# ========================= 特征重要性 =========================
if best_model and hasattr(best_model[1], 'feature_importances_'):
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model[1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特征:")
    print(fi.head(10).to_string(index=False))
    
    # 可视化
    plt.figure(figsize=(10,6))
    plt.barh(fi.head(15)['feature'], fi.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importance - {best_model[0]}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    fi_plot_file = os.path.join(MODEL_DIR, f"feature_importance_{timestamp}.png")
    plt.savefig(fi_plot_file, dpi=100)
    plt.close()
    print(f"特征重要性图已保存: {fi_plot_file}")

# ========================= 保存最佳模型 =========================
if best_model:
    model_name, model = best_model
    model_file = os.path.join(MODEL_DIR, f"best_model_{model_name}_{timestamp}.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"最佳模型已保存: {model_file}")

# ========================= 保存预测结果 =========================
y_pred_best = best_model[1].predict(X_test)
pred_df = pd.DataFrame({
    'id': df.iloc[X_test.shape[0]*-1:]['id'].values,  # 对应测试集
    'y_true': y_test,
    'y_pred': y_pred_best
})
pred_file = os.path.join(MODEL_DIR, f"predictions_{timestamp}.csv")
pred_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
print(f"预测结果已保存: {pred_file}")
print("\n 所有任务完成！") 