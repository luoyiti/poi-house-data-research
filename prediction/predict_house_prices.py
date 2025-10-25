"""
房价预测多模型训练与评估 - 优化版（无集成）
目标：R² > 0.7
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_style("whitegrid")

warnings.filterwarnings('ignore')

# ========================= 配置 =========================
DATA_FILE = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\final_house_features.csv"
MODEL_DIR = os.path.join(os.path.dirname(DATA_FILE), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42

# ========================= 加载数据 =========================
df = pd.read_csv(DATA_FILE)
print(f" 数据加载成功，样本数: {len(df):,}, 特征数: {len(df.columns)}")

# ========================= 数据预处理 =========================
def preprocess_data(df):
    """数据预处理"""
    # 检查缺失值
    print("缺失值统计:")
    print(df.isnull().sum())
    
    # 移除可能的异常值 (价格过高或过低的极端值)
    price_col = 'price_per_meter'
    Q1 = df[price_col].quantile(0.05)
    Q3 = df[price_col].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_len = len(df)
    df = df[(df[price_col] >= lower_bound) & (df[price_col] <= upper_bound)]
    print(f"移除异常值: {original_len - len(df)} 个样本")
    
    return df

df = preprocess_data(df)

# ========================= 特征工程 =========================
def create_features(df):
    """创建新特征"""
    # 交互特征
    df['size_poi_interaction'] = df['Size_num'] * df['poi_total_score']
    df['bedroom_poi_interaction'] = df['Bedroom'] * df['poi_total_score']
    
    # 综合评分特征
    df['overall_quality_score'] = (
        df['Building_Hardware_Score'] + 
        df['Structure_Score'] + 
        df['Elevator_Floor_Score'] + 
        df['Orientation_Score']
    ) / 4
    
    # 房间总数
    df['total_rooms'] = df['Bedroom'] + df['Washroom'] + df['Livingroom'] + df['Kitchen']
    
    # 面积与房间比例
    df['size_per_room'] = df['Size_num'] / df['total_rooms'].replace(0, 1)
    
    return df

df = create_features(df)

# ========================= 特征与目标 =========================
target = 'price_per_meter'

# 扩展特征集
feature_cols = [
    'Building_Hardware_Score',
    'Structure_Score', 
    'Elevator_Floor_Score',
    'Orientation_Score',
    'StairsUnitRatio_Score',
    'Size_num',
    'Bedroom',
    'Washroom', 
    'Livingroom',
    'Kitchen',
    'poi_total_score',
    'size_poi_interaction',
    'bedroom_poi_interaction',
    'overall_quality_score',
    'total_rooms',
    'size_per_room'
]

# 移除可能包含NaN的特征
feature_cols = [col for col in feature_cols if col in df.columns and not df[col].isnull().any()]

print(f"最终使用特征数: {len(feature_cols)}")
print("特征列表:", feature_cols)

X = df[feature_cols]
y = df[target]

# ========================= 数据集划分 =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=pd.qcut(y, q=5, duplicates='drop')
)
print(f"训练集: {len(X_train):,} 样本, 测试集: {len(X_test):,} 样本")

# ========================= 特征缩放 =========================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================= 优化模型定义 =========================
models = {
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'Lasso': Lasso(random_state=RANDOM_STATE),
    'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    'SVR': SVR(),
    'MLP': MLPRegressor(random_state=RANDOM_STATE, max_iter=1000)
}

# ========================= 超参数调优 =========================
param_grids = {
    'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
    'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10, -1],
        'learning_rate': [0.05, 0.1, 0.15],
        'num_leaves': [31, 63, 127],
        'subsample': [0.8, 0.9, 1.0]
    },
    'SVR': {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'early_stopping': [True]
    }
}

# ========================= 模型训练与评估 =========================
def evaluate_regression(y_true, y_pred):
    """计算回归指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算平均绝对百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

overall_results = {}
best_model = None
best_r2 = -np.inf
tuned_models = {}

print("\n开始模型调优...")
for model_name, model in tqdm(models.items(), desc="调优模型"):
    print(f"\n调优 {model_name} ...")
    
    try:
        if model_name in param_grids:
            # 使用网格搜索调优
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            
            best_model_instance = grid_search.best_estimator_
            tuned_models[model_name] = best_model_instance
            print(f"  最佳参数: {grid_search.best_params_}")
        else:
            # 直接训练
            model.fit(X_train_scaled, y_train)
            best_model_instance = model
            tuned_models[model_name] = best_model_instance
        
        # 预测和评估
        y_pred_test = best_model_instance.predict(X_test_scaled)
        metrics = evaluate_regression(y_test, y_pred_test)
        overall_results[model_name] = metrics
        
        # 交叉验证
        cv_scores = cross_val_score(best_model_instance, X_train_scaled, y_train, 
                                  cv=5, scoring='r2')
        
        print(f"  测试集 R2: {metrics['R2']:.4f}")
        print(f"  交叉验证 R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.1f}%")
        
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = (model_name, best_model_instance)
            
    except Exception as e:
        print(f"模型训练失败: {e}")
        overall_results[model_name] = None

# ========================= 生成详细评估报告 =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = os.path.join(MODEL_DIR, f"optimized_regression_report_{timestamp}.txt")

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("房价预测多模型训练评估报告 (优化版 - 无集成)\n")
    f.write(f"生成时间: {datetime.now()}\n")
    f.write("="*80 + "\n\n")
    
    f.write("【数据概况】\n")
    f.write(f"总样本数: {len(df):,}\n")
    f.write(f"特征数: {len(feature_cols)}\n")
    f.write(f"训练集: {len(X_train):,}\n")
    f.write(f"测试集: {len(X_test):,}\n\n")
    
    f.write("【模型性能对比】\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'模型':<20} {'R2':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}\n")
    f.write("-" * 70 + "\n")
    
    for model_name, metrics in overall_results.items():
        if metrics:
            f.write(f"{model_name:<20} {metrics['R2']:.4f}    {metrics['RMSE']:<10.2f} {metrics['MAE']:<10.2f} {metrics['MAPE']:<8.1f}%\n")
    
    f.write("-" * 70 + "\n\n")
    
    if best_model:
        f.write(f"【最佳模型】{best_model[0]} - R2={best_r2:.4f}\n")
        
        if best_r2 >= 0.7:
            f.write("🎉 目标达成！R² > 0.7\n")
        else:
            f.write("⚠️ 未达到目标 R² > 0.7，建议进一步优化\n")

print(f"\n 报告已生成: {report_file}")

# ========================= 可视化结果 =========================
fig = plt.figure(figsize=(16, 12))

# 1. 模型性能对比
ax1 = plt.subplot(2, 2, 1)
model_names = [name for name, metrics in overall_results.items() if metrics]
r2_scores = [metrics['R2'] for metrics in overall_results.values() if metrics]
colors = ['green' if r2 >= 0.7 else 'orange' for r2 in r2_scores]

bars = plt.bar(range(len(model_names)), r2_scores, color=colors, alpha=0.7)
plt.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='目标线 (R²=0.7)')
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.ylabel('R² Score', fontsize=12)
plt.title('模型性能对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)

# 在柱子上添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. 特征重要性
if best_model and hasattr(best_model[1], 'feature_importances_'):
    ax2 = plt.subplot(2, 2, 2)
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model[1].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.barh(fi.head(10)['feature'], fi.head(10)['importance'], color='skyblue')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 10 特征重要性 - {best_model[0]}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 为特征重要性添加数值标签
    for i, (importance, feature) in enumerate(zip(fi.head(10)['importance'], fi.head(10)['feature'])):
        plt.text(importance, i, f' {importance:.3f}', va='center', fontsize=9)

# 3. 预测 vs 实际值
ax3 = plt.subplot(2, 2, 3)
if best_model:
    y_pred_best = best_model[1].predict(X_test_scaled)
    plt.scatter(y_test, y_pred_best, alpha=0.6, color='steelblue')
    
    # 添加理想线
    max_val = max(y_test.max(), y_pred_best.max())
    min_val = min(y_test.min(), y_pred_best.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('实际价格', fontsize=12)
    plt.ylabel('预测价格', fontsize=12)
    plt.title(f'预测 vs 实际值 (R²={best_r2:.4f})', fontsize=14, fontweight='bold')

# 4. 残差图
ax4 = plt.subplot(2, 2, 4)
if best_model:
    residuals = y_test - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.6, color='coral')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('预测价格', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title('残差分析', fontsize=14, fontweight='bold')

plt.tight_layout()
plot_file = os.path.join(MODEL_DIR, f"optimized_results_{timestamp}.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print(f"结果图已保存: {plot_file}")

# ========================= 保存最佳模型和预处理器 =========================
if best_model:
    model_name, model = best_model
    
    # 保存模型
    model_file = os.path.join(MODEL_DIR, f"best_model_{model_name}_{timestamp}.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'metrics': overall_results[model_name] if model_name in overall_results else None
        }, f)
    
    # 保存预测结果
    y_pred_best = model.predict(X_test_scaled)
    pred_df = pd.DataFrame({
        'id': df.iloc[X_test.index]['id'].values if 'id' in df.columns else X_test.index,
        'y_true': y_test,
        'y_pred': y_pred_best,
        'residual': y_test - y_pred_best
    })
    pred_file = os.path.join(MODEL_DIR, f"predictions_{timestamp}.csv")
    pred_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
    
    print(f"最佳模型已保存: {model_file}")
    print(f"预测结果已保存: {pred_file}")
    
    # 最终结果汇总
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    print(f"最佳模型: {best_model[0]}")
    print(f"测试集 R²: {best_r2:.4f}")
   

print("\n所有任务完成！")