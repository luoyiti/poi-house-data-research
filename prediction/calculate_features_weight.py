import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_building_hardware_weights(df):
    """
    计算建筑硬件特征的最优权重（只包含电梯+楼层组合、结构、朝向、梯户比）
    这里只在最后使用了自动赋权，是因为都是文本数据，转化成数值数据之前需要人工赋值
    删除线性回归方法，只使用相关性和随机森林，优化零权重和负相关问题
    """
    print("=== 计算建筑硬件特征最优权重 ===\n")
    
    # 1. 创建基础评分特征
    def create_basic_scores(df):
        # 建筑结构评分
        def get_structure_score(structure):
            mapping = {
                '钢混结构': 0.95,
                '混合结构': 0.75,
                '砖混结构': 0.45,
                '砖木结构': 0.25
            }
            return mapping.get(structure, 0.5)
        
        df['Structure_Score'] = df['BuildingStructure'].apply(get_structure_score)
        
        # 电梯+楼层组合评分
        def get_elevator_floor_score(elevator, floor_level):
            if elevator == 1:  # 有电梯
                if floor_level in ['中楼层', '高楼层', '顶层']:
                    return 1.0
                elif floor_level in ['低楼层', '底层']:
                    return 0.7
                else:
                    return 0.5
            elif elevator == 0:  # 无电梯
                if floor_level in ['底层', '低楼层']:
                    return 0.8
                elif floor_level == '中楼层':
                    return 0.5
                elif floor_level in ['高楼层', '顶层']:
                    return 0.3
                else:
                    return 0.5
            else:
                return 0.5
        
        df['Elevator_Floor_Score'] = df.apply(
            lambda row: get_elevator_floor_score(row['Elevator'], row['Floor_level']), axis=1)
        
        # 朝向评分
        def get_orientation_score(only_south, south_north):
            if only_south == 1 and south_north == 0:
                return 0.75
            elif only_south == 0 and south_north == 1:
                return 0.95
            elif only_south == 0 and south_north == 0:
                return 0.40
            else:
                return 0.5
        
        df['Orientation_Score'] = df.apply(
            lambda row: get_orientation_score(row['Only_South'], row['South_North']), axis=1)
        
        # 梯户比原值
        df['StairsUnitRatio_Original'] = df['StairsUnitRatio_num']
        
        return df
    
    # 创建基础评分
    df = create_basic_scores(df)
    
    # 2. 准备硬件特征
    features = ['Structure_Score', 'Elevator_Floor_Score', 'Orientation_Score', 'StairsUnitRatio_Original']
    feature_names = {
        'Structure_Score': '建筑结构',
        'Elevator_Floor_Score': '电梯楼层组合',
        'Orientation_Score': '房屋朝向',
        'StairsUnitRatio_Original': '梯户比密度'
    }
    
    X = df[features].copy()
    y = df['price_per_meter']
    X = X.fillna(X.median())
    
    analysis_results = []
    analysis_results.append("=== 建筑硬件特征权重分析 ===\n")
    analysis_results.append(f"样本数量: {len(X)}")
    
    # 3. 方法一：相关性分析（处理NaN与负号）
    analysis_results.append("\n--- 方法一：基于相关性分析 ---")
    correlations = X.corrwith(y)
    
    # 修正NaN与负值
    correlations = correlations.fillna(0.01)
    abs_correlations = correlations.abs()
    abs_correlations = abs_correlations + 0.01  # 防止除零
    
    correlation_weights = abs_correlations / abs_correlations.sum()
    
    for feature, corr in correlations.items():
        analysis_results.append(f"  {feature_names[feature]} 与房价相关性: {corr:.4f}")
    
    # 4. 方法二：随机森林重要性
    analysis_results.append("\n--- 方法二：基于随机森林特征重要性 ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_importance = pd.Series(rf_model.feature_importances_, index=features)
    # 避免出现完全为0
    rf_importance = rf_importance + 1e-4
    rf_weights = rf_importance / rf_importance.sum()
    
    for feature, imp in rf_importance.sort_values(ascending=False).items():
        analysis_results.append(f"  {feature_names[feature]} 重要性: {imp:.4f}")
    
    # 5. 综合权重（防止为0 & 平滑）
    analysis_results.append("\n--- 综合权重分配（推荐） ---")
    weights_df = pd.DataFrame({
        'Correlation': correlation_weights,
        'RandomForest': rf_weights
    })
    
    weights_df['Final_Weight'] = weights_df.mean(axis=1)
    
    # Softmax平滑
    exp_weights = np.exp(weights_df['Final_Weight'])
    final_weights = exp_weights / exp_weights.sum()
    final_weights = pd.Series(final_weights.values, index=features)
    
    for feature, weight in final_weights.sort_values(ascending=False).items():
        analysis_results.append(f"  {feature_names[feature]}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 6. 综合评分（正向化处理）
    stairs_scaled = (df['StairsUnitRatio_Original'] - df['StairsUnitRatio_Original'].min()) / (
        df['StairsUnitRatio_Original'].max() - df['StairsUnitRatio_Original'].min())
    stairs_normalized = 1 - stairs_scaled  # 负向指标反转
    
    df['StairsUnitRatio_Score'] = stairs_normalized
    df['Building_Hardware_Score'] = (
        df['Structure_Score'] * final_weights['Structure_Score'] +
        df['Elevator_Floor_Score'] * final_weights['Elevator_Floor_Score'] +
        df['Orientation_Score'] * final_weights['Orientation_Score'] +
        stairs_normalized * final_weights['StairsUnitRatio_Original']
    )
    
    # 7. 输出相关性结果
    hardware_corr = df['Building_Hardware_Score'].corr(df['price_per_meter'])
    if hardware_corr < 0:
        df['Building_Hardware_Score'] = 1 - df['Building_Hardware_Score']
        hardware_corr = df['Building_Hardware_Score'].corr(df['price_per_meter'])
    analysis_results.append(f"\n建筑硬件综合评分与房价相关性: {hardware_corr:.4f}")
    
    # 8. 绘制权重图
    plt.figure(figsize=(6, 4))
    final_weights.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title("建筑硬件特征最终权重（平滑修正后）")
    plt.xlabel("权重值")
    plt.tight_layout()
    plt.show()
    
    for line in analysis_results:
        print(line)
    
    return final_weights.to_dict(), weights_df, hardware_corr, df, analysis_results


def save_final_dataset(df, optimal_weights, output_path):
    print("\n=== 保存最终数据集 ===")
    required_columns = [
        'id', 'price_per_meter',
        'Building_Hardware_Score',
        'Structure_Score', 'Elevator_Floor_Score',
        'Orientation_Score', 'StairsUnitRatio_Score',
        'Size_num', 'Bedroom', 'Washroom', 'Livingroom', 'Kitchen'
    ]
    
    poi_columns = [col for col in df.columns if 'POI' in col or 'poi' in col or '显著' in col]
    if poi_columns:
        required_columns.extend(poi_columns)
        print(f"找到POI相关列: {poi_columns}")
    
    final_columns = [col for col in required_columns if col in df.columns]
    final_df = df[final_columns].copy()
    
    print(f"最终数据集形状: {final_df.shape}")
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"精简数据集已保存至: {output_path}")
    
    return final_df


def save_analysis_to_txt(analysis_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in analysis_results:
            f.write(line + '\n')
    print(f"分析结果已保存至: {output_path}")


def read_data_with_encoding(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            print(f"尝试使用 {encoding} 编码读取文件...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except Exception:
            continue
    print("尝试使用错误忽略方式读取文件...")
    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    print("使用错误忽略方式成功读取文件")
    return df


def main():
    file_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\house_predicting_features.csv"
    print("=== 开始处理建筑硬件特征 ===")
    print(f"文件路径: {file_path}")
    
    df = read_data_with_encoding(file_path)
    print(f"原始数据形状: {df.shape}")
    
    optimal_weights, weights_details, hardware_corr, df_with_scores, analysis_results = calculate_building_hardware_weights(df)
    
    output_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\final_house_features.csv"
    final_dataset = save_final_dataset(df_with_scores, optimal_weights, output_path)
    
    analysis_output_path = r"D:\HuaweiMoveData\Users\32549\OneDrive\大数据小组作业\hardware_analysis_results.txt"
    save_analysis_to_txt(analysis_results, analysis_output_path)
    
    print(f"\n=== 处理完成 ===")
    print(f"建筑硬件综合评分与房价相关性: {hardware_corr:.4f}")
    
    return final_dataset, optimal_weights


if __name__ == "__main__":
    final_dataset, weights = main()
