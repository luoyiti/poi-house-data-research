import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import pandas as pd

st.markdown("## 房产价值预测模型构建")

st.markdown("""
            本研究选择岭回归、Lasso回归、弹性网络、随机森林、梯度提升树、XGBoost、LightGBM、
            支持向量机和多层感知器共九种机器学习算法进行房价预测建模。
            """)

# 创建POI权重数据框
poi_weights_data = {
    'POI类别': ['amenity', 'shop', 'tourism', 'leisure', 'healthcare', 
                'Public_transport', 'railway', 'aeroway', 'sport', 'education', 'office'],
    '先验权重': [0.8, 0.7, 0.6, 0.6, 1.0, 1.0, 0.8, 0.3, 0.5, 1.0, 0.6]
}

df_poi_weights = pd.DataFrame(poi_weights_data)

st.markdown("### POI兴趣点先验权重表")
st.dataframe(df_poi_weights, hide_index=True)

# 创建房屋结构类型数据框
structure_data = {
    '特征取值': ['钢混结构', '混合结构', '砖混结构', '砖木结构', '其他/未知'],
    '得分': [0.95, 0.75, 0.45, 0.25, 0.50],
    '说明': ['稳定性最佳、抗震性更好', '稳定性较好', '稳定性一般', '稳定性较差', '/']
}

df_structure = pd.DataFrame(structure_data)

st.markdown("### 建筑结构人工赋值表")
st.dataframe(df_structure, hide_index=True)

# 创建电梯与楼层组合评分数据框
elevator_floor_data = {
    '特征取值': ['有电梯 + 中/高楼层/顶层', '有电梯 + 低楼层/底层', '无电梯 + 底层/低楼层', 
                '无电梯 + 中楼层', '无电梯 + 高楼层/顶层', '其他情况'],
    '得分': [1.00, 0.70, 0.80, 0.50, 0.30, 0.50],
    '说明': ['电梯便利性与视野采光最佳组合', '有电梯但楼层较低便利性稍差', '无电梯但楼层低步行便利',
            '无电梯中等楼层步行一般', '无电梯高楼层步行不便', '/']
}

df_elevator_floor = pd.DataFrame(elevator_floor_data)

st.markdown("### 电梯楼层组合人工赋值表")
st.dataframe(df_elevator_floor, hide_index=True)

# 创建房屋朝向评分数据框
orientation_data = {
    '特征取值': ['南北通透', '纯南向', '其他朝向', '其他情况'],
    '得分': [0.95, 0.75, 0.40, 0.25],
    '说明': ['通风采光最佳,市场最受欢迎', '采光好但通风稍差', '采光通风相对较差', '/']
}

df_orientation = pd.DataFrame(orientation_data)

st.markdown("### 房屋朝向人工赋值表")
st.dataframe(df_orientation, hide_index=True)

# 创建模型参数表格
st.markdown("### 机器学习模型关键参数配置")

model_params_data = {
    '模型名称': [
        '岭回归',
        'Lasso回归',
        '弹性网络',
        '弹性网络',
        '随机森林',
        '随机森林',
        '随机森林',
        '随机森林',
        '梯度提升树',
        '梯度提升树',
        '梯度提升树',
        '梯度提升树',
        'XGBoost',
        'XGBoost',
        'XGBoost',
        'XGBoost',
        'XGBoost',
        'LightGBM',
        'LightGBM',
        'LightGBM',
        'LightGBM',
        'LightGBM',
        '支持向量机回归',
        '支持向量机回归',
        '支持向量机回归',
        '多层感知器',
        '多层感知器',
        '多层感知器',
        '多层感知器'
    ],
    '关键参数': [
        'alpha',
        'alpha',
        'alpha',
        'l1_ratio',
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'n_estimators',
        'learning_rate',
        'max_depth',
        'subsample',
        'n_estimators',
        'max_depth',
        'learning_rate',
        'subsample',
        'colsample_bytree',
        'n_estimators',
        'max_depth',
        'learning_rate',
        'num_leaves',
        'subsample',
        'C',
        'kernel',
        'gamma',
        'hidden_layer_sizes',
        'alpha',
        'learning_rate',
        'early_stopping'
    ],
    '参数含义说明': [
        'L2正则化强度',
        'L1正则化强度',
        '总体正则化强度',
        'L1与L2正则化比例',
        '树的数量',
        '树的最大深度',
        '内部节点分裂最小样本数',
        '叶节点最小样本数',
        '弱学习器数量',
        '学习率',
        '树的最大深度',
        '样本采样比例',
        '树的数量',
        '树的最大深度',
        '学习率',
        '样本采样比例',
        '特征采样比例',
        '树的数量',
        '树的最大深度(-1表示无限制)',
        '学习率',
        '叶节点数量',
        '样本采样比例',
        '惩罚系数',
        '核函数类型',
        '核函数系数',
        '隐藏层结构',
        'L2正则化系数',
        '学习率策略',
        '早停法启用'
    ],
    '参数取值范围': [
        '[0.1, 1.0, 10.0, 100.0]',
        '[0.001, 0.01, 0.1, 1.0]',
        '[0.01, 0.1, 1.0]',
        '[0.2, 0.5, 0.8]',
        '[100, 200, 300]',
        '[10, 15, 20, None]',
        '[2, 5, 10]',
        '[1, 2, 4]',
        '[100, 200, 300]',
        '[0.05, 0.1, 0.15]',
        '[3, 5, 7]',
        '[0.8, 0.9, 1.0]',
        '[100, 200, 300]',
        '[6, 8, 10]',
        '[0.05, 0.1, 0.15]',
        '[0.8, 0.9, 1.0]',
        '[0.8, 0.9, 1.0]',
        '[100, 200, 300]',
        '[6, 8, 10, -1]',
        '[0.05, 0.1, 0.15]',
        '[31, 63, 127]',
        '[0.8, 0.9, 1.0]',
        '[0.1, 1.0, 10.0, 100.0]',
        "['rbf', 'linear']",
        "['scale', 'auto']",
        '[(50,), (100,), (50,50), (100,50)]',
        '[0.0001, 0.001, 0.01]',
        "['constant', 'adaptive']",
        '[True]'
    ]
}

df_model_params = pd.DataFrame(model_params_data)

st.dataframe(df_model_params, hide_index=True, use_container_width=True)

st.markdown("""
**说明**:
- 所有模型均采用网格搜索(GridSearchCV)进行超参数调优
- 使用5折交叉验证评估模型性能
- 评估指标包括:MAE(平均绝对误差)、MSE(均方误差)、RMSE(均方根误差)、R²(决定系数)、MAPE(平均绝对百分比误差)
""")

# 创建模型评估结果表格
st.markdown("### 模型评估结果对比")

model_results_data = {
    '模型名称': [
        '岭回归',
        'Lasso回归',
        '弹性网络',
        '随机森林',
        '梯度提升树',
        'XGBoost',
        'LightGBM',
        'SVR',
        'MLP'
    ],
    'R²': [
        0.0287,
        0.0283,
        0.0288,
        0.7292,
        0.7195,
        0.7245,
        0.6955,
        0.0919,
        0.1549
    ],
    'RMSE': [
        9860.26,
        9862.74,
        9860.02,
        5206.74,
        5298.55,
        5251.65,
        5521.20,
        9534.51,
        9197.77
    ],
    'MAE': [
        7951.21,
        7956.58,
        7950.35,
        3412.02,
        3535.69,
        3500.13,
        3688.06,
        7450.14,
        7300.37
    ],
    'MAPE': [
        37.2,
        37.1,
        37.2,
        15.0,
        15.6,
        15.5,
        16.3,
        32.6,
        33.1
    ],
    '交叉验证R²均值': [
        0.0283,
        0.0280,
        0.0285,
        0.7285,
        0.7190,
        0.7240,
        0.6950,
        0.0915,
        0.1540
    ],
    '交叉验证R²标准差': [
        0.0084,
        0.0085,
        0.0084,
        0.0123,
        0.0118,
        0.0121,
        0.0135,
        0.0102,
        0.0156
    ]
}

df_model_results = pd.DataFrame(model_results_data)

# 使用样式突出显示最佳结果


st.dataframe(df_model_results, hide_index=True, use_container_width=True)

image = Image.open("./image/model_performance_comparison.jpeg")
st.image(image, caption="模型评估结果对比图", use_container_width=True)

image = Image.open("./image/predicted_vs_actual_values.jpeg")
st.image(image, caption="预测值与实际值散点图", use_container_width=True)

image = Image.open("./image/residual_analysis.jpeg")
st.image(image, caption="残差分析图", use_container_width=True)