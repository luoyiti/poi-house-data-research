import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import pandas as pd

st.markdown("# 基于空间聚类的北京房价市场划分")

st.markdown("## 空间聚类肘部图")

image = Image.open("image/ElbowImage.png")
st.image(image, caption="肘部法则图", use_container_width=True)

st.markdown("## 北京房价市场空间聚类结果")

image = Image.open("image/SKATER_BEIJING.png")
st.image(image, caption="北京房价市场空间聚类结果", use_container_width=True)

st.markdown("## 各聚类区域特征分析")

# 创建DataFrame
data = {
    '标识符': [0, 1, 2, 3, 4, 5],
    '主要影响因素': [
        '医疗设施稀缺, 旅游景点丰富, 商业设施丰富',
        '医疗设施丰富, 旅游景点丰富, 教育资源丰富',
        '商业设施丰富, 建筑结构好, 旅游资源稀缺',
        '商业设施稀缺, 医疗设施稀缺, 距离市中心远',
        '距离市中心远, 节点接近中心性低, 医疗设施稀缺',
        '距离市中心远, 商业设施稀缺, 医疗设施稀缺'
    ],
    '平均房产价格': [29101, 35737, 27356, 14453, 18455, 15316]
}

df = pd.DataFrame(data)

# 格式化价格列,添加千位分隔符
df['平均房产价格'] = df['平均房产价格'].apply(lambda x: f"{x:,}")

# 显示表格
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("## 北京市房价市场划分")
image = Image.open("image/SKATER_BEIJING_filtered_convex_hull.png")
st.image(image, caption="北京市房价市场划分", use_container_width=True)
