import streamlit as st

st.set_page_config(
    page_title="基于POI空间特征的邻里环境与房产价值关系研究",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("基于POI空间特征的邻里环境与房产价值关系研究")

st.markdown("""
            
罗一逖231820309

吴林洁231820307

> 项目仓库: [点击这里](https://github.com/luoyiti/Urban_Dynamic_System_Multi-Source_Data_Research)

## 功能介绍

- 📊 数据统计：查看和分析数据

- 🔮 模型构建：使用机器学习模型

""")

# 可选：添加一些首页内容
col1, col2 = st.columns(2)

with col1:
    st.info("数据分析", icon="📊")
    st.write("探索你的数据")
    
with col2:
    st.info("模型预测", icon="🔮")
    st.write("AI 预测功能")
    