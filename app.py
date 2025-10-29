import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="基于POI空间特征的邻里环境与房产价值关系研究",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("基于POI空间特征的邻里环境与房产价值关系研究")

st.markdown("""
            
罗一逖 231820309

吴林洁 231820307

> 项目仓库: [点击这里](https://github.com/luoyiti/Urban_Dynamic_System_Multi-Source_Data_Research)

## 功能介绍

- 📊 数据分析：数据描述与分析

- 🔮 模型预测：数学建模与预测

""")

# 可选：添加一些首页内容
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("数据分析")
    st.write("数据描述与分析")
    
with col2:
    st.info("模型预测")
    st.write("数学建模与预测")

with col3:
    st.info("市场划分")
    st.write("基于空间聚类的市场划分")

with col4:
    st.info("论文成果")
    st.write("研究论文与报告")

@st.cache_data
def read_md(path: str, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)

md_path = "README.md"  # 你的本地路径
content = read_md(md_path)
st.markdown(content)  # 直接渲染