import streamlit as st
import streamlit.components.v1 as components

from PIL import Image

# 从文件读取图片

st.markdown("## 住房与POI兴趣点分布地图")
img = Image.open("./image/beijing_house_poi_map.png")
st.image(img, caption="住房与POI兴趣点分布地图", use_container_width=True)

st.markdown("## 北京房价分部地图")
# 显示图片
img = Image.open("./image/beijing_house_price_map.png")
st.image(img, caption="北京房价分部地图", use_container_width=True)

st.markdown("## 住房-POI兴趣点网络图")
img = Image.open("./image/yeast_networkx.png")
st.image(img, caption="总网络", use_container_width=True)
img = Image.open("./image/subgraph_spring.png")
st.image(img, caption="子网络(提取5000个节点)", use_container_width=True)