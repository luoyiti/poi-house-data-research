import os
import glob
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

with open("archive/POI_House.pdf", "rb") as f:
    pdf_bytes = f.read()

st.markdown("# 论文成果展示")

pdf_viewer(
    pdf_bytes,
    height=850,
    width=0,            # 0 表示自适应容器宽度
    # pages_to_render=None,   # 大文件可指定如 [1,2,3] 分页渲染
    # render_text=True,
)