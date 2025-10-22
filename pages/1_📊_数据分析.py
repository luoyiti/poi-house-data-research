import streamlit as st
import streamlit.components.v1 as components
    

# 完整的 HTML 页面
html_code = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        button {
            background-color: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #764ba2;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>🎯 交互式 HTML 组件</h1>
        <p>这是一个完整的 HTML 页面</p>
        <button onclick="alert('Hello from HTML!')">点击我</button>
    </div>
</body>
</html>
"""

# 嵌入 HTML，设置高度
components.html(html_code, height=300)