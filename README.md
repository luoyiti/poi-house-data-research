# 基于POI空间特征的邻里环境与房产价值关系研究

> 城市动态系统多源数据研究项目

**项目作者：** 罗一逖 (231820309)、吴林洁 (231820307)

**项目可视化网页：** [点击访问在线演示](http://localhost:8501/)

**项目仓库：** [GitHub](https://github.com/luoyiti/Urban_Dynamic_System_Multi-Source_Data_Research)

---

## 📋 目录

- [项目简介](#项目简介)
- [研究问题](#研究问题)
- [数据说明](#数据说明)
- [研究方法](#研究方法)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [安装与运行](#安装与运行)
- [主要功能](#主要功能)
- [研究成果](#研究成果)

---

## 📖 项目简介

本项目基于北京市房产数据和POI（Point of Interest，兴趣点）数据，运用空间分析、网络分析、机器学习等多种方法，深入研究邻里环境对房产价值的影响机制。通过整合多源地理空间数据，构建房产-POI网络，揭示城市空间结构与房价分布的内在关系。

## 🔍 研究问题

1. **空间分布特征：** 住房社区的邻里环境（由POI兴趣点表征）呈现怎样的空间分布特征？
2. **价格影响机制：** 不同类型的POI对房产价格有什么样的影响？影响程度如何？
3. **价格预测模型：** 如何利用POI空间特征构建有效的房价预测模型？
4. **市场空间划分：** 如何基于POI特征和空间关系对北京房价市场进行合理划分？


## 📊 数据说明

### 数据来源

- **房产数据：** 北京市二手房交易数据，包含价格、位置、楼层、建筑结构等信息
- **POI数据：** 通过OpenStreetMap（OSM）爬取的北京市兴趣点数据
- **地理数据：** 北京市行政区划、道路网络等地理信息数据

**数据下载链接：** [点击这里](https://box.nju.edu.cn/d/ea1107e0d0f740ffacde/)

### 数据文件说明

#### 1. `house_poi_data.json` - 房产与POI整合数据
```json
{
    "id": "房产唯一标识",
    "price_per_meter": "单价（元/平方米）",
    "lon": "经度",
    "lat": "纬度",
    "pois": [
        {
            "id": "POI标识",
            "first_tag": "POI主要类型",
            "distance": "距离（米）"
        }
    ]
}
```

#### 2. `poi_data.json` - POI兴趣点数据（共计40,887条）
```json
{
    "id": "POI唯一标识",
    "lon": "经度",
    "lat": "纬度",
    "tags": {
        "amenity/shop/tourism/...": "具体类型"
    }
}
```

#### 3. `house_property.csv` - 详细房产信息
- **基础信息：** id, price_per_meter, lon, lat
- **建筑信息：** floor（楼层）, BuildingStructure（建筑结构）
- **区域信息：** town（区域）, community_name（小区）, county（所属区）

#### 4. 其他数据文件
- `house_poi_graph.gml` - 房产-POI网络图数据
- `gdf_houses_clustered.geojson` - 空间聚类结果
- `house_poi_skater_clusters.json` - SKATER聚类结果

## 🔬 研究方法

### 1. 数据预处理
- POI数据爬取与清洗（使用Overpass API）
- 房产数据标准化处理
- 空间距离计算（Haversine公式）
- 数据整合与特征工程

### 2. 特征重要性分析
- **随机森林回归：** 计算各类POI对房价的影响权重
- **相关性分析：** 探索空间特征与房价的关系
- **可视化展示：** 特征重要性排序与热图

### 3. 机器学习预测模型
本研究选择岭回归、Lasso回归、弹性网络、随机森林、梯度提升树、XGBoost、LightGBM、支持向量机和多层感知器共九种机器学习算法进行房价预测建模，比较各模型的预测性能，选择最优模型进行深入分析。  

### 4. 空间聚类分析（SKATER算法）
- **目标：** 基于POI特征和空间邻近性对房产市场进行划分
- **方法：** Spatial K'luster Analysis by Tree Edge Removal
- **权重构建：** 
  - 距离带权重（DistanceBand, 阈值8000米）
  - K近邻权重（KNN, k=1000）
  - 权重融合（w_union）
- **聚类特征：**
  - POI可达性（医疗、教育、商业、交通等）
  - 地理位置（与市中心距离）
  - 建筑特征（楼层、结构等
  - 网络中心性指标）

### 5. 网络分析
- **网络构建：** 房产-POI二部图
- **网络指标：**
  - 度中心性（Degree Centrality）
  - 接近中心性（Closeness Centrality）
  - 介数中心性（Betweenness Centrality）
  - PageRank值
- **网络可视化：** NetworkX + Matplotlib

### 6. 地理空间可视化
- **交互式地图：** Folium热力图、散点图
- **静态地图：** Matplotlib + Contextily底图
- **空间分布图：** GeoPandas + Shapely

## 🛠️ 技术栈

### 核心技术
- **编程语言：** Python 3.13
- **Web框架：** Streamlit
- **数据处理：** Pandas, NumPy
- **地理空间：** GeoPandas, Shapely, Geopy
- **机器学习：** Scikit-learn, XGBoost, LightGBM
- **深度学习：** PyTorch
- **网络分析：** NetworkX
- **空间分析：** LibPySAL, Spopt
- **可视化：** Matplotlib, Seaborn, Folium, Contextily
- **数据爬取：** Overpy (Overpass API)

### 依赖库
详见 [`requirements.txt`](requirements.txt)

## 📁 项目结构

```
Urban_Dynamic_System_Multi-Source_Data_Research/
│
├── requirements.txt                # Python依赖包列表
├── README.md                       # 项目说明文档
│
├── pages/                          # 可视化展示页面
│   ├── image/                      # 生成的图表和地图
│   │   ├── beijing_house_price_map.png
│   │   ├── SKATER_BEIJING.png      # 聚类结果图
│   │   ├── ElbowImage.png          # 肘部法则图
│   │   └── ...                     # 其他数据可视化图片
│   ├── 数据展示.html                # 描述性分析展示
│   ├── 模型预测.html                # 模型预测展示
│   ├── 市场划分.html                # 市场划分展示
│   └── 论文成果.html                # 研究成果与论文展示
│
├── data/                           # 数据文件目录
│   ├── house_data.json             # 房产基础数据
│   ├── poi_data.json               # POI兴趣点数据
│   ├── house_poi_data.json         # 房产-POI整合数据
│   ├── house_property.csv          # 房产详细属性
│   ├── house_poi_graph.gml         # 房产-POI网络图
│   ├── gdf_houses_clustered.*      # 聚类结果（shapefile/geojson）
│   └── beijing/                    # 北京市地理数据（shapefiles）
│
├── prediction/                     # 机器学习预测模块
│   ├── construct_features.py       # 特征构建
│   ├── construct_POI_significance.py  # POI重要性计算
│   ├── calculate_features_weight.py   # 特征权重计算
│   ├── predict_house_prices.py     # 房价预测模型训练
│   ├── visualization.py            # 预测结果可视化
│   ├── visualization_on_map.py     # 地图可视化
│   └── abstract_merged_data.py     # 数据抽象与合并
│
├── view/                           # 可视化脚本
│   ├── house_on_map.py             # 房产地图可视化
│   ├── house_poi_on_map.ipynb      # 房产-POI交互地图
│   ├── house_with_price.ipynb      # 房价分布可视化
│   └── poi_with_price.ipynb        # POI与房价关系可视化
│
├── tools/                          # 工具函数
│   ├── distance_calculator.py      # 距离计算工具
│   └── data_tool.py                # 数据处理工具
│
├── scrapy/                         # 数据爬取脚本
│
└── *.ipynb                         # Jupyter Notebooks
    ├── cluster.ipynb               # 聚类分析
    ├── cluster_analysis.ipynb      # 聚类分析详细版
    ├── network.ipynb               # 网络分析
    ├── poi_net.ipynb               # POI网络构建
    ├── dataProcess.ipynb           # 数据处理
    ├── poi_process.ipynb           # POI数据处理
    ├── poiStratch.ipynb            # POI数据爬取
    └── EncodeInformation.ipynb     # 深度学习编码

```

## 🚀 安装与运行

### 环境要求
- Python 3.8+
- pip 或 conda

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/luoyiti/Urban_Dynamic_System_Multi-Source_Data_Research.git
cd Urban_Dynamic_System_Multi-Source_Data_Research
```

2. **创建虚拟环境（推荐）**
```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 或使用conda
conda create -n urban_research python=3.13
conda activate urban_research
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载数据**
- 从 [数据链接](https://box.nju.edu.cn/d/ea1107e0d0f740ffacde/) 下载数据文件
- 解压到 `data/` 目录


应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

### Jupyter Notebook使用

```bash
jupyter notebook
# 或
jupyter lab
```

## 🎯 主要功能



🏙️ 数据展示[详细报告点击这里](http://localhost:8000/pages/数据展示.html)

住房与POI分布地图

房价热力图

房产-POI关联网络

数据统计分析

🤖 模型预测[详细报告点击这里](http://localhost:8000/pages/模型预测.html)

多模型性能对比

特征重要性分析

预测结果可视化

模型评估指标

🗺️ 市场划分[详细报告点击这里](http://localhost:8000/pages/市场划分.html)

肘部法确定K值

SKATER空间聚类

聚类特征对比

聚类结果可视化

📄 论文成果[详细报告点击这里](http://localhost:8000/pages/论文成果.html)

研究报告展示

可视化图表汇总

数据与参考链接

## 📈 研究成果

### 主要发现

1. **POI类型影响差异显著**
   - 教育设施、医疗设施对高端房价影响最大
   - 公交站点密度与房价呈正相关
   - 旅游景点对特定区域房价有正向影响

2. **空间聚类特征明显**
   - 北京房价市场可划分为6个空间聚类区域
   - 高价区集中在核心城区，具备完善的POI配套
   - 低价区位于远郊，POI可达性较低

3. **网络中心性与房价关系**
   - 网络接近中心性高的房产价格普遍较高
   - 房产-POI网络呈现小世界特征
   - 度中心性反映房产周边配套丰富度

4. **预测模型性能**
   - 集成学习模型（XGBoost, LightGBM）表现最佳
   - 深度学习模型在复杂特征学习上有优势
   - 空间特征显著提升模型预测精度

### 指标体系

#### POI重要性指标

$$score_{i, j}^{raw} = \sum_{k \in POI_j} e^{-\alpha * dist(i,k)}, \text{if } dist(i,k) \leq D_{max}$$

其中：
- 𝑗表示POI类别
- dist(i,k)为房源i到 POIk 的地理距离（单位：公里）
- α为距离衰减系数，设置为0.5
- D_max为最大考虑距离，设置为1公里


#### 嵌入方法

- **Node2Vec：** 将网络拓扑信息转换为高维向量表示
- **H3 Embedding：** 将地理空间信息转换为分层六边形编码

## 📚 相关资源

- **OpenStreetMap：** [https://www.openstreetmap.org](https://www.openstreetmap.org)
- **Overpass API：** [https://overpass-api.de](https://overpass-api.de)
- **LibPySAL：** [https://pysal.org/libpysal](https://pysal.org/libpysal)
- **Streamlit：** [https://streamlit.io](https://streamlit.io)

## 👥 贡献者

- **罗一逖** - 数据处理、POI数据爬取、空间聚类分析、可视化开发
- **吴林洁** - 数据分析、模型构建、论文撰写、可视化优化与演示

## 📄 许可证

本项目仅供学术研究使用。

## 📧 联系方式

如有任何问题或建议，欢迎通过以下方式联系：

- GitHub Issues: [提交问题](https://github.com/luoyiti/Urban_Dynamic_System_Multi-Source_Data_Research/issues)
- Email: [项目邮箱](mailto:231820309@smail.nju.edu.cn)

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**

