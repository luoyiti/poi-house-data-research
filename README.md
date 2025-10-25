# 基于POI空间特征的邻里环境与房产价值关系研究

项目可视化网页:[点击这里](https://poi-houses-research-9yja5cw77orgedwvsnqs2k.streamlit.app/)

## 研究数据

房产价格数据 + poi数据整合

house_poi_data.json 数据包括

    "id": house_id,

    "price_per_meter": price,

    "lon" 纬度数据

    "lat" 经度数据

    "pois" 包括所有附近一公米范围内的poi兴趣点(已去除住房兴趣点)

其中 poi 数据还被保存在poi_data.json中(共计40887条), 包括

    id 独特的标识

    lon 纬度数据

    lat 经度数据

    tags poi特征数据, 根据poi类型不同而内容不同，需要额外处理

更详细的住房信息保存在house_data.json中，包括

    id, price_per_meter, lon, lat

    floor 楼层信息, town 区域名称, community_name 小区名称, county 所属区名

数据保存地:[点击这里](https://box.nju.edu.cn/d/ea1107e0d0f740ffacde/)

## 研究问题

住房社区的邻居环境(poi兴趣点代表)呈现怎样的分布特征，其对于房产价格有什么样的影响？

## 研究方法

1. 设置poi节点指标，计算poi重要性权重

2. 运用机器学习方法，根据poi点重要性预测房产价格

3. 运用机器学习模型，根据所有poi兴趣点，划分房产市场

聚类分析: 提取附近poi兴趣点特征，对房产进行分类

机器学习方法: 根据决策树等方法，构造价格预测模型(例如将房产附近拥有的地标输入，生成一个可能的价格)

网络分析: 将多个poi兴趣点、多个住房点连接成一个大型网络，采用pagerank算法、图神经网络等方式预测房产价格
(例如，就pagerank算法而言，一个社区内的住房点更可能有相同的住房价格，于是每一个节点有其自己的价格特征，邻近节点之间存在网络连接，连接越多高价格的房产，则其自身越有可能有更高的价格)

## 指标设置

poi指标:

$Importance_{POI} = w_A * A + w_b * B + w_c * C$

Importance代表节点的价值重要性，它由节点附近的房产价格推导得出

A 代表着POI点的类型重要性

B 代表着POI点的地理位置中心性

C 代表着POI点的网络接近中心性

其中, A, B, C 均是归一化后的参数

Node2Vec 将网络信息转换为高维向量

H3 Embedding 将地理信息转换为高维向量

