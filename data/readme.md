# HZMetro

---

Metro Crowd Flow from Hangzhou Metro System. ([PVCGN](https://github.com/HCPLab-SYSU/PVCGN))

## Info

- Time span: 1/1/2019 - 1/25/2019, 05:30~23:30, 73 intervals
- Number of stations: 80
- Interval: 15min
- Feature: inflow, outflow
- (train, val, test) split: (18, 2, 5)
- (in_len, out_len): (4, 4)

## Data Description

- `train.pkl, val.pkl, test.pkl`: dict with 4 keys 
  ```
  'x': ndarray, (Batch, 4, 80, 2), float32, flow data
  'y': ndarray, (Batch, 4, 80, 2), float32, flow data
  'xtime': ndarray, (Batch, 4), datetime64[ns], datetime data
  'ytime': ndarray, (Batch, 4), datetime64[ns], datetime data
  ```
  The `Batch` is `1188, 132, 330` for `train, val, test`, respectively. 
- `graph_hz_conn.pkl`: 0-1 connectivity graph with self loop  
  ```
  ndarray, (80, 80), int64, symmetric, 1's on diagonal
  total 248 directed edges (including self loops; an undirected edge is equivalent to two directed edges)
  ```
- `graph_hz_sml.pkl`: similarity graph, `S(i, j) = exp(-DTW(xi, xj))` computed from train set  
  ```
  ndarray, (80, 80), float64, symmetric, 1's on diagonal
  values < 0.1 are already filtered (set to 0)
  total 2502 directed edges (including self loops; an undirected edge is equivalent to two directed edges)
  ```
- `graph_hz_cor.pkl`: correlation graph, `C(i, j) = D(i, j)/sum_k D(i, k)`,   
  `D(i, j)` means the number of passengers traveling from j to i, computed from train set
  ```
  ndarray, (80, 80), float64, asymmetric, diagonal (some zero and non-zero values)
  values < 0.02 are already filtered (set to 0)
  total 1094 directed edges
  ```
## Load Data

```python
import pickle
with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
```

# 中文版数据介绍

https://mp.weixin.qq.com/s/_J_dIgGv89Iod_oDqrihnw
