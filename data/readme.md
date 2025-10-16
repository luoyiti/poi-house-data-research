# SHMetro

---

Metro Crowd Flow from Shanghai Metro System. ([PVCGN](https://github.com/HCPLab-SYSU/PVCGN))

## Info

- Time span: 7/1/2016 - 9/30/2016, 05:30~23:30, 73 intervals
- Number of stations: 288
- Interval: 15min
- Feature: inflow, outflow
- (train, val, test) split: (62, 9, 21)
- (in_len, out_len): (4, 4)

## Data Description

- `train.pkl, val.pkl, test.pkl`: dict with 4 keys 
  ```
  'x': ndarray, (Batch, 4, 288, 2), float64, flow data
  'y': ndarray, (Batch, 4, 288, 2), float64, flow data
  'xtime': ndarray, (Batch, 4), datetime64[ns], datetime data
  'ytime': ndarray, (Batch, 4), datetime64[ns], datetime data
  ```
  The `Batch` is `4092, 594, 1386` for `train, val, test`, respectively. 
- `graph_sh_conn.pkl`: 0-1 connectivity graph with self loop  
  ```
  ndarray, (288, 288), float64, symmetric, 1's on diagonal
  total 958 directed edges (including self loops; an undirected edge is equivalent to two directed edges)
  ```
- `graph_sh_sml.pkl`: similarity graph, `S(i, j) = exp(-DTW(xi, xj))` computed from train set  
  ```
  ndarray, (288, 288), float64, asymmetric, 1's on diagonal
  the number of nonzero values in each column is 10, since the top-10 values are chosen,
  smaller values are filtered (set to 0)
  note that S(i, j) is viewed as the weight from j to i
  total 2880 directed edges (including self-loops)
  ```
- `graph_sh_cor.pkl`: correlation graph, `C(i, j) = D(i, j)/sum_k D(i, k)`,   
  `D(i, j)` means the number of passengers traveling from j to i, computed from train set
  ```
  ndarray, (288, 288), float64, asymmetric, diagonal (some zero and non-zero values)
  the number of nonzero values in each column is 10, since the top-10 values are chosen,
  smaller values are filtered (set to 0)
  note that C(i, j) is viewed as the weight from j to i
  total 2880 directed edges
  ```

## Load Data

```python
import pickle
with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
```