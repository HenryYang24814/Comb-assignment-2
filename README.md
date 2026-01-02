# A2

---

## Q3
```
# Main Algo
def tsp_app(G: n*n)
require: G satisfies the triangle inequality

τ ← MST(G);
σ ← DFS(G);
return σ;

ensure: DIST(TSP(G), G) ≤ DIST(σ, G) ≤ 2 * DIST(TSP(G), G)
```

### testing
```bash
python tsp_app.py q3_test.txt
```
---
### Used file
- `mst.py`
- `DFS.py`
- `tsp_app.py`(Main)

### calculate_tour_weight(helper method in tsp_app.py)
将返回的permutation加上start_point(index 0) 得到Hamiltonian cycle 并计算total weight

### TODO

- [ ] 完成MST和DFS伪代码以及解释
- [ ] `calculate_tour_weight`这个method可以模块化, 用于Q2?