## 2024-05-23 - Vectorized IoU Calculation
**Learning:** Even simple arithmetic like IoU calculation can become a bottleneck when done in nested Python loops (O(N*M)). Vectorizing this using NumPy broadcasting yielded a ~28x speedup (53ms -> 1.8ms for 100x100 matrix).
**Action:** Always look for nested loops over numpy arrays and replace them with broadcasted operations. Be careful with integer truncation - always cast to float32 before division.
