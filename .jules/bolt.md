## 2024-03-24 - [Vectorized IoU Calculation]
**Learning:** Replacing nested Python loops with NumPy broadcasting for IoU (Intersection over Union) calculations yielded a ~15x speedup (0.048s -> 0.003s) for 100x100 comparisons.
**Action:** Always vectorize pairwise matrix operations like IoU, distance matrices, or similarity scores when working with tracking or matching algorithms.
