## 2024-03-24 - [Vectorized IoU Calculation]
**Learning:** Nested Python loops for matrix operations (like IoU) are a significant performance bottleneck in tracking systems, especially as the number of objects increases.
**Action:** Always prefer NumPy broadcasting for $O(N \times M)$ pairwise calculations. Ensure input data (like trackers) is converted to NumPy arrays before broadcasting to avoid type errors.
