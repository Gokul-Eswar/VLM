## 2024-03-22 - [CRITICAL] Missing Input Validation in Deployment API
**Vulnerability:** The BentoML deployment endpoints (`track_image`, `track_batch`, `search_tracks`) completely lacked input validation, allowing potentially unlimited memory consumption via oversized images or batches, or database/search stress via long queries.
**Learning:** Even when security limits are documented or "expected" in memory, they might not be implemented in code. "Security by Assumption" is a dangerous pattern. Deployment layers often get less scrutiny than core algorithms.
**Prevention:** Always explicitly validate all inputs at the API boundary (size, length, type) before passing them to internal processing logic. Use integration tests that specifically target boundary conditions.
