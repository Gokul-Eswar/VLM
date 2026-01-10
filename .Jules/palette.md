## 2024-05-23 - [Adaptive Contrast and Transparent Overlays]
**Learning:** Hardcoded white text on randomly generated colored backgrounds often fails accessibility standards (WCAG). Similarly, opaque statistical overlays block critical visual information in video feeds.
**Action:** Always calculate text color based on background luminance (e.g., using Rec. 601 coefficients) and use alpha blending (`cv2.addWeighted`) for informational overlays to maintain context.
