## 2024-03-24 - [Accessible Text Contrast in Computer Vision]
**Learning:** Computer vision overlays often place text on dynamically colored bounding boxes. Standard white text becomes illegible on light backgrounds (yellow, cyan). Using Rec. 601 luminance coefficients (`0.299*R + 0.587*G + 0.114*B`) to dynamically switch between black and white text ensures readability regardless of the random track ID color.
**Action:** Always implement a `get_contrast_color(bg_color)` helper when visualizing data on dynamic backgrounds.
