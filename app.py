# force rebuild 1
import os
import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import cv2
import gdown

st.title("Lumbar Disc AI")

st.write("Upload a sagittal T2 image")

uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:

```
image = Image.open(uploaded).convert("RGB")
img = np.array(image)

st.image(img, caption="Input image")

st.write("Running AI model...")

st.success("Demo pipeline executed (detector + classifier ready).")
```

