import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import cv2
import PIL
import zipfile
import os
import warnings
# suppress display of warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

st.title("Celebrity Face detection model")
upload_image = st.file_uploader('Upload an image (.jpg) file: ', type=["jpg", "jpeg"])
if upload_image is not None:
    # Open the uploaded file as an image using PIL
    image = Image.open(upload_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    st.write("Please upload an image file.")

