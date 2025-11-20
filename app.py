import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import zipfile
import os
import warnings
import vggface
# suppress display of warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm


st.title("Celebrity Face detection model")
upload_image = st.file_uploader('Upload an image (.jpg) file: ', type=["jpg", "jpeg"])
if upload_image is not None:
    # Open the uploaded file as an image using PIL
    image = Image.open(upload_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
else:
    st.write("Please upload an image file.")

# Create model to generate embeddings for input image
embeddings_model = vgg_face()
embeddings_model.load_weights('https://drive.google.com/file/d/14EArmcPAKIWR3RDM1zye9Fqa0NsLgM-t/view?usp=sharing')

from tensorflow.keras.models import Model
vgg_face_descriptor = Model(inputs=embeddings_model.layers[0].input, outputs=embeddings_model.layers[-2].output)

# Generate embeddings for input image
img = BGRTORGB(img_path)
img = (img / 255.).astype(np.float32)
img = cv2.resize(img, dsize = (224,224))
embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
test_embeddings=np.array(embedding_vector)

StdScaler = StandardScaler()
pca = PCA(n_components=128)
x_test_img_std = StdScaler.transform(test_embeddings)
x_test_img_pca = pca.transform(x_test_img_std)
Test = np.expand_dims(x_test_img_pca, axis=0)
st.write("input image processed successfully")




