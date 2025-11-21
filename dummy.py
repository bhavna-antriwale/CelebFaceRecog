import streamlit as st
import gdown
import os
from tensorflow.keras.models import load_model
import pandas as pd

H5_FILE_PATH = "vgg_face_weights.h5"
# The Google Drive File ID of your .h5 file
GOOGLE_DRIVE_FILE_ID = '14EArmcPAKIWR3RDM1zye9Fqa0NsLgM-t'

@st.cache_resource
def download_and_load_model(file_id, output_path):
    """Downloads the .h5 file from Google Drive to use as weights"""
    if not os.path.exists(output_path):
        with st.spinner("Downloading weightsfrom Google Drive..."):
            # Use gdown to download the file by ID
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            st.success("weights downloaded successfully!")
    
    # Load the model using Keras/TensorFlow
    model = load_model(output_path)
    return model

st.title("weights file from google drive")

model = download_and_load_model(GOOGLE_DRIVE_FILE_ID, H5_FILE_PATH)
st.write("weights file loaded and ready for use!")
