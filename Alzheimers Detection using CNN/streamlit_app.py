import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('model.h5')

# Define the image size for model input
IMG_SIZE = (128, 128)

# Custom CSS for aesthetics
st.markdown(
    """
    <style>
    .title {
        margin-top:0px;
        color: #FF5733; /* Coral */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .text {
        color: #EFA18A; /* Slate Gray */
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .uploaded-image {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction {
        color: #FF5733; /* Coral */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .confidence {
        color: #FF5600; /* Coral */
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the title and description
st.markdown("<h1 class='title'>Alzheimer's Detection using CNN</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='text'>Upload a brain ultrasound image to predict the presence of Alzheimer's disease using a pre-trained deep learning model.</h2>", unsafe_allow_html=True)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload an image.")

def preprocess_image(image):
    """Preprocess the image to fit the model input requirements."""
    # Convert image to RGB
    img_array = np.array(image.convert('RGB'))
    
    # Resize image to target size
    img_array = tf.image.resize(img_array, IMG_SIZE)
    
    # Normalize image to range [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    """Predict the class of the uploaded image using the pre-trained model."""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# Display the file uploader
uploaded_file = st.sidebar.file_uploader(label="Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        predicted_idx = predict(image)
        
        class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
        predicted_label = class_labels[predicted_idx]
        
        st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.sidebar.write("Please upload an image to make a prediction.")
